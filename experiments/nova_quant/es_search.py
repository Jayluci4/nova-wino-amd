"""Evolution Strategy for NOVA-Quant rotation discovery.

Follows the NOVA pattern: ES search in a structured space -> evaluate on hardware
-> select elites -> mutate -> repeat. The key difference from NOVA's Winograd
point search is that the genome encodes rotation *structure* (block sizes, sign
vectors) rather than continuous coordinates.

Mutation operators:
- Sign flip: flip k random bits in a stage's sign vector
- Block size change: change a stage's block size to adjacent power of 2
- Stage add/remove: add or remove a rotation stage (rare)

Crossover:
- Uniform sign crossover: per-bit selection from two parents
- Stage swap: take structure from one parent, signs from another
"""

import torch
import random
import time
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass, field

from .rotation import NOVARotation, RotationStage
from .cost_function import RotationMetrics


# Valid block sizes for MI300X (powers of 2, cache-line aligned)
BLOCK_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


@dataclass
class SearchConfig:
    """Configuration for the ES search."""
    dim: int = 8192
    pop_size: int = 32
    n_generations: int = 200
    n_restarts: int = 5
    n_stages_range: Tuple[int, int] = (1, 3)
    block_sizes: List[int] = field(default_factory=lambda: BLOCK_SIZES)
    sign_mutation_rate: float = 0.02
    block_mutation_prob: float = 0.1
    stage_mutation_prob: float = 0.02
    elite_fraction: float = 0.25
    crossover_prob: float = 0.7
    seed: int = 42


@dataclass
class SearchResult:
    """Result of an ES search run."""
    best_rotation: NOVARotation
    best_cost: float
    best_metrics: RotationMetrics
    history: List[dict] = field(default_factory=list)
    total_evaluations: int = 0
    total_time_s: float = 0.0


class NOVAQuantES:
    """Evolution Strategy for discovering optimal rotation matrices.

    The NOVA twist: the search space is structured rotations (cascaded
    block-diagonal Hadamard with learnable signs), not dense matrices.
    This means every candidate is O(n log n) to apply and naturally
    hardware-friendly.
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self.dim = config.dim
        self.rng = random.Random(config.seed)
        # Filter block sizes to those that divide dim
        self.valid_block_sizes = [
            b for b in config.block_sizes if config.dim % b == 0
        ]

    def random_rotation(self, seed: Optional[int] = None) -> NOVARotation:
        """Create a random rotation with random structure and signs."""
        if seed is not None:
            gen = torch.Generator(device='cpu').manual_seed(seed)
        else:
            gen = None

        n_stages = self.rng.randint(*self.config.n_stages_range)
        stages = []
        for _ in range(n_stages):
            block_size = self.rng.choice(self.valid_block_sizes)
            if gen is not None:
                signs = torch.randint(0, 2, (self.dim,), generator=gen) * 2 - 1
            else:
                signs = torch.randint(0, 2, (self.dim,)) * 2 - 1
            stages.append(RotationStage(block_size=block_size, signs=signs.float()))

        return NOVARotation(dim=self.dim, stages=stages)

    def mutate(self, rotation: NOVARotation) -> NOVARotation:
        """Mutate a rotation. Most mutations are sign flips (cheap to evaluate)."""
        rot = rotation.clone()

        for stage in rot.stages:
            # Flip signs: Poisson-distributed number of bit flips
            n_flips = max(1, int(torch.distributions.Poisson(
                self.dim * self.config.sign_mutation_rate
            ).sample().item()))
            n_flips = min(n_flips, self.dim)
            flip_idx = torch.randperm(self.dim)[:n_flips]
            stage.signs[flip_idx] *= -1

        # Maybe change block size of a random stage
        if self.rng.random() < self.config.block_mutation_prob and rot.stages:
            idx = self.rng.randrange(len(rot.stages))
            rot.stages[idx].block_size = self.rng.choice(self.valid_block_sizes)

        # Maybe add or remove a stage
        if self.rng.random() < self.config.stage_mutation_prob:
            if len(rot.stages) > 1 and self.rng.random() < 0.5:
                # Remove random stage
                rot.stages.pop(self.rng.randrange(len(rot.stages)))
            elif len(rot.stages) < self.config.n_stages_range[1]:
                # Add random stage
                new_stage = RotationStage(
                    block_size=self.rng.choice(self.valid_block_sizes),
                    signs=(torch.randint(0, 2, (self.dim,)) * 2 - 1).float(),
                )
                pos = self.rng.randrange(len(rot.stages) + 1)
                rot.stages.insert(pos, new_stage)

        return rot

    def crossover(self, a: NOVARotation, b: NOVARotation) -> NOVARotation:
        """Crossover two rotations via uniform sign crossover on shared stages."""
        # Take structure from the fitter parent (a), mix signs from both
        child = a.clone()

        n_shared = min(len(child.stages), len(b.stages))
        for i in range(n_shared):
            if child.stages[i].block_size == b.stages[i].block_size:
                # Same block size: uniform sign crossover
                mask = torch.randint(0, 2, (self.dim,)).bool()
                child.stages[i].signs[mask] = b.stages[i].signs[mask]
            else:
                # Different block sizes: pick one parent's stage
                if self.rng.random() < 0.5:
                    child.stages[i] = RotationStage(
                        block_size=b.stages[i].block_size,
                        signs=b.stages[i].signs.clone(),
                    )

        return child

    def run(
        self,
        cost_fn: Callable[[NOVARotation], Tuple[float, RotationMetrics, float]],
        callback: Optional[Callable[[int, int, float, NOVARotation], None]] = None,
    ) -> SearchResult:
        """Run the full ES search with restarts.

        Args:
            cost_fn: Function mapping rotation -> (cost, metrics, latency).
            callback: Optional progress callback(restart, generation, best_cost, best_rot).

        Returns:
            SearchResult with best rotation found.
        """
        cfg = self.config
        overall_best_rot = None
        overall_best_cost = float('inf')
        overall_best_metrics = None
        history = []
        total_evals = 0
        t_start = time.time()

        for restart in range(cfg.n_restarts):
            # Initialize population
            population = [
                self.random_rotation(seed=cfg.seed + restart * cfg.pop_size + i)
                for i in range(cfg.pop_size)
            ]

            # Evaluate initial population
            costs_metrics = [cost_fn(rot) for rot in population]
            costs = [c for c, _, _ in costs_metrics]
            metrics_list = [m for _, m, _ in costs_metrics]
            total_evals += len(population)

            for gen in range(cfg.n_generations):
                # Sort by cost
                ranked = sorted(
                    zip(costs, population, metrics_list),
                    key=lambda x: x[0]
                )

                # Track best
                if ranked[0][0] < overall_best_cost:
                    overall_best_cost = ranked[0][0]
                    overall_best_rot = ranked[0][1].clone()
                    overall_best_metrics = ranked[0][2]

                # Record history
                if gen % 10 == 0:
                    entry = {
                        'restart': restart,
                        'generation': gen,
                        'best_cost': ranked[0][0],
                        'mean_cost': sum(costs) / len(costs),
                        'best_structure': str(ranked[0][1]),
                    }
                    history.append(entry)

                    if callback:
                        callback(restart, gen, ranked[0][0], ranked[0][1])

                # Elite selection
                n_elite = max(2, int(cfg.pop_size * cfg.elite_fraction))
                elites = [(c, rot, m) for c, rot, m in ranked[:n_elite]]

                # Generate next population
                new_pop = [rot.clone() for _, rot, _ in elites]
                new_costs = [c for c, _, _ in elites]
                new_metrics = [m for _, _, m in elites]

                while len(new_pop) < cfg.pop_size:
                    # Select parents from elite
                    _, p1, _ = self.rng.choice(elites)
                    _, p2, _ = self.rng.choice(elites)

                    if self.rng.random() < cfg.crossover_prob:
                        child = self.crossover(p1, p2)
                    else:
                        child = p1.clone()

                    child = self.mutate(child)

                    # Evaluate child
                    cost, metrics, latency = cost_fn(child)
                    total_evals += 1

                    new_pop.append(child)
                    new_costs.append(cost)
                    new_metrics.append(metrics)

                population = new_pop
                costs = new_costs
                metrics_list = new_metrics

            # End-of-restart report
            print(f"  Restart {restart+1}/{cfg.n_restarts} complete. "
                  f"Best cost this restart: {min(costs):.6f}")

        total_time = time.time() - t_start

        return SearchResult(
            best_rotation=overall_best_rot,
            best_cost=overall_best_cost,
            best_metrics=overall_best_metrics,
            history=history,
            total_evaluations=total_evals,
            total_time_s=total_time,
        )


class StructureAwareES(NOVAQuantES):
    """Extended ES that separately optimizes structure and signs.

    Phase 1: Search over structures (block sizes, n_stages) with random signs.
    Phase 2: Fix the best structure, optimize signs only.

    This two-phase approach is more efficient because structure changes are
    expensive (cache miss for latency) while sign changes are cheap.
    """

    def __init__(self, config: SearchConfig):
        super().__init__(config)

    def run_two_phase(
        self,
        cost_fn: Callable,
        structure_gens: int = 50,
        sign_gens: int = 150,
        callback: Optional[Callable] = None,
    ) -> SearchResult:
        """Two-phase search: structure exploration then sign optimization."""
        cfg = self.config

        # Phase 1: Structure search with higher mutation rates
        print("Phase 1: Exploring rotation structures...")
        struct_config = SearchConfig(
            dim=cfg.dim,
            pop_size=cfg.pop_size,
            n_generations=structure_gens,
            n_restarts=cfg.n_restarts,
            n_stages_range=cfg.n_stages_range,
            block_sizes=cfg.block_sizes,
            sign_mutation_rate=0.05,  # Higher rate for exploration
            block_mutation_prob=0.3,  # More structure mutation
            stage_mutation_prob=0.1,
            elite_fraction=cfg.elite_fraction,
            crossover_prob=cfg.crossover_prob,
            seed=cfg.seed,
        )
        struct_es = NOVAQuantES(struct_config)
        phase1_result = struct_es.run(cost_fn, callback)

        # Phase 2: Fix structure, optimize signs only
        best_structure = phase1_result.best_rotation
        print(f"\nPhase 2: Optimizing signs for structure {best_structure}...")

        sign_config = SearchConfig(
            dim=cfg.dim,
            pop_size=cfg.pop_size * 2,  # Larger population for sign search
            n_generations=sign_gens,
            n_restarts=1,  # No restarts needed, structure is fixed
            n_stages_range=(best_structure.n_stages, best_structure.n_stages),
            block_sizes=best_structure.block_sizes,
            sign_mutation_rate=0.01,  # Fine-grained sign optimization
            block_mutation_prob=0.0,  # No structure mutation
            stage_mutation_prob=0.0,
            elite_fraction=cfg.elite_fraction,
            crossover_prob=cfg.crossover_prob,
            seed=cfg.seed + 10000,
        )

        # Seed initial population with best structure's signs + mutations
        sign_es = NOVAQuantES(sign_config)
        # Override random_rotation to use best structure
        original_random = sign_es.random_rotation

        def seeded_rotation(seed=None):
            rot = best_structure.clone()
            # Randomize signs only
            for stage in rot.stages:
                n_flips = int(self.dim * 0.1)
                flip_idx = torch.randperm(self.dim)[:n_flips]
                stage.signs[flip_idx] *= -1
            return rot

        sign_es.random_rotation = seeded_rotation
        phase2_result = sign_es.run(cost_fn, callback)

        # Combine results
        if phase2_result.best_cost < phase1_result.best_cost:
            final = phase2_result
        else:
            final = phase1_result

        final.history = phase1_result.history + phase2_result.history
        final.total_evaluations = (phase1_result.total_evaluations
                                   + phase2_result.total_evaluations)
        final.total_time_s = phase1_result.total_time_s + phase2_result.total_time_s
        return final
