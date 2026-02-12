"""NOVA-Quant: Evolution Strategy discovery of hardware-optimal rotation matrices.

Instead of using random Hadamard rotations (QuIP#) or SGD-learned dense matrices
(SpinQuant), NOVA-Quant uses Evolution Strategies to discover structured rotations
that are both high-precision AND hardware-native on AMD MI300X.

The search space is cascaded block-diagonal Hadamard transforms with learnable
sign vectors. This preserves the O(n log n) computational advantage while allowing
the ES to optimize for the specific activation distribution and hardware characteristics.
"""
