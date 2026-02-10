#!/usr/bin/env python3
"""Generate all figures for the NOVA MI300X technical report."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import shutil
import os

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

AMD_RED = '#ED1C24'
AMD_DARK = '#333333'
NOVA_BLUE = '#1565C0'
NOVA_LIGHT = '#42A5F5'
BROKEN_ORANGE = '#FF6F00'
GREEN = '#2E7D32'
GRAY = '#9E9E9E'

# ─────────────────────────────────────────────────────────────
# Figure 1: Batch=1 Latency — NOVA vs MIOpen
# ─────────────────────────────────────────────────────────────
def fig_latency_b1():
    layers = ['conv2_x\n64ch, 56×56', 'conv3_x\n128ch, 28×28',
              'conv4_x\n256ch, 14×14', 'conv5_x\n512ch, 7×7']
    miopen = [0.035, 0.036, 0.056, 0.050]
    nova =   [0.029, 0.025, 0.024, 0.026]

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, miopen, width, label='MIOpen F(2,3)',
                   color=AMD_RED, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, nova, width, label='NOVA F(6,3)',
                   color=NOVA_BLUE, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Add speedup annotations
    for i in range(len(layers)):
        speedup = miopen[i] / nova[i]
        ax.annotate(f'{speedup:.1f}× faster',
                   xy=(x[i] + width/2, nova[i]),
                   xytext=(0, 8), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold', color=NOVA_BLUE)

    ax.set_ylabel('Latency (ms) — lower is better')
    ax.set_title('Batch=1 Inference Latency: NOVA Beats MIOpen on Every Layer', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.075)

    fig.savefig(f'{FIGDIR}/latency_b1.pdf')
    fig.savefig(f'{FIGDIR}/latency_b1.png')
    plt.close()
    print("  latency_b1.pdf")

# ─────────────────────────────────────────────────────────────
# Figure 2: ImageNetV2 Accuracy
# ─────────────────────────────────────────────────────────────
def fig_imagenet_accuracy():
    labels = ['FP32\nBaseline', 'MIOpen\nFP16', 'NOVA F(4,3)\nFP16',
              'NOVA F(6,3)\nFP16', 'Standard F(6,3)\nFP16']
    values = [63.15, 63.18, 63.12, 63.29, 31.07]
    colors = [GRAY, AMD_RED, NOVA_LIGHT, NOVA_BLUE, BROKEN_ORANGE]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
               f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Add NaN annotation for standard
    ax.annotate('221,000 NaN values\nCATASTROPHIC FAILURE',
               xy=(4, 31.07), xytext=(3.0, 45),
               fontsize=9, color=BROKEN_ORANGE, fontweight='bold',
               ha='center',
               arrowprops=dict(arrowstyle='->', color=BROKEN_ORANGE, lw=1.5))

    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('ImageNetV2 Accuracy (10,000 images, ResNet-50)', fontweight='bold')
    ax.set_ylim(0, 75)
    ax.axhline(y=63.15, color=GRAY, linestyle='--', alpha=0.5, linewidth=0.8)

    fig.savefig(f'{FIGDIR}/imagenet_accuracy.pdf')
    fig.savefig(f'{FIGDIR}/imagenet_accuracy.png')
    plt.close()
    print("  imagenet_accuracy.pdf")

# ─────────────────────────────────────────────────────────────
# Figure 3: Numerical Stability — NaN/Inf counts
# ─────────────────────────────────────────────────────────────
def fig_stability():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.0))

    # Left: Relative error comparison
    methods = ['NOVA\nF(6,3)', 'Standard\nF(6,3)']
    errors = [5.2, 200.9]  # percent relative error
    colors_err = [NOVA_BLUE, BROKEN_ORANGE]

    bars = ax1.bar(methods, errors, color=colors_err, alpha=0.85, width=0.5)
    ax1.set_ylabel('Relative Error vs FP32 (%)')
    ax1.set_title('FP16 Numerical Error', fontweight='bold')
    for bar, val in zip(bars, errors):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 250)

    # Right: NaN counts
    categories = ['NOVA\nHIP', 'Std\nHIP', 'NOVA\nImgNet', 'Std\nImgNet']
    nans = [0, 618, 0, 221000]
    colors_nan = [NOVA_BLUE, BROKEN_ORANGE, NOVA_BLUE, BROKEN_ORANGE]

    x_pos = np.arange(len(categories))
    bars2 = ax2.bar(x_pos, nans, color=colors_nan, alpha=0.85, width=0.55)
    ax2.set_ylabel('NaN + Inf Count')
    ax2.set_title('Numerical Safety', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_yscale('symlog', linthresh=1)
    ax2.set_ylim(0, 500000)
    for bar, val in zip(bars2, nans):
        label = f'{val//1000}K' if val >= 1000 else (str(val) if val > 0 else 'ZERO')
        color = BROKEN_ORANGE if val > 0 else GREEN
        if val >= 100000:
            # Large values: label inside bar, centered vertically
            y = 5000  # middle of bar in symlog scale
            ax2.text(bar.get_x() + bar.get_width()/2, y,
                    label, ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white')
        elif val > 0:
            y = bar.get_height() * 1.5
            ax2.text(bar.get_x() + bar.get_width()/2, y,
                    label, ha='center', fontsize=9, fontweight='bold', color=color)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, 0.8,
                    label, ha='center', fontsize=9, fontweight='bold', color=color)

    plt.tight_layout(w_pad=3)
    fig.subplots_adjust(right=0.97)
    fig.savefig(f'{FIGDIR}/stability.pdf')
    fig.savefig(f'{FIGDIR}/stability.png')
    plt.close()
    print("  stability.pdf")

# ─────────────────────────────────────────────────────────────
# Figure 4: Architecture diagram
# ─────────────────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    boxes = [
        (0.3, 1.0, 1.8, 1.0, 'Input\n[B,C,H,W]', GRAY),
        (2.5, 1.0, 1.8, 1.0, 'Input\nTransform\nB$^T$·tile·B', NOVA_LIGHT),
        (4.7, 1.0, 1.8, 1.0, 'rocBLAS\nBatched GEMM\n64 × [K,C]·[C,P]', AMD_RED),
        (6.9, 1.0, 1.8, 1.0, 'Output\nTransform\nA·M·A$^T$', NOVA_LIGHT),
        (9.1, 1.0, 0.8, 1.0, 'Output', GRAY),
    ]

    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.3,
                                        edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=8, fontweight='bold')

    # Arrows
    arrow_style = dict(arrowstyle='->', color=AMD_DARK, lw=1.5)
    for x1, x2 in [(2.1, 2.5), (4.3, 4.7), (6.5, 6.9), (8.7, 9.1)]:
        ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5), arrowprops=arrow_style)

    # Labels
    ax.text(3.4, 2.3, 'HIP Kernel\nWave Shuffles', ha='center', fontsize=7,
           color=NOVA_BLUE, fontstyle='italic')
    ax.text(5.6, 2.3, 'FP16→FP32 acc\nMFMA units', ha='center', fontsize=7,
           color=AMD_RED, fontstyle='italic')
    ax.text(7.8, 2.3, 'HIP Kernel\nWave Shuffles', ha='center', fontsize=7,
           color=NOVA_BLUE, fontstyle='italic')

    # Weight path
    ax.text(5.6, 0.5, 'Weights [K,C,3,3] → Filter Transform (cached) → U [64,K,C]',
           ha='center', fontsize=7, color=AMD_DARK, fontstyle='italic')

    ax.set_title('NOVA F(6,3) Multi-Pass Architecture on MI300X', fontweight='bold', fontsize=11)

    fig.savefig(f'{FIGDIR}/architecture.pdf')
    fig.savefig(f'{FIGDIR}/architecture.png')
    plt.close()
    print("  architecture.pdf")

# ─────────────────────────────────────────────────────────────
# Figure 5: Full performance table as visual (all batch sizes)
# ─────────────────────────────────────────────────────────────
def fig_full_perf():
    with open('/root/nova_kernel_benchmark.json') as f:
        data = json.load(f)

    configs = [r['config'] for r in data['results']]
    miopen = [r['ms_miopen'] for r in data['results']]
    nova = [r['ms_nova_hip'] for r in data['results']]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - width/2, miopen, width, label='MIOpen F(2,3)',
                   color=AMD_RED, alpha=0.85)
    bars2 = ax.bar(x + width/2, nova, width, label='NOVA F(6,3)',
                   color=NOVA_BLUE, alpha=0.85)

    # Mark B=1 region
    ax.axvspan(-0.5, 3.5, alpha=0.08, color=GREEN)
    ax.text(1.5, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1] > 0.1 else 0.35,
           'NOVA WINS\n(batch=1)', ha='center', fontsize=10,
           fontweight='bold', color=GREEN, alpha=0.7)

    ax.set_ylabel('Latency (ms) — lower is better')
    ax.set_title('Full Performance Comparison: ResNet-50 Layers on MI300X', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha='right', fontsize=8)
    ax.legend(loc='upper left')

    fig.savefig(f'{FIGDIR}/full_perf.pdf')
    fig.savefig(f'{FIGDIR}/full_perf.png')
    plt.close()
    print("  full_perf.pdf")

# ─────────────────────────────────────────────────────────────
# Copy SD output image
# ─────────────────────────────────────────────────────────────
def copy_sd_image():
    src = '/root/nova_sd_output.png'
    dst = f'{FIGDIR}/sd_output.png'
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  sd_output.png (copied)")
    else:
        print(f"  WARNING: {src} not found")

# ─────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating figures...")
    fig_latency_b1()
    fig_imagenet_accuracy()
    fig_stability()
    fig_architecture()
    fig_full_perf()
    copy_sd_image()
    print("Done.")
