#!/usr/bin/env python3
"""
Visualize JAXBench Level 1 and Level 2 results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/Users/aryatschand/Documents/GitHub/JAXBench/results/checkpoint_level1.json') as f:
    level1_data = json.load(f)

with open('/Users/aryatschand/Documents/GitHub/JAXBench/results/checkpoint_level2.json') as f:
    level2_data = json.load(f)

# Extract data for Level 1
level1_speedups = []
level1_jax_ms = []
level1_pytorch_ms = []

for task in level1_data['tasks']:
    if task.get('correctness_success') and task.get('speedup') is not None:
        level1_speedups.append(task['speedup'])
        if task.get('jax_ms') is not None:
            level1_jax_ms.append(task['jax_ms'])
        if task.get('pytorch_xla_ms') is not None:
            level1_pytorch_ms.append(task['pytorch_xla_ms'])

# Extract data for Level 2
level2_speedups = []
level2_jax_ms = []
level2_pytorch_ms = []

for task in level2_data['tasks']:
    if task.get('correctness_success') and task.get('speedup') is not None:
        level2_speedups.append(task['speedup'])
        if task.get('jax_ms') is not None:
            level2_jax_ms.append(task['jax_ms'])
        if task.get('pytorch_xla_ms') is not None:
            level2_pytorch_ms.append(task['pytorch_xla_ms'])

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('JAXBench: JAX vs PyTorch/XLA Performance Comparison', fontsize=14, fontweight='bold')

# Color scheme
jax_color = '#2ecc71'
pytorch_color = '#e74c3c'
speedup_color = '#3498db'

# Top Left: Level 1 Speedup Histogram (LOG SCALE) - 20 bins
ax1 = axes[0, 0]
min_speedup1 = min(level1_speedups)
max_speedup1 = max(level1_speedups)
bins1 = np.logspace(np.log10(max(0.1, min_speedup1 * 0.9)), np.log10(max_speedup1 * 1.1), 20)
ax1.hist(level1_speedups, bins=bins1, color=speedup_color, edgecolor='black', alpha=0.7, linewidth=0.5)
ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='1.0x (parity)')
ax1.axvline(x=np.mean(level1_speedups), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(level1_speedups):.2f}x')
ax1.axvline(x=np.median(level1_speedups), color='green', linestyle=':', linewidth=2, label=f'Median: {np.median(level1_speedups):.2f}x')
ax1.set_xscale('log')
ax1.set_xlabel('Speedup (JAX / PyTorch/XLA) - Log Scale', fontsize=11)
ax1.set_ylabel('Number of Tasks', fontsize=11)
ax1.set_title(f'Level 1 Speedup Distribution (n={len(level1_speedups)})', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.grid(axis='x', alpha=0.2, which='both')

# Top Right: Level 2 Speedup Histogram (LOG SCALE) - 20 bins
ax2 = axes[0, 1]
min_speedup2 = min(level2_speedups)
max_speedup2 = max(level2_speedups)
bins2 = np.logspace(np.log10(max(0.1, min_speedup2 * 0.9)), np.log10(max_speedup2 * 1.1), 20)
ax2.hist(level2_speedups, bins=bins2, color=speedup_color, edgecolor='black', alpha=0.7, linewidth=0.5)
ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='1.0x (parity)')
ax2.axvline(x=np.mean(level2_speedups), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(level2_speedups):.2f}x')
ax2.axvline(x=np.median(level2_speedups), color='green', linestyle=':', linewidth=2, label=f'Median: {np.median(level2_speedups):.2f}x')
ax2.set_xscale('log')
ax2.set_xlabel('Speedup (JAX / PyTorch/XLA) - Log Scale', fontsize=11)
ax2.set_ylabel('Number of Tasks', fontsize=11)
ax2.set_title(f'Level 2 Speedup Distribution (n={len(level2_speedups)})', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.grid(axis='x', alpha=0.2, which='both')

# Bottom Left: Level 1 Raw Runtime Histogram - 25 bins
ax3 = axes[1, 0]
max_runtime1 = max(max(level1_jax_ms), max(level1_pytorch_ms))
bins3 = np.logspace(np.log10(0.1), np.log10(max_runtime1 + 1), 25)
ax3.hist(level1_jax_ms, bins=bins3, color=jax_color, edgecolor='black', alpha=0.6, linewidth=0.5, label=f'JAX (mean: {np.mean(level1_jax_ms):.2f}ms)')
ax3.hist(level1_pytorch_ms, bins=bins3, color=pytorch_color, edgecolor='black', alpha=0.6, linewidth=0.5, label=f'PyTorch/XLA (mean: {np.mean(level1_pytorch_ms):.2f}ms)')
ax3.set_xscale('log')
ax3.set_xlabel('Runtime (ms) - Log Scale', fontsize=11)
ax3.set_ylabel('Number of Tasks', fontsize=11)
ax3.set_title('Level 1 Runtime Distribution', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(axis='y', alpha=0.3)
ax3.grid(axis='x', alpha=0.2, which='both')

# Bottom Right: Level 2 Raw Runtime Histogram - 25 bins
ax4 = axes[1, 1]
max_runtime2 = max(max(level2_jax_ms), max(level2_pytorch_ms))
bins4 = np.logspace(np.log10(0.1), np.log10(max_runtime2 + 1), 25)
ax4.hist(level2_jax_ms, bins=bins4, color=jax_color, edgecolor='black', alpha=0.6, linewidth=0.5, label=f'JAX (mean: {np.mean(level2_jax_ms):.2f}ms)')
ax4.hist(level2_pytorch_ms, bins=bins4, color=pytorch_color, edgecolor='black', alpha=0.6, linewidth=0.5, label=f'PyTorch/XLA (mean: {np.mean(level2_pytorch_ms):.2f}ms)')
ax4.set_xscale('log')
ax4.set_xlabel('Runtime (ms) - Log Scale', fontsize=11)
ax4.set_ylabel('Number of Tasks', fontsize=11)
ax4.set_title('Level 2 Runtime Distribution', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right')
ax4.grid(axis='y', alpha=0.3)
ax4.grid(axis='x', alpha=0.2, which='both')

plt.tight_layout()
plt.subplots_adjust(top=0.92)

output_path = '/Users/aryatschand/Documents/GitHub/JAXBench/results/jaxbench_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

pdf_path = '/Users/aryatschand/Documents/GitHub/JAXBench/results/jaxbench_visualization.pdf'
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Saved: {pdf_path}")
