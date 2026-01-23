#!/usr/bin/env python3
"""
JAXBench Results Visualization

Generates histograms showing:
1. Level 1 speedups (JAX vs PyTorch/XLA)
2. Level 1 numerical differences (max_diff)
3. Level 2 speedups
4. Level 2 numerical differences

Usage:
    python visualize_results.py results/jaxbench_*.json
    python visualize_results.py --level1 results/level1.json --level2 results/level2.json
"""

import os
import sys
import json
import glob
import argparse
from typing import List, Dict, Optional
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Please install matplotlib and numpy: pip install matplotlib numpy")
    sys.exit(1)


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_latest_results(results_dir: str = "results", level: Optional[int] = None) -> List[str]:
    """Find the latest result files."""
    pattern = os.path.join(results_dir, "jaxbench_*.json")
    files = glob.glob(pattern)
    
    # Filter by level if specified
    if level:
        # We need to check inside the files to determine level
        # For now, just return all and filter later
        pass
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def extract_metrics(results: Dict) -> Dict:
    """Extract speedup and max_diff metrics from results."""
    metrics = {
        "speedups": [],
        "max_diffs": [],
        "task_names": [],
        "task_ids": [],
        "successes": 0,
        "failures": 0,
    }
    
    for task in results.get("tasks", []):
        task_id = task.get("task_id", "?")
        task_name = task.get("task_name", "Unknown")
        
        if task.get("correctness_success"):
            metrics["successes"] += 1
            
            speedup = task.get("speedup")
            if speedup is not None:
                metrics["speedups"].append(speedup)
                metrics["task_names"].append(task_name[:30])
                metrics["task_ids"].append(task_id)
            
            max_diff = task.get("max_diff")
            if max_diff is not None:
                metrics["max_diffs"].append(max_diff)
        else:
            metrics["failures"] += 1
    
    return metrics


def plot_histograms(level1_metrics: Optional[Dict], level2_metrics: Optional[Dict], 
                    output_path: str = "jaxbench_results.png"):
    """Generate 4-panel histogram figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("JAXBench Results: JAX vs PyTorch/XLA on TPU v6e", fontsize=14, fontweight='bold')
    
    # Color scheme
    speedup_color = '#2ecc71'  # Green
    diff_color = '#3498db'     # Blue
    
    # Panel 1: Level 1 Speedups
    ax1 = axes[0, 0]
    if level1_metrics and level1_metrics["speedups"]:
        speedups = level1_metrics["speedups"]
        ax1.hist(speedups, bins=20, color=speedup_color, edgecolor='black', alpha=0.7)
        ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='1.0x (parity)')
        ax1.axvline(x=np.mean(speedups), color='orange', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(speedups):.2f}x')
        ax1.set_xlabel('Speedup (JAX / PyTorch/XLA)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Level 1 Speedups (n={len(speedups)}, {level1_metrics["successes"]}/{level1_metrics["successes"]+level1_metrics["failures"]} passed)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No Level 1 data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Level 1 Speedups')
    
    # Panel 2: Level 1 Numerical Differences
    ax2 = axes[0, 1]
    if level1_metrics and level1_metrics["max_diffs"]:
        diffs = level1_metrics["max_diffs"]
        ax2.hist(diffs, bins=20, color=diff_color, edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(diffs), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(diffs):.4f}')
        ax2.set_xlabel('Max Absolute Difference')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Level 1 Numerical Differences (n={len(diffs)})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        # Log scale if values vary widely
        if max(diffs) / (min(diffs) + 1e-10) > 100:
            ax2.set_xscale('log')
    else:
        ax2.text(0.5, 0.5, 'No Level 1 data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Level 1 Numerical Differences')
    
    # Panel 3: Level 2 Speedups
    ax3 = axes[1, 0]
    if level2_metrics and level2_metrics["speedups"]:
        speedups = level2_metrics["speedups"]
        ax3.hist(speedups, bins=20, color=speedup_color, edgecolor='black', alpha=0.7)
        ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='1.0x (parity)')
        ax3.axvline(x=np.mean(speedups), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(speedups):.2f}x')
        ax3.set_xlabel('Speedup (JAX / PyTorch/XLA)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Level 2 Speedups (n={len(speedups)}, {level2_metrics["successes"]}/{level2_metrics["successes"]+level2_metrics["failures"]} passed)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Level 2 data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Level 2 Speedups')
    
    # Panel 4: Level 2 Numerical Differences
    ax4 = axes[1, 1]
    if level2_metrics and level2_metrics["max_diffs"]:
        diffs = level2_metrics["max_diffs"]
        ax4.hist(diffs, bins=20, color=diff_color, edgecolor='black', alpha=0.7)
        ax4.axvline(x=np.mean(diffs), color='orange', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(diffs):.4f}')
        ax4.set_xlabel('Max Absolute Difference')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Level 2 Numerical Differences (n={len(diffs)})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        if max(diffs) / (min(diffs) + 1e-10) > 100:
            ax4.set_xscale('log')
    else:
        ax4.text(0.5, 0.5, 'No Level 2 data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Level 2 Numerical Differences')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved histogram to {output_path}")
    
    return fig


def print_summary(metrics: Dict, level: int):
    """Print summary statistics."""
    print(f"\n{'='*50}")
    print(f"Level {level} Summary")
    print(f"{'='*50}")
    
    total = metrics["successes"] + metrics["failures"]
    print(f"Tasks: {metrics['successes']}/{total} passed ({100*metrics['successes']/total:.1f}%)")
    
    if metrics["speedups"]:
        speedups = metrics["speedups"]
        print(f"\nSpeedups (JAX vs PyTorch/XLA):")
        print(f"  Mean:   {np.mean(speedups):.2f}x")
        print(f"  Median: {np.median(speedups):.2f}x")
        print(f"  Min:    {min(speedups):.2f}x")
        print(f"  Max:    {max(speedups):.2f}x")
        print(f"  >1.0x:  {sum(1 for s in speedups if s > 1.0)}/{len(speedups)} ({100*sum(1 for s in speedups if s > 1.0)/len(speedups):.1f}%)")
    
    if metrics["max_diffs"]:
        diffs = metrics["max_diffs"]
        print(f"\nNumerical Differences:")
        print(f"  Mean:   {np.mean(diffs):.6f}")
        print(f"  Median: {np.median(diffs):.6f}")
        print(f"  Min:    {min(diffs):.6f}")
        print(f"  Max:    {max(diffs):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize JAXBench results")
    parser.add_argument("files", nargs="*", help="Result JSON files to visualize")
    parser.add_argument("--level1", type=str, help="Level 1 results JSON file")
    parser.add_argument("--level2", type=str, help="Level 2 results JSON file")
    parser.add_argument("--output", "-o", type=str, default="jaxbench_results.png",
                       help="Output image path")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory to search for result files")
    args = parser.parse_args()
    
    level1_metrics = None
    level2_metrics = None
    
    # Load specified files
    if args.level1:
        results = load_results(args.level1)
        level1_metrics = extract_metrics(results)
        print_summary(level1_metrics, 1)
    
    if args.level2:
        results = load_results(args.level2)
        level2_metrics = extract_metrics(results)
        print_summary(level2_metrics, 2)
    
    # Load from positional arguments
    for filepath in args.files:
        results = load_results(filepath)
        metrics = extract_metrics(results)
        
        # Try to determine level from filename or content
        if "level1" in filepath.lower() or "level_1" in filepath.lower():
            level1_metrics = metrics
            print_summary(metrics, 1)
        elif "level2" in filepath.lower() or "level_2" in filepath.lower():
            level2_metrics = metrics
            print_summary(metrics, 2)
        else:
            # Check number of tasks - Level 1 has simpler names
            task_names = [t.get("task_name", "") for t in results.get("tasks", [])]
            has_conv = any("conv" in name.lower() for name in task_names)
            if has_conv:
                level2_metrics = metrics
                print_summary(metrics, 2)
            else:
                level1_metrics = metrics
                print_summary(metrics, 1)
    
    # Auto-find latest results if nothing specified
    if not args.files and not args.level1 and not args.level2:
        print("No files specified, searching for latest results...")
        files = find_latest_results(args.results_dir)
        if files:
            print(f"Found {len(files)} result files")
            for f in files[:4]:  # Load up to 4 most recent
                print(f"  Loading: {f}")
                results = load_results(f)
                metrics = extract_metrics(results)
                
                # Determine level
                task_names = [t.get("task_name", "") for t in results.get("tasks", [])]
                has_conv = any("conv" in name.lower() for name in task_names)
                if has_conv:
                    if level2_metrics is None:
                        level2_metrics = metrics
                        print_summary(metrics, 2)
                else:
                    if level1_metrics is None:
                        level1_metrics = metrics
                        print_summary(metrics, 1)
        else:
            print(f"No result files found in {args.results_dir}/")
            return
    
    # Generate plots
    if level1_metrics or level2_metrics:
        plot_histograms(level1_metrics, level2_metrics, args.output)
        
        # Also save summary to JSON
        summary = {
            "generated_at": datetime.now().isoformat(),
            "level1": {
                "total": level1_metrics["successes"] + level1_metrics["failures"] if level1_metrics else 0,
                "passed": level1_metrics["successes"] if level1_metrics else 0,
                "mean_speedup": float(np.mean(level1_metrics["speedups"])) if level1_metrics and level1_metrics["speedups"] else None,
                "mean_diff": float(np.mean(level1_metrics["max_diffs"])) if level1_metrics and level1_metrics["max_diffs"] else None,
            } if level1_metrics else None,
            "level2": {
                "total": level2_metrics["successes"] + level2_metrics["failures"] if level2_metrics else 0,
                "passed": level2_metrics["successes"] if level2_metrics else 0,
                "mean_speedup": float(np.mean(level2_metrics["speedups"])) if level2_metrics and level2_metrics["speedups"] else None,
                "mean_diff": float(np.mean(level2_metrics["max_diffs"])) if level2_metrics and level2_metrics["max_diffs"] else None,
            } if level2_metrics else None,
        }
        
        summary_path = args.output.replace('.png', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")
    else:
        print("No data to visualize")


if __name__ == "__main__":
    main()

