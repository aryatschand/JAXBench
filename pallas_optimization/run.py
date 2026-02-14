#!/usr/bin/env python3
"""
Pallas Optimization CLI

Generate and evaluate Pallas TPU kernels.

Usage:
    python -m pallas_optimization.run --workload llama3_8b_gqa
    python -m pallas_optimization.run --list
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pallas Kernel Optimization")
    parser.add_argument("--list", action="store_true", help="List available workloads")
    parser.add_argument("--workload", type=str, help="Workload to optimize")
    parser.add_argument("--output", type=str, help="Output file for generated kernel")
    parser.add_argument("--provider", type=str, default="bedrock", help="LLM provider")
    parser.add_argument("--model", type=str, default="opus", help="LLM model")

    args = parser.parse_args()

    if args.list:
        print("Available workloads:")
        print("  - llama3_8b_gqa")
        print("  - llama3_8b_rope")
        print("  - llama3_8b_swiglu")
        print("  - gemma3_27b_sliding")
        # TODO: Load from workload registry
        return

    if not args.workload:
        print("Please specify --workload or use --list")
        return

    print(f"Pallas optimization for: {args.workload}")
    print("TODO: Implement optimization pipeline")
    # TODO: Implement using PallasTranslator


if __name__ == "__main__":
    main()
