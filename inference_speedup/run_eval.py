"""CLI for running inference speedup evaluations.

Usage:
    # Single model, full evaluation
    python -m inference_speedup.run_eval --model llama3_8b

    # All models
    python -m inference_speedup.run_eval --all

    # With optimized kernels
    python -m inference_speedup.run_eval --model llama3_8b --optimized rmsnorm swiglu_mlp

    # Custom config
    python -m inference_speedup.run_eval --model gla_1_3b --seq-len 1024 --num-layers 12

    # Prefill only (faster)
    python -m inference_speedup.run_eval --model mamba2_2_7b --no-decode

    # Output to JSON
    python -m inference_speedup.run_eval --all --output results.json
"""

import argparse
import json
import sys

from inference_speedup.config import ALL_MODELS
from inference_speedup.eval_harness import evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate kernel speedup impact on end-to-end inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', choices=list(ALL_MODELS.keys()),
                        help='Model to evaluate')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all models')
    parser.add_argument('--seq-len', type=int, default=2048,
                        help='Sequence length for prefill (default: 2048)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Override number of layers (default: per-model eval_layers)')
    parser.add_argument('--optimized', nargs='+', default=None,
                        help='Kernels to swap with Pallas (e.g., rmsnorm swiglu_mlp)')
    parser.add_argument('--no-profile', action='store_true',
                        help='Skip kernel profiling')
    parser.add_argument('--no-decode', action='store_true',
                        help='Skip decode benchmark (faster)')
    parser.add_argument('--decode-prompt-len', type=int, default=128,
                        help='Prompt length for decode (default: 128)')
    parser.add_argument('--decode-gen-tokens', type=int, default=64,
                        help='Tokens to generate in decode (default: 64)')
    parser.add_argument('--output', default=None,
                        help='Save results to JSON file')
    parser.add_argument('--list-kernels', action='store_true',
                        help='List available Pallas kernels and exit')
    args = parser.parse_args()

    if args.list_kernels:
        from inference_speedup.pallas_kernels import AVAILABLE_PALLAS_KERNELS
        print("Available Pallas-optimized kernels:")
        for name in AVAILABLE_PALLAS_KERNELS:
            print(f"  {name}")
        return

    if not args.model and not args.all:
        parser.error("Specify --model or --all")

    models = list(ALL_MODELS.keys()) if args.all else [args.model]
    all_results = {}

    for model_name in models:
        results = evaluate_model(
            model_name,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_layers=args.num_layers,
            profile=not args.no_profile,
            optimized_kernels=args.optimized,
            decode=not args.no_decode,
            decode_prompt_len=args.decode_prompt_len,
            decode_gen_tokens=args.decode_gen_tokens,
        )
        all_results[model_name] = results

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
