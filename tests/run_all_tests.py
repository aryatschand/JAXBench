#!/usr/bin/env python3
"""
Run all JAXBench tests.

This script runs all tests to verify the benchmark setup is correct:
1. TPU connection tests
2. LLM client tests  
3. Matmul benchmark test (JAX + PyTorch/XLA)

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --quick  # Skip slow tests
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test(test_name, test_func):
    """Run a test and return result."""
    print(f"\n{'#' * 70}")
    print(f"# Running: {test_name}")
    print(f"{'#' * 70}\n")
    
    try:
        result = test_func()
        return result == 0 if isinstance(result, int) else result
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run JAXBench tests")
    parser.add_argument("--quick", action="store_true", 
                       help="Skip slow tests (TPU benchmark)")
    parser.add_argument("--tpu-only", action="store_true",
                       help="Only run TPU tests")
    parser.add_argument("--llm-only", action="store_true",
                       help="Only run LLM tests")
    args = parser.parse_args()
    
    print("=" * 70)
    print("JAXBench Test Suite")
    print("=" * 70)
    
    results = {}
    
    # TPU Connection Tests
    if not args.llm_only:
        from tests.test_tpu_connection import main as test_tpu_connection
        results["TPU Connection"] = run_test("TPU Connection Tests", test_tpu_connection)
    
    # LLM Client Tests
    if not args.tpu_only:
        from tests.test_llm_client import main as test_llm_client
        results["LLM Client"] = run_test("LLM Client Tests", test_llm_client)
    
    # Matmul Benchmark Test (slow)
    if not args.quick and not args.llm_only:
        from tests.test_matmul_benchmark import main as test_matmul
        results["Matmul Benchmark"] = run_test("Matmul Benchmark Test", test_matmul)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        if not passed:
            all_passed = False
        print(f"  {test_name}: {status}")
    
    print()
    if all_passed:
        print("🎉 All tests passed! JAXBench is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run translation: python scripts/run_benchmark.py --level 1 --tasks 1,2,3")
        print("  2. View results: python scripts/visualize_results.py --level1 results/checkpoint_level1.json")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

