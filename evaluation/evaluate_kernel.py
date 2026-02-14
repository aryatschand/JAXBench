"""
Single Kernel Evaluation API

This is the main API for evaluating LLM-generated Pallas kernels in the optimization loop.
Takes generated code, a benchmark reference, and hardware config, returns evaluation results.

Usage:
    from evaluation.evaluate_kernel import evaluate_kernel, EvaluationResult, HardwareConfig

    result = evaluate_kernel(
        generated_code="def pallas_kernel(...): ...",
        benchmark_ref="kernelbench:level1:1",  # or "real_workloads:llama3:gqa"
        hardware_config=HardwareConfig(tpu_type="v5e-4"),
    )

    if result.correct:
        print(f"Speedup: {result.speedup}x")
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
from enum import Enum
import time


class BenchmarkType(Enum):
    """Type of benchmark to evaluate against."""
    KERNELBENCH = "kernelbench"      # PyTorch KernelBench -> JAX baseline
    JAXKERNELBENCH = "jaxkernelbench"  # JAX translated baseline
    REAL_WORKLOADS = "real_workloads"  # Model-specific JAX baselines


@dataclass
class HardwareConfig:
    """Hardware configuration for evaluation."""
    # TPU Configuration
    tpu_type: str = "v5e-4"  # v5e-4, v5e-8, v4-8, v4-16, v4-32
    tpu_zone: str = "us-central1-b"
    tpu_project: str = "jaxbench"

    # Benchmark Settings
    num_warmup: int = 5
    num_iters: int = 50

    # Precision Settings
    dtype: str = "bfloat16"

    # SSH Settings (for remote TPU execution)
    ssh_key_path: Optional[str] = None
    ssh_user: Optional[str] = None
    tpu_ip: Optional[str] = None  # If already have a TPU

    # Additional config
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of kernel evaluation."""
    # Core metrics
    correct: bool
    speedup: float  # vs baseline (>1.0 means faster)

    # Timing (milliseconds)
    baseline_ms: float
    kernel_ms: float
    baseline_std_ms: float = 0.0
    kernel_std_ms: float = 0.0

    # Correctness details
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    mean_abs_diff: float = 0.0
    rtol: float = 1e-3
    atol: float = 1e-3

    # Metadata
    benchmark_ref: str = ""
    hardware_config: Optional[HardwareConfig] = None

    # Error handling
    error: Optional[str] = None
    compilation_error: Optional[str] = None
    runtime_error: Optional[str] = None

    # Timestamps
    timestamp: str = ""
    evaluation_time_s: float = 0.0


def parse_benchmark_ref(benchmark_ref: str) -> tuple:
    """
    Parse benchmark reference string.

    Formats:
        - "kernelbench:level1:1" -> (BenchmarkType.KERNELBENCH, "level1", "1")
        - "jaxkernelbench:level2:5" -> (BenchmarkType.JAXKERNELBENCH, "level2", "5")
        - "real_workloads:llama3:gqa" -> (BenchmarkType.REAL_WORKLOADS, "llama3", "gqa")

    Returns:
        (benchmark_type, category, identifier)
    """
    parts = benchmark_ref.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid benchmark_ref format: {benchmark_ref}. "
            "Expected format: 'type:category:identifier' "
            "(e.g., 'kernelbench:level1:1' or 'real_workloads:llama3:gqa')"
        )

    type_str, category, identifier = parts

    try:
        benchmark_type = BenchmarkType(type_str)
    except ValueError:
        valid_types = [t.value for t in BenchmarkType]
        raise ValueError(f"Invalid benchmark type: {type_str}. Valid types: {valid_types}")

    return benchmark_type, category, identifier


def evaluate_kernel(
    generated_code: str,
    benchmark_ref: str,
    hardware_config: Optional[HardwareConfig] = None,
    correctness_tol: Optional[Dict[str, float]] = None,
    timeout_s: float = 300.0,
) -> EvaluationResult:
    """
    Evaluate a single LLM-generated kernel against a benchmark baseline.

    This is the main API for the kernel optimization loop. It handles:
    1. Parsing the generated code
    2. Loading the appropriate baseline from benchmarks/
    3. Running both on TPU hardware
    4. Comparing outputs for correctness
    5. Benchmarking performance

    Args:
        generated_code: LLM-generated Pallas kernel code as a string.
            Must define a function called `pallas_kernel` that takes
            the same inputs as the baseline and returns the same output.

        benchmark_ref: Reference to the benchmark to evaluate against.
            Format: "type:category:identifier"
            Examples:
                - "kernelbench:level1:1" (task 1 from KernelBench level 1)
                - "jaxkernelbench:level2:5" (JAX translation of task 5, level 2)
                - "real_workloads:llama3:gqa" (Llama3 GQA attention)
                - "real_workloads:llama3:rope" (Llama3 RoPE)
                - "real_workloads:gemma3:sliding" (Gemma3 sliding window)

        hardware_config: TPU/hardware configuration. If None, uses defaults.

        correctness_tol: Correctness tolerance. Default: {"rtol": 1e-3, "atol": 1e-3}

        timeout_s: Maximum time for evaluation in seconds.

    Returns:
        EvaluationResult with correctness, speedup, and detailed metrics.

    Example:
        >>> result = evaluate_kernel(
        ...     generated_code='''
        ...     def pallas_kernel(x, y):
        ...         return x @ y
        ...     ''',
        ...     benchmark_ref="real_workloads:llama3:gqa",
        ...     hardware_config=HardwareConfig(tpu_type="v5e-4"),
        ... )
        >>> print(f"Correct: {result.correct}, Speedup: {result.speedup:.2f}x")
    """
    start_time = time.time()

    # Use defaults if not provided
    if hardware_config is None:
        hardware_config = HardwareConfig()

    if correctness_tol is None:
        correctness_tol = {"rtol": 1e-3, "atol": 1e-3}

    rtol = correctness_tol.get("rtol", 1e-3)
    atol = correctness_tol.get("atol", 1e-3)

    # Parse benchmark reference
    try:
        benchmark_type, category, identifier = parse_benchmark_ref(benchmark_ref)
    except ValueError as e:
        return EvaluationResult(
            correct=False,
            speedup=0.0,
            baseline_ms=0.0,
            kernel_ms=0.0,
            error=str(e),
            benchmark_ref=benchmark_ref,
            hardware_config=hardware_config,
        )

    # TODO: Implement actual evaluation logic
    # This is a placeholder - will be filled in when proto is provided
    #
    # The implementation will:
    # 1. Load baseline from benchmarks/{type}/{category}/{identifier}.py
    # 2. Parse generated_code and extract pallas_kernel function
    # 3. SSH to TPU or use local TPU
    # 4. Run baseline, capture output and timing
    # 5. Run generated kernel, capture output and timing
    # 6. Compare outputs using jnp.allclose(rtol, atol)
    # 7. Return EvaluationResult

    evaluation_time = time.time() - start_time

    # Placeholder result - replace with actual implementation
    return EvaluationResult(
        correct=False,
        speedup=0.0,
        baseline_ms=0.0,
        kernel_ms=0.0,
        rtol=rtol,
        atol=atol,
        benchmark_ref=benchmark_ref,
        hardware_config=hardware_config,
        error="evaluate_kernel() not yet implemented - awaiting proto specification",
        evaluation_time_s=evaluation_time,
    )


def evaluate_kernel_batch(
    generated_codes: list,
    benchmark_refs: list,
    hardware_config: Optional[HardwareConfig] = None,
    **kwargs,
) -> list:
    """
    Evaluate multiple kernels in batch.

    Args:
        generated_codes: List of generated code strings
        benchmark_refs: List of benchmark references (same length as generated_codes)
        hardware_config: Shared hardware config for all evaluations
        **kwargs: Additional args passed to evaluate_kernel()

    Returns:
        List of EvaluationResult objects
    """
    if len(generated_codes) != len(benchmark_refs):
        raise ValueError("generated_codes and benchmark_refs must have same length")

    results = []
    for code, ref in zip(generated_codes, benchmark_refs):
        result = evaluate_kernel(code, ref, hardware_config, **kwargs)
        results.append(result)

    return results


# Convenience functions for listing available benchmarks

def list_available_benchmarks() -> Dict[str, list]:
    """
    List all available benchmark references.

    Returns:
        Dict with keys for each BenchmarkType, values are lists of references.
    """
    # TODO: Implement by scanning benchmarks/ directory
    return {
        "kernelbench": [
            "kernelbench:level1:1",
            "kernelbench:level1:2",
            # ... populated dynamically
        ],
        "jaxkernelbench": [
            "jaxkernelbench:level1:1",
            # ... populated dynamically
        ],
        "real_workloads": [
            "real_workloads:llama3:gqa",
            "real_workloads:llama3:rope",
            "real_workloads:llama3:swiglu",
            "real_workloads:gemma3:sliding",
            "real_workloads:mixtral:moe",
            "real_workloads:deepseek_v3:mla",
        ],
    }


if __name__ == "__main__":
    # Quick test
    print("Available benchmarks:")
    for btype, refs in list_available_benchmarks().items():
        print(f"  {btype}:")
        for ref in refs[:3]:
            print(f"    - {ref}")

    print("\nTesting evaluate_kernel (placeholder):")
    result = evaluate_kernel(
        generated_code="def pallas_kernel(x): return x",
        benchmark_ref="real_workloads:llama3:gqa",
    )
    print(f"  Result: correct={result.correct}, error={result.error}")
