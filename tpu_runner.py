"""
Modal TPU Runner for JAXBench.

This script provides the main entry point for running JAX workloads on TPUs via Modal.

Usage:
    modal run tpu_runner.py
    modal run tpu_runner.py --tpu-type v4-8
    modal run tpu_runner.py::run_benchmark
"""

import modal
from typing import Optional

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("jaxbench-tpu")

# TPU-optimized image with JAX
tpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "jax[tpu]==0.4.30",
        extra_options="-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    )
    .pip_install(
        "flax>=0.8.0",
        "optax>=0.2.0", 
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
    )
)


# ============================================================================
# TPU Functions
# ============================================================================

@app.function(
    image=tpu_image,
    cloud="gcp",
    accelerator="tpu-v4-8",
    timeout=3600,
)
def test_tpu_connection():
    """Test basic TPU connectivity and JAX setup."""
    import jax
    import jax.numpy as jnp
    
    print("=" * 60)
    print("TPU Connection Test")
    print("=" * 60)
    
    # Check devices
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device}")
    
    # Check TPU backend
    print(f"\nDefault backend: {jax.default_backend()}")
    print(f"Device count: {jax.device_count()}")
    print(f"Local device count: {jax.local_device_count()}")
    
    # Simple computation test
    print("\nRunning simple computation test...")
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    print(f"Matrix multiply result shape: {y.shape}")
    print(f"Result sum: {float(y.sum())}")
    
    print("\n✅ TPU connection test PASSED")
    
    return {
        "num_devices": len(devices),
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
    }


@app.function(
    image=tpu_image,
    cloud="gcp",
    accelerator="tpu-v4-8",
    timeout=7200,
)
def run_benchmark(
    benchmark_name: str = "matmul",
    size: int = 4096,
    num_iterations: int = 100,
):
    """
    Run a JAX benchmark on TPU.
    
    Args:
        benchmark_name: Name of benchmark ("matmul", "attention", "mlp")
        size: Matrix/sequence size
        num_iterations: Number of iterations for timing
    
    Returns:
        Benchmark results dict
    """
    import jax
    import jax.numpy as jnp
    import time
    
    print(f"Running benchmark: {benchmark_name}")
    print(f"Size: {size}, Iterations: {num_iterations}")
    print(f"Devices: {jax.device_count()}")
    
    # Compile and warmup
    if benchmark_name == "matmul":
        @jax.jit
        def compute(x, y):
            return jnp.dot(x, y)
        
        x = jnp.ones((size, size), dtype=jnp.float32)
        y = jnp.ones((size, size), dtype=jnp.float32)
        args = (x, y)
        
    elif benchmark_name == "attention":
        @jax.jit
        def compute(q, k, v):
            scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(64)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum('bhqk,bhkd->bhqd', weights, v)
        
        batch, heads, seq, dim = 8, 16, size, 64
        q = jnp.ones((batch, heads, seq, dim), dtype=jnp.float32)
        k = jnp.ones((batch, heads, seq, dim), dtype=jnp.float32)
        v = jnp.ones((batch, heads, seq, dim), dtype=jnp.float32)
        args = (q, k, v)
        
    elif benchmark_name == "mlp":
        @jax.jit
        def compute(x, w1, w2):
            h = jax.nn.gelu(jnp.dot(x, w1))
            return jnp.dot(h, w2)
        
        x = jnp.ones((size, size), dtype=jnp.float32)
        w1 = jnp.ones((size, size * 4), dtype=jnp.float32)
        w2 = jnp.ones((size * 4, size), dtype=jnp.float32)
        args = (x, w1, w2)
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        result = compute(*args)
        result.block_until_ready()
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        result = compute(*args)
        result.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    import numpy as np
    times = np.array(times)
    
    results = {
        "benchmark": benchmark_name,
        "size": size,
        "num_iterations": num_iterations,
        "mean_ms": float(times.mean() * 1000),
        "std_ms": float(times.std() * 1000),
        "min_ms": float(times.min() * 1000),
        "max_ms": float(times.max() * 1000),
        "throughput_gflops": None,  # Could calculate for matmul
    }
    
    print(f"\nResults:")
    print(f"  Mean: {results['mean_ms']:.3f} ms")
    print(f"  Std:  {results['std_ms']:.3f} ms")
    print(f"  Min:  {results['min_ms']:.3f} ms")
    print(f"  Max:  {results['max_ms']:.3f} ms")
    
    return results


@app.function(
    image=tpu_image,
    cloud="gcp", 
    accelerator="tpu-v4-8",
    timeout=3600,
)
def run_custom_workload(code: str):
    """
    Run custom JAX code on TPU.
    
    Args:
        code: Python code string to execute
    
    Returns:
        Result of execution
    """
    import jax
    import jax.numpy as jnp
    
    # Create execution namespace with common imports
    namespace = {
        "jax": jax,
        "jnp": jnp,
        "np": __import__("numpy"),
    }
    
    # Execute the code
    exec(code, namespace)
    
    # Return result if defined
    return namespace.get("result", None)


# ============================================================================
# CLI Entry Point
# ============================================================================

@app.local_entrypoint()
def main(
    test: bool = False,
    benchmark: Optional[str] = None,
    size: int = 4096,
    iterations: int = 100,
):
    """
    JAXBench TPU Runner.
    
    Args:
        test: Run TPU connection test
        benchmark: Run a benchmark (matmul, attention, mlp)
        size: Benchmark size
        iterations: Number of iterations
    """
    if test or (benchmark is None):
        print("Running TPU connection test...")
        result = test_tpu_connection.remote()
        print(f"\nTest result: {result}")
    
    if benchmark:
        print(f"\nRunning {benchmark} benchmark...")
        result = run_benchmark.remote(
            benchmark_name=benchmark,
            size=size,
            num_iterations=iterations,
        )
        print(f"\nBenchmark result: {result}")


if __name__ == "__main__":
    # For local testing (won't have TPU access)
    print("Use 'modal run tpu_runner.py' to run on TPU")

