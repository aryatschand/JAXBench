"""
SwiGLU MLP for Llama 3.1

Extracted from MaxText. SwiGLU uses a gated activation:
  output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

Llama 3.1 MLP dims:
- 8B:  emb_dim=4096, mlp_dim=14336
- 70B: emb_dim=8192, mlp_dim=28672
- 405B: emb_dim=16384, mlp_dim=53248
"""

import jax
import jax.numpy as jnp
from functools import partial

# Problem sizes from real Llama 3.1 models
LLAMA_8B = {
    'name': 'Llama-3.1-8B',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 4096,
    'mlp_dim': 14336,
}

LLAMA_70B = {
    'name': 'Llama-3.1-70B',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 8192,
    'mlp_dim': 28672,
}

LLAMA_405B = {
    'name': 'Llama-3.1-405B',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 16384,
    'mlp_dim': 53248,
}

# Batched inference scenarios
LLAMA_8B_BATCH32 = {
    'name': 'Llama-8B-Batch32',
    'batch': 32,
    'seq_len': 512,
    'emb_dim': 4096,
    'mlp_dim': 14336,
}


def swiglu_mlp(
    x: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    up_kernel: jnp.ndarray,
    down_kernel: jnp.ndarray,
) -> jnp.ndarray:
    """
    SwiGLU MLP as used in Llama 3.1.

    Computes: output = (SiLU(x @ gate) * (x @ up)) @ down

    Args:
        x: [batch, seq_len, emb_dim]
        gate_kernel: [emb_dim, mlp_dim]
        up_kernel: [emb_dim, mlp_dim]
        down_kernel: [mlp_dim, emb_dim]

    Returns:
        output: [batch, seq_len, emb_dim]
    """
    # Gate projection with SiLU activation
    gate = jnp.dot(x, gate_kernel)
    gate = jax.nn.silu(gate)

    # Up projection (linear)
    up = jnp.dot(x, up_kernel)

    # Element-wise gating
    hidden = gate * up

    # Down projection
    output = jnp.dot(hidden, down_kernel)

    return output


def create_inputs(config: dict, dtype=jnp.bfloat16):
    """Create random inputs matching Llama config."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    batch = config['batch']
    seq_len = config['seq_len']
    emb_dim = config['emb_dim']
    mlp_dim = config['mlp_dim']

    x = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=dtype)
    gate_kernel = jax.random.normal(k2, (emb_dim, mlp_dim), dtype=dtype) * 0.02
    up_kernel = jax.random.normal(k3, (emb_dim, mlp_dim), dtype=dtype) * 0.02
    down_kernel = jax.random.normal(k4, (mlp_dim, emb_dim), dtype=dtype) * 0.02

    return x, gate_kernel, up_kernel, down_kernel


def benchmark_swiglu(config: dict, num_warmup: int = 5, num_iters: int = 50):
    """Benchmark SwiGLU MLP for a given config."""
    import time

    x, gate_kernel, up_kernel, down_kernel = create_inputs(config)

    # JIT compile
    mlp_fn = jax.jit(swiglu_mlp)

    # Warmup
    for _ in range(num_warmup):
        output = mlp_fn(x, gate_kernel, up_kernel, down_kernel)
        output.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output = mlp_fn(x, gate_kernel, up_kernel, down_kernel)
        output.block_until_ready()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iters * 1000

    # Calculate FLOPS
    batch = config['batch']
    seq_len = config['seq_len']
    emb_dim = config['emb_dim']
    mlp_dim = config['mlp_dim']

    # gate: batch * seq * emb * mlp * 2
    # up: batch * seq * emb * mlp * 2
    # down: batch * seq * mlp * emb * 2
    # Total: 3 matmuls
    flops = batch * seq_len * emb_dim * mlp_dim * 2 * 3
    tflops = flops / (avg_time_ms / 1000) / 1e12

    return {
        'config': config['name'],
        'time_ms': avg_time_ms,
        'tflops': tflops,
        'shape': list(output.shape),
    }


if __name__ == '__main__':
    print("=" * 80)
    print("LLAMA 3.1 SwiGLU MLP BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    configs = [LLAMA_8B, LLAMA_70B, LLAMA_405B, LLAMA_8B_BATCH32]

    print(f"{'Config':<25} | {'Time (ms)':>10} | {'TFLOPS':>8} | Output Shape")
    print("-" * 80)

    for config in configs:
        result = benchmark_swiglu(config)
        print(f"{result['config']:<25} | {result['time_ms']:>10.2f} | {result['tflops']:>8.1f} | {result['shape']}")
