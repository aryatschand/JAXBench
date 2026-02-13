"""
Dense Projections - Extracted from MaxText

Linear transformations used throughout LLMs:
- QKV projections in attention
- MLP layers (gate, up, down projections)
- Output projections

Source: MaxText/layers/linears.py
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import time


def dense_projection(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Basic dense/linear projection.

    Args:
        x: Input [..., in_features]
        kernel: Weight matrix [in_features, out_features]
        bias: Optional bias [out_features]

    Returns:
        Output [..., out_features]
    """
    output = jnp.dot(x, kernel)
    if bias is not None:
        output = output + bias
    return output


def fused_qkv_projection(
    x: jnp.ndarray,
    qkv_kernel: jnp.ndarray,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fused Q, K, V projection (common in efficient implementations).

    Args:
        x: Input [batch, seq_len, hidden_dim]
        qkv_kernel: Fused weights [hidden_dim, (num_heads + 2*num_kv_heads) * head_dim]
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (GQA)
        head_dim: Dimension per head

    Returns:
        q: [batch, seq_len, num_heads, head_dim]
        k: [batch, seq_len, num_kv_heads, head_dim]
        v: [batch, seq_len, num_kv_heads, head_dim]
    """
    batch, seq_len, _ = x.shape

    # Project
    qkv = jnp.dot(x, qkv_kernel)

    # Split
    q_dim = num_heads * head_dim
    k_dim = num_kv_heads * head_dim
    v_dim = num_kv_heads * head_dim

    q = qkv[..., :q_dim].reshape(batch, seq_len, num_heads, head_dim)
    k = qkv[..., q_dim:q_dim+k_dim].reshape(batch, seq_len, num_kv_heads, head_dim)
    v = qkv[..., q_dim+k_dim:].reshape(batch, seq_len, num_kv_heads, head_dim)

    return q, k, v


def swiglu_mlp(
    x: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    up_kernel: jnp.ndarray,
    down_kernel: jnp.ndarray,
) -> jnp.ndarray:
    """
    SwiGLU MLP block (used in Llama, Gemma, etc.).

    Args:
        x: Input [batch, seq_len, hidden_dim]
        gate_kernel: [hidden_dim, intermediate_dim]
        up_kernel: [hidden_dim, intermediate_dim]
        down_kernel: [intermediate_dim, hidden_dim]

    Returns:
        Output [batch, seq_len, hidden_dim]
    """
    gate = jnp.dot(x, gate_kernel)
    up = jnp.dot(x, up_kernel)

    # SwiGLU: silu(gate) * up
    hidden = jax.nn.silu(gate) * up

    # Down projection
    output = jnp.dot(hidden, down_kernel)

    return output


# =============================================================================
# Problem Sizes from Real LLMs
# =============================================================================

# QKV Projections
LLAMA_8B_QKV = {
    'name': 'Llama-8B-QKV',
    'type': 'qkv',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 4096,
    'num_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
}

LLAMA_70B_QKV = {
    'name': 'Llama-70B-QKV',
    'type': 'qkv',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 8192,
    'num_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
}

# MLP Blocks
LLAMA_8B_MLP = {
    'name': 'Llama-8B-MLP',
    'type': 'mlp',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 4096,
    'intermediate_dim': 14336,
}

LLAMA_70B_MLP = {
    'name': 'Llama-70B-MLP',
    'type': 'mlp',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 8192,
    'intermediate_dim': 28672,
}

GEMMA_27B_MLP = {
    'name': 'Gemma-27B-MLP',
    'type': 'mlp',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 4608,
    'intermediate_dim': 36864,
}

DEEPSEEK_MLP = {
    'name': 'DeepSeek-V3-MLP',
    'type': 'mlp',
    'batch': 1,
    'seq_len': 2048,
    'hidden_dim': 7168,
    'intermediate_dim': 18432,
}

# Batched variants
LLAMA_8B_MLP_BATCH32 = {
    'name': 'Llama-8B-MLP-B32',
    'type': 'mlp',
    'batch': 32,
    'seq_len': 512,
    'hidden_dim': 4096,
    'intermediate_dim': 14336,
}

# Output projection (logits)
LLAMA_8B_LOGITS = {
    'name': 'Llama-8B-Logits',
    'type': 'linear',
    'batch': 1,
    'seq_len': 2048,
    'in_features': 4096,
    'out_features': 128256,  # Llama 3 vocab
}

LLAMA_70B_LOGITS = {
    'name': 'Llama-70B-Logits',
    'type': 'linear',
    'batch': 1,
    'seq_len': 2048,
    'in_features': 8192,
    'out_features': 128256,
}

PROBLEM_SIZES = [
    LLAMA_8B_QKV,
    LLAMA_70B_QKV,
    LLAMA_8B_MLP,
    LLAMA_70B_MLP,
    GEMMA_27B_MLP,
    DEEPSEEK_MLP,
    LLAMA_8B_MLP_BATCH32,
    LLAMA_8B_LOGITS,
    LLAMA_70B_LOGITS,
]


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_qkv(config: dict, warmup: int = 5, iters: int = 20):
    """Benchmark fused QKV projection."""
    batch = config['batch']
    seq_len = config['seq_len']
    hidden_dim = config['hidden_dim']
    num_heads = config['num_heads']
    num_kv_heads = config['num_kv_heads']
    head_dim = config['head_dim']

    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    x = jax.random.normal(k1, (batch, seq_len, hidden_dim), dtype=jnp.bfloat16)
    kernel = jax.random.normal(k2, (hidden_dim, qkv_dim), dtype=jnp.bfloat16)

    qkv_fn = jax.jit(lambda x: fused_qkv_projection(
        x, kernel, num_heads, num_kv_heads, head_dim))

    for _ in range(warmup):
        q, k, v = qkv_fn(x)
        q.block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        q, k, v = qkv_fn(x)
        q.block_until_ready()
    end = time.perf_counter()

    time_ms = (end - start) / iters * 1000
    flops = 2 * batch * seq_len * hidden_dim * qkv_dim
    tflops = flops / time_ms / 1e9

    return {
        'config': config['name'],
        'time_ms': time_ms,
        'tflops': tflops,
    }


def benchmark_mlp(config: dict, warmup: int = 5, iters: int = 20):
    """Benchmark SwiGLU MLP."""
    batch = config['batch']
    seq_len = config['seq_len']
    hidden_dim = config['hidden_dim']
    intermediate_dim = config['intermediate_dim']

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    x = jax.random.normal(keys[0], (batch, seq_len, hidden_dim), dtype=jnp.bfloat16)
    gate_kernel = jax.random.normal(keys[1], (hidden_dim, intermediate_dim), dtype=jnp.bfloat16)
    up_kernel = jax.random.normal(keys[2], (hidden_dim, intermediate_dim), dtype=jnp.bfloat16)
    down_kernel = jax.random.normal(keys[3], (intermediate_dim, hidden_dim), dtype=jnp.bfloat16)

    mlp_fn = jax.jit(lambda x: swiglu_mlp(x, gate_kernel, up_kernel, down_kernel))

    for _ in range(warmup):
        output = mlp_fn(x)
        output.block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        output = mlp_fn(x)
        output.block_until_ready()
    end = time.perf_counter()

    time_ms = (end - start) / iters * 1000
    # 2 up projections + 1 down projection
    flops = 2 * batch * seq_len * (2 * hidden_dim * intermediate_dim + intermediate_dim * hidden_dim)
    tflops = flops / time_ms / 1e9

    return {
        'config': config['name'],
        'time_ms': time_ms,
        'tflops': tflops,
    }


def benchmark_linear(config: dict, warmup: int = 5, iters: int = 20):
    """Benchmark linear projection."""
    batch = config['batch']
    seq_len = config['seq_len']
    in_features = config['in_features']
    out_features = config['out_features']

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    x = jax.random.normal(k1, (batch, seq_len, in_features), dtype=jnp.bfloat16)
    kernel = jax.random.normal(k2, (in_features, out_features), dtype=jnp.bfloat16)

    linear_fn = jax.jit(lambda x: dense_projection(x, kernel))

    for _ in range(warmup):
        output = linear_fn(x)
        output.block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        output = linear_fn(x)
        output.block_until_ready()
    end = time.perf_counter()

    time_ms = (end - start) / iters * 1000
    flops = 2 * batch * seq_len * in_features * out_features
    tflops = flops / time_ms / 1e9

    return {
        'config': config['name'],
        'time_ms': time_ms,
        'tflops': tflops,
    }


def run_all_benchmarks():
    """Run benchmarks for all problem sizes."""
    print("=" * 80)
    print("DENSE PROJECTION BENCHMARK (Real LLM Sizes)")
    print("=" * 80)

    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
    except:
        pass

    print()
    print(f"{'Config':<25} | {'Time (ms)':>10} | {'TFLOPS':>8}")
    print("-" * 50)

    results = []
    for config in PROBLEM_SIZES:
        try:
            if config['type'] == 'qkv':
                result = benchmark_qkv(config)
            elif config['type'] == 'mlp':
                result = benchmark_mlp(config)
            else:
                result = benchmark_linear(config)
            results.append(result)
            print(f"{result['config']:<25} | {result['time_ms']:>10.2f} | {result['tflops']:>8.1f}")
        except Exception as e:
            print(f"{config['name']:<25} | {'FAILED':>10} | {str(e)[:15]}")

    print("-" * 50)
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
