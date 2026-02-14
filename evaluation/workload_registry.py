"""
Workload Registry for Pallas Kernel Evaluation.

Defines the interface between JAX baselines and Pallas kernels.
Each workload specifies:
- Input generation function
- JAX baseline function
- Expected output shape
- Correctness tolerances
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, List
import importlib.util
from pathlib import Path


@dataclass
class WorkloadConfig:
    """Configuration for a single workload."""
    name: str
    model: str  # llama3, gemma3, mixtral, deepseek_v3
    category: str  # attention, mlp, rope, moe
    config: Dict[str, Any]
    input_generator: Callable  # () -> tuple of inputs
    baseline_fn: Callable  # (*inputs) -> output
    rtol: float = 1e-2
    atol: float = 1e-2


# Registry of all workloads
WORKLOADS: Dict[str, WorkloadConfig] = {}


def register_workload(workload: WorkloadConfig):
    """Register a workload for evaluation."""
    WORKLOADS[workload.name] = workload


def get_workload(name: str) -> WorkloadConfig:
    """Get a registered workload by name."""
    if name not in WORKLOADS:
        raise ValueError(f"Unknown workload: {name}. Available: {list(WORKLOADS.keys())}")
    return WORKLOADS[name]


def list_workloads() -> List[str]:
    """List all registered workloads."""
    return list(WORKLOADS.keys())


# ============================================================================
# Llama 3.1 Workloads
# ============================================================================

def _llama3_gqa_inputs(config):
    """Generate inputs for Llama3 GQA attention."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    batch = config['batch']
    seq_len = config['seq_len']
    num_query_heads = config['num_query_heads']
    num_kv_heads = config['num_kv_heads']
    head_dim = config['head_dim']

    query = jax.random.normal(k1, (batch, seq_len, num_query_heads, head_dim), dtype=jnp.bfloat16)
    key_tensor = jax.random.normal(k2, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value = jax.random.normal(k3, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    return (query, key_tensor, value)


def _llama3_gqa_baseline(query, key, value, num_kv_heads):
    """Llama3 GQA baseline implementation."""
    batch, seq_len, num_query_heads, head_dim = query.shape
    num_groups = num_query_heads // num_kv_heads

    # Expand KV heads
    key = jnp.repeat(key[:, :, :, None, :], num_groups, axis=3)
    key = key.reshape(batch, seq_len, num_query_heads, head_dim)
    value = jnp.repeat(value[:, :, :, None, :], num_groups, axis=3)
    value = value.reshape(batch, seq_len, num_query_heads, head_dim)

    # Transpose for attention
    query = query.transpose(0, 2, 1, 3)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)

    scale = head_dim ** -0.5
    attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale

    # Causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attn_weights = jnp.where(mask, attn_weights, -1e9)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)
    output = output.transpose(0, 2, 1, 3)

    return output


def _llama3_rope_inputs(config):
    """Generate inputs for Llama3 RoPE."""
    key = jax.random.PRNGKey(42)

    batch = config['batch']
    seq_len = config['seq_len']
    num_heads = config['num_heads']
    head_dim = config['head_dim']

    x = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
    return (x,)


def _llama3_rope_baseline(x, theta=500000.0):
    """Llama3 RoPE baseline implementation."""
    batch, seq_len, num_heads, head_dim = x.shape

    # Compute frequencies
    dim_pairs = head_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, dim_pairs, dtype=jnp.float32) / dim_pairs))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, freqs)

    cos = jnp.cos(angles).astype(x.dtype)
    sin = jnp.sin(angles).astype(x.dtype)

    # Reshape for broadcasting
    cos = cos[None, :, None, :]  # [1, seq, 1, dim//2]
    sin = sin[None, :, None, :]

    # Apply rotation
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    output = jnp.concatenate([rotated_x1, rotated_x2], axis=-1)
    return output


def _llama3_swiglu_inputs(config):
    """Generate inputs for Llama3 SwiGLU MLP."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    batch = config['batch']
    seq_len = config['seq_len']
    emb_dim = config['emb_dim']
    mlp_dim = config['mlp_dim']

    x = jax.random.normal(k1, (batch, seq_len, emb_dim), dtype=jnp.bfloat16)
    gate_kernel = jax.random.normal(k2, (emb_dim, mlp_dim), dtype=jnp.bfloat16) * 0.02
    up_kernel = jax.random.normal(k3, (emb_dim, mlp_dim), dtype=jnp.bfloat16) * 0.02
    down_kernel = jax.random.normal(k4, (mlp_dim, emb_dim), dtype=jnp.bfloat16) * 0.02

    return (x, gate_kernel, up_kernel, down_kernel)


def _llama3_swiglu_baseline(x, gate_kernel, up_kernel, down_kernel):
    """Llama3 SwiGLU MLP baseline implementation."""
    gate = jax.nn.silu(jnp.dot(x, gate_kernel))
    up = jnp.dot(x, up_kernel)
    hidden = gate * up
    output = jnp.dot(hidden, down_kernel)
    return output


# ============================================================================
# Register Llama 3.1 Workloads
# ============================================================================

# Llama 3.1 8B GQA
_llama3_8b_gqa_config = {
    'batch': 1, 'seq_len': 2048, 'num_query_heads': 32,
    'num_kv_heads': 8, 'head_dim': 128
}
register_workload(WorkloadConfig(
    name='llama3_8b_gqa',
    model='llama3',
    category='attention',
    config=_llama3_8b_gqa_config,
    input_generator=lambda: _llama3_gqa_inputs(_llama3_8b_gqa_config),
    baseline_fn=lambda q, k, v: _llama3_gqa_baseline(q, k, v, num_kv_heads=8),
    rtol=1e-2, atol=1e-2,
))

# Llama 3.1 70B GQA
_llama3_70b_gqa_config = {
    'batch': 1, 'seq_len': 2048, 'num_query_heads': 64,
    'num_kv_heads': 8, 'head_dim': 128
}
register_workload(WorkloadConfig(
    name='llama3_70b_gqa',
    model='llama3',
    category='attention',
    config=_llama3_70b_gqa_config,
    input_generator=lambda: _llama3_gqa_inputs(_llama3_70b_gqa_config),
    baseline_fn=lambda q, k, v: _llama3_gqa_baseline(q, k, v, num_kv_heads=8),
    rtol=1e-2, atol=1e-2,
))

# Llama 3.1 8B RoPE
_llama3_8b_rope_config = {
    'batch': 1, 'seq_len': 2048, 'num_heads': 32, 'head_dim': 128
}
register_workload(WorkloadConfig(
    name='llama3_8b_rope',
    model='llama3',
    category='rope',
    config=_llama3_8b_rope_config,
    input_generator=lambda: _llama3_rope_inputs(_llama3_8b_rope_config),
    baseline_fn=lambda x: _llama3_rope_baseline(x, theta=500000.0),
    rtol=1e-2, atol=1e-2,
))

# Llama 3.1 8B SwiGLU
_llama3_8b_mlp_config = {
    'batch': 1, 'seq_len': 2048, 'emb_dim': 4096, 'mlp_dim': 14336
}
register_workload(WorkloadConfig(
    name='llama3_8b_swiglu',
    model='llama3',
    category='mlp',
    config=_llama3_8b_mlp_config,
    input_generator=lambda: _llama3_swiglu_inputs(_llama3_8b_mlp_config),
    baseline_fn=_llama3_swiglu_baseline,
    rtol=1e-2, atol=1e-2,
))

# Llama 3.1 70B SwiGLU
_llama3_70b_mlp_config = {
    'batch': 1, 'seq_len': 2048, 'emb_dim': 8192, 'mlp_dim': 28672
}
register_workload(WorkloadConfig(
    name='llama3_70b_swiglu',
    model='llama3',
    category='mlp',
    config=_llama3_70b_mlp_config,
    input_generator=lambda: _llama3_swiglu_inputs(_llama3_70b_mlp_config),
    baseline_fn=_llama3_swiglu_baseline,
    rtol=1e-2, atol=1e-2,
))


# ============================================================================
# Gemma 3 Workloads
# ============================================================================

def _gemma3_sliding_inputs(config):
    """Generate inputs for Gemma3 sliding window attention."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    batch = config['batch']
    seq_len = config['seq_len']
    num_query_heads = config['num_query_heads']
    num_kv_heads = config['num_kv_heads']
    head_dim = config['head_dim']

    query = jax.random.normal(k1, (batch, seq_len, num_query_heads, head_dim), dtype=jnp.bfloat16)
    key_tensor = jax.random.normal(k2, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    value = jax.random.normal(k3, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.bfloat16)

    return (query, key_tensor, value)


def _gemma3_sliding_baseline(query, key, value, num_kv_heads, window_size=4096, soft_cap=50.0):
    """Gemma3 sliding window attention with QK norm and soft capping."""
    batch, seq_len, num_query_heads, head_dim = query.shape
    num_groups = num_query_heads // num_kv_heads

    # QK normalize
    query = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + 1e-6)
    key = key / (jnp.linalg.norm(key, axis=-1, keepdims=True) + 1e-6)

    # Expand KV heads
    key = jnp.repeat(key[:, :, :, None, :], num_groups, axis=3)
    key = key.reshape(batch, seq_len, num_query_heads, head_dim)
    value = jnp.repeat(value[:, :, :, None, :], num_groups, axis=3)
    value = value.reshape(batch, seq_len, num_query_heads, head_dim)

    # Transpose
    query = query.transpose(0, 2, 1, 3)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)

    scale = head_dim ** -0.5
    attn_logits = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale

    # Soft cap
    attn_logits = soft_cap * jnp.tanh(attn_logits / soft_cap)

    # Sliding window mask
    positions = jnp.arange(seq_len)
    distances = positions[:, None] - positions[None, :]
    mask = (distances >= 0) & (distances < window_size)
    attn_logits = jnp.where(mask, attn_logits, -1e9)

    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)
    output = output.transpose(0, 2, 1, 3)

    return output


# Gemma 3 27B Sliding Window
_gemma3_27b_config = {
    'batch': 1, 'seq_len': 2048, 'num_query_heads': 32,
    'num_kv_heads': 16, 'head_dim': 144, 'window_size': 4096
}
register_workload(WorkloadConfig(
    name='gemma3_27b_sliding',
    model='gemma3',
    category='attention',
    config=_gemma3_27b_config,
    input_generator=lambda: _gemma3_sliding_inputs(_gemma3_27b_config),
    baseline_fn=lambda q, k, v: _gemma3_sliding_baseline(q, k, v, num_kv_heads=16),
    rtol=1e-2, atol=1e-2,
))
