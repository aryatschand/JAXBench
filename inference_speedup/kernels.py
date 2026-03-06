"""Kernel registry — vanilla JAX implementations + swap mechanism.

Every compute kernel used in the models is registered here. The registry allows
swapping vanilla JAX implementations with optimized Pallas kernels, then re-JITting
the model to measure the impact on tokens/s.

Usage:
    from inference_speedup.kernels import get_kernel, swap_kernel, reset_kernels

    # Use vanilla
    fn = get_kernel('rmsnorm')
    out = fn(x, weight, eps=1e-6)

    # Swap in optimized version
    from inference_speedup.pallas_kernels import pallas_rmsnorm
    swap_kernel('rmsnorm', pallas_rmsnorm)

    # Reset to vanilla
    reset_kernels()
"""

import jax
import jax.numpy as jnp
from functools import partial

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_vanilla_registry = {}
_active_registry = {}


def _register_vanilla(name, fn):
    """Register a vanilla JAX kernel implementation."""
    _vanilla_registry[name] = fn
    _active_registry[name] = fn


def get_kernel(name):
    """Get the currently active kernel implementation."""
    return _active_registry[name]


def swap_kernel(name, fn):
    """Swap a kernel with an optimized implementation."""
    if name not in _vanilla_registry:
        raise KeyError(f"Unknown kernel: {name}. Available: {list(_vanilla_registry)}")
    _active_registry[name] = fn


def reset_kernels():
    """Reset all kernels to vanilla JAX implementations."""
    _active_registry.update(_vanilla_registry)


def list_kernels():
    """Return dict of kernel names → whether they are currently swapped."""
    return {
        name: (_active_registry[name] is not _vanilla_registry[name])
        for name in _vanilla_registry
    }


# ---------------------------------------------------------------------------
# Vanilla JAX kernel implementations
# ---------------------------------------------------------------------------

def vanilla_rmsnorm(x, weight, eps=1e-6):
    """RMS normalization: x * rsqrt(mean(x^2) + eps) * weight."""
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / rms).astype(x.dtype) * weight


def vanilla_rope(q, k, positions, theta=500_000.0):
    """Rotary position embeddings applied to query and key tensors.

    Args:
        q: (B, S, H, D) query tensor
        k: (B, S, Hkv, D) key tensor
        positions: (S,) position indices
        theta: RoPE frequency base
    Returns:
        (q_rotated, k_rotated) with same shapes
    """
    D = q.shape[-1]
    half_d = D // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, half_d, dtype=jnp.float32) / half_d))
    angles = jnp.outer(positions.astype(jnp.float32), freqs)  # (S, D//2)
    cos = jnp.cos(angles)[None, :, None, :]  # (1, S, 1, D//2)
    sin = jnp.sin(angles)[None, :, None, :]

    def rotate(x):
        x1 = x[..., :half_d].astype(jnp.float32)
        x2 = x[..., half_d:].astype(jnp.float32)
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
        return jnp.concatenate([r1, r2], axis=-1).astype(x.dtype)

    return rotate(q), rotate(k)


def vanilla_gqa_attention(query, key, value, mask=None):
    """Grouped-query attention (Llama3-style).

    Args:
        query: (B, S, Hq, D)
        key:   (B, S, Hkv, D)
        value: (B, S, Hkv, D)
        mask:  (S, S) or None for causal
    Returns:
        (B, S, Hq, D)
    """
    B, S, Hq, D = query.shape
    Hkv = key.shape[2]
    G = Hq // Hkv  # group factor

    # Expand KV heads to match Q heads
    key = jnp.repeat(key, G, axis=2)      # (B, S, Hq, D)
    value = jnp.repeat(value, G, axis=2)

    # Transpose to (B, H, S, D)
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)

    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale

    # Causal mask
    if mask is None:
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn = jnp.where(mask, attn, jnp.finfo(query.dtype).min)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(query.dtype)

    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    return out.transpose(0, 2, 1, 3)  # (B, S, Hq, D)


def vanilla_gqa_attention_decode(query, key, value, kv_cache_k, kv_cache_v, cache_len):
    """GQA attention for single-token decode with KV cache.

    Args:
        query: (B, 1, Hq, D)
        key:   (B, 1, Hkv, D)   — new KV for this step
        value: (B, 1, Hkv, D)
        kv_cache_k: (B, Hkv, max_seq, D)
        kv_cache_v: (B, Hkv, max_seq, D)
        cache_len: scalar — current filled length
    Returns:
        output (B, 1, Hq, D), updated_cache_k, updated_cache_v
    """
    B, _, Hq, D = query.shape
    Hkv = key.shape[2]
    G = Hq // Hkv

    # Update cache at position cache_len
    new_k = key.transpose(0, 2, 1, 3)[:, :, 0, :]    # (B, Hkv, D)
    new_v = value.transpose(0, 2, 1, 3)[:, :, 0, :]   # (B, Hkv, D)
    kv_cache_k = kv_cache_k.at[:, :, cache_len, :].set(new_k)
    kv_cache_v = kv_cache_v.at[:, :, cache_len, :].set(new_v)

    # Expand KV heads
    k_full = jnp.repeat(kv_cache_k, G, axis=1)  # (B, Hq, max_seq, D)
    v_full = jnp.repeat(kv_cache_v, G, axis=1)

    # Attention: Q (B, Hq, 1, D) @ K (B, Hq, seq, D) -> (B, Hq, 1, seq)
    q = query.transpose(0, 2, 1, 3)  # (B, Hq, 1, D)
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k_full) * scale

    # Mask: only attend to positions <= cache_len
    positions = jnp.arange(kv_cache_k.shape[2])
    causal_mask = (positions <= cache_len)[None, None, None, :]
    attn = jnp.where(causal_mask, attn, jnp.finfo(query.dtype).min)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(query.dtype)

    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v_full)
    return out.transpose(0, 2, 1, 3), kv_cache_k, kv_cache_v


def vanilla_swiglu_mlp(x, w_gate, w_up, w_down):
    """SwiGLU MLP: (SiLU(x @ gate) * (x @ up)) @ down.

    Args:
        x:      (B, S, D)
        w_gate: (D, FFN)
        w_up:   (D, FFN)
        w_down: (FFN, D)
    Returns:
        (B, S, D)
    """
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down


def vanilla_gated_linear_attention(query, key, value, gate_logits):
    """Gated linear attention — parallel form.

    Args:
        query: (B, H, S, D)
        key:   (B, H, S, D)
        value: (B, H, S, D)
        gate_logits: (B, H, S) — per-head forget gate logits
    Returns:
        (B, H, S, D)
    """
    B, H, S, D = query.shape
    gate = jax.nn.sigmoid(gate_logits)

    # Gated causal mask via cumulative log-sum
    log_gate = jnp.log(gate + 1e-8)
    log_gate_cumsum = jnp.cumsum(log_gate, axis=-1)
    M = jnp.exp(log_gate_cumsum[:, :, :, None] - log_gate_cumsum[:, :, None, :])
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.float32))
    M = M * causal[None, None, :, :]

    scores = jnp.einsum('bhsd,bhtd->bhst',
                        query.astype(jnp.float32),
                        key.astype(jnp.float32))
    scores = scores * M

    norm = jnp.sum(jnp.abs(scores), axis=-1, keepdims=True)
    norm = jnp.maximum(norm, 1.0)
    scores = scores / norm

    return jnp.einsum('bhst,bhtd->bhsd', scores.astype(query.dtype), value)


def vanilla_gla_decode(query, key, value, gate_logits, state):
    """GLA recurrent decode: S_t = g_t * S_{t-1} + k_t^T v_t, o_t = q_t @ S_t.

    Args:
        query: (B, H, 1, D)
        key:   (B, H, 1, D)
        value: (B, H, 1, D)
        gate_logits: (B, H, 1) — forget gate logit for this step
        state: (B, H, D, D) — recurrent state
    Returns:
        output (B, H, 1, D), new_state (B, H, D, D)
    """
    gate = jax.nn.sigmoid(gate_logits[:, :, 0])  # (B, H)

    k = key[:, :, 0, :]    # (B, H, D)
    v = value[:, :, 0, :]  # (B, H, D)
    q = query[:, :, 0, :]  # (B, H, D)

    # Update state: S = g * S + k^T v
    kv = jnp.einsum('bhd,bhe->bhde', k.astype(jnp.float32), v.astype(jnp.float32))
    new_state = gate[:, :, None, None] * state + kv

    # Output: o = q @ S
    out = jnp.einsum('bhd,bhde->bhe', q.astype(jnp.float32), new_state)
    out = out.astype(query.dtype)
    return out[:, :, None, :], new_state


def vanilla_ssd_attention(query, key, value, A_log):
    """Mamba-2 SSD — parallel form with selective decay.

    Args:
        query: (B, H, S, D) — C (output projection)
        key:   (B, H, S, D) — B (input projection)
        value: (B, H, S, D) — x (hidden state)
        A_log: (B, H, S) — input-dependent decay (log-space)
    Returns:
        (B, H, S, D)
    """
    B, H, S, D = query.shape

    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)

    L = jnp.exp(log_a_cumsum[:, :, :, None] - log_a_cumsum[:, :, None, :])
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.float32))
    L = L * causal[None, None, :, :]

    scores = jnp.einsum('bhsd,bhtd->bhst',
                        query.astype(jnp.float32),
                        key.astype(jnp.float32))
    scores = scores * L

    scores_sum = jnp.sum(scores, axis=-1, keepdims=True)
    scores_sum = jnp.where(jnp.abs(scores_sum) < 1e-6, 1.0, scores_sum)
    scores = scores / jnp.maximum(jnp.abs(scores_sum), 1.0)

    return jnp.einsum('bhst,bhtd->bhsd', scores.astype(query.dtype), value)


def vanilla_ssd_decode(query, key, value, A_log, state):
    """Mamba-2 SSD recurrent decode step.

    Args:
        query: (B, H, 1, D)
        key:   (B, H, 1, D)
        value: (B, H, 1, D)
        A_log: (B, H, 1)
        state: (B, H, D, D) — SSM state
    Returns:
        output (B, H, 1, D), new_state (B, H, D, D)
    """
    a = jax.nn.sigmoid(A_log[:, :, 0].astype(jnp.float32))  # (B, H)

    k = key[:, :, 0, :]    # (B, H, D)
    v = value[:, :, 0, :]  # (B, H, D)
    q = query[:, :, 0, :]  # (B, H, D)

    kv = jnp.einsum('bhd,bhe->bhde', k.astype(jnp.float32), v.astype(jnp.float32))
    new_state = a[:, :, None, None] * state + kv

    out = jnp.einsum('bhd,bhde->bhe', q.astype(jnp.float32), new_state)
    out = out.astype(query.dtype)
    return out[:, :, None, :], new_state


def vanilla_token_embed(token_ids, embed_table):
    """Token embedding lookup with sqrt(d) scaling.

    Args:
        token_ids: (B, S) int32
        embed_table: (V, D) bfloat16
    Returns:
        (B, S, D)
    """
    D = embed_table.shape[1]
    return embed_table[token_ids] * (D ** 0.5)


# ---------------------------------------------------------------------------
# Register all vanilla kernels
# ---------------------------------------------------------------------------

_register_vanilla('rmsnorm', vanilla_rmsnorm)
_register_vanilla('rope', vanilla_rope)
_register_vanilla('gqa_attention', vanilla_gqa_attention)
_register_vanilla('gqa_attention_decode', vanilla_gqa_attention_decode)
_register_vanilla('swiglu_mlp', vanilla_swiglu_mlp)
_register_vanilla('gated_linear_attention', vanilla_gated_linear_attention)
_register_vanilla('gla_decode', vanilla_gla_decode)
_register_vanilla('ssd_attention', vanilla_ssd_attention)
_register_vanilla('ssd_decode', vanilla_ssd_decode)
_register_vanilla('token_embed', vanilla_token_embed)
