"""Llama-3.1-8B — full transformer decoder in pure JAX.

Architecture: GQA attention + SwiGLU MLP + RoPE + RMSNorm
All compute kernels go through the kernel registry for swappability.

Supports:
  - Prefill: process full prompt in parallel
  - Decode: autoregressive generation with KV cache
"""

import jax
import jax.numpy as jnp
from inference_speedup.kernels import get_kernel


def init_weights(config, rng, num_layers=None):
    """Initialize random weights matching Llama3-8B architecture."""
    n_layers = num_layers or config['eval_layers']
    D = config['d_model']
    Hq = config['num_heads']
    Hkv = config['num_kv_heads']
    Dh = config['head_dim']
    FFN = config['ffn_dim']
    V = config['vocab_size']

    def make(rng, shape, scale=0.02):
        return jax.random.normal(rng, shape, dtype=jnp.bfloat16) * scale

    keys = jax.random.split(rng, n_layers * 7 + 3)
    ki = iter(range(len(keys)))

    layers = []
    for _ in range(n_layers):
        layer = {
            'attn_norm': jnp.ones(D, dtype=jnp.bfloat16),
            'wq': make(keys[next(ki)], (D, Hq * Dh)),
            'wk': make(keys[next(ki)], (D, Hkv * Dh)),
            'wv': make(keys[next(ki)], (D, Hkv * Dh)),
            'wo': make(keys[next(ki)], (Hq * Dh, D)),
            'ffn_norm': jnp.ones(D, dtype=jnp.bfloat16),
            'w_gate': make(keys[next(ki)], (D, FFN)),
            'w_up': make(keys[next(ki)], (D, FFN)),
            'w_down': make(keys[next(ki)], (FFN, D)),
        }
        layers.append(layer)

    weights = {
        'embed': make(keys[next(ki)], (V, D)),
        'final_norm': jnp.ones(D, dtype=jnp.bfloat16),
        'lm_head': make(keys[next(ki)], (D, V)),
        'layers': layers,
    }
    return weights


def prefill(weights, token_ids, config):
    """Full forward pass for prefill (process entire prompt).

    Args:
        weights: weight dict from init_weights()
        token_ids: (B, S) int32 token IDs
        config: model config dict
    Returns:
        logits: (B, S, V) — next-token logits for each position
        kv_cache: list of (cache_k, cache_v) per layer
    """
    B, S = token_ids.shape
    Hq = config['num_heads']
    Hkv = config['num_kv_heads']
    Dh = config['head_dim']
    eps = config['rms_norm_eps']

    x = get_kernel('token_embed')(token_ids, weights['embed'])
    positions = jnp.arange(S)

    kv_cache = []
    for layer in weights['layers']:
        # Pre-attention norm
        h = get_kernel('rmsnorm')(x, layer['attn_norm'], eps=eps)

        # QKV projections
        q = (h @ layer['wq']).reshape(B, S, Hq, Dh)
        k = (h @ layer['wk']).reshape(B, S, Hkv, Dh)
        v = (h @ layer['wv']).reshape(B, S, Hkv, Dh)

        # RoPE
        q, k = get_kernel('rope')(q, k, positions, theta=config['rope_theta'])

        # GQA attention
        attn_out = get_kernel('gqa_attention')(q, k, v)
        attn_out = attn_out.reshape(B, S, Hq * Dh)

        # Output projection + residual
        x = x + attn_out @ layer['wo']

        # Store KV cache for decode (B, Hkv, S, Dh)
        cache_k = k.transpose(0, 2, 1, 3)
        cache_v = v.transpose(0, 2, 1, 3)
        kv_cache.append((cache_k, cache_v))

        # Pre-MLP norm + SwiGLU + residual
        h = get_kernel('rmsnorm')(x, layer['ffn_norm'], eps=eps)
        x = x + get_kernel('swiglu_mlp')(h, layer['w_gate'], layer['w_up'], layer['w_down'])

    # Final norm + LM head
    x = get_kernel('rmsnorm')(x, weights['final_norm'], eps=eps)
    logits = x @ weights['lm_head']
    return logits, kv_cache


def decode_step(weights, token_id, kv_cache, pos, config):
    """Single autoregressive decode step with KV cache.

    Args:
        weights: weight dict
        token_id: (B, 1) int32
        kv_cache: list of (cache_k, cache_v), each (B, Hkv, max_seq, Dh)
        pos: scalar int — current sequence position
        config: model config dict
    Returns:
        logits: (B, 1, V), updated_kv_cache
    """
    B = token_id.shape[0]
    Hq = config['num_heads']
    Hkv = config['num_kv_heads']
    Dh = config['head_dim']
    eps = config['rms_norm_eps']

    x = get_kernel('token_embed')(token_id, weights['embed'])
    positions = jnp.array([pos])

    new_kv_cache = []
    for i, layer in enumerate(weights['layers']):
        h = get_kernel('rmsnorm')(x, layer['attn_norm'], eps=eps)

        q = (h @ layer['wq']).reshape(B, 1, Hq, Dh)
        k = (h @ layer['wk']).reshape(B, 1, Hkv, Dh)
        v = (h @ layer['wv']).reshape(B, 1, Hkv, Dh)

        q, k = get_kernel('rope')(q, k, positions, theta=config['rope_theta'])

        cache_k, cache_v = kv_cache[i]
        attn_out, cache_k, cache_v = get_kernel('gqa_attention_decode')(
            q, k, v, cache_k, cache_v, pos
        )
        attn_out = attn_out.reshape(B, 1, Hq * Dh)
        x = x + attn_out @ layer['wo']
        new_kv_cache.append((cache_k, cache_v))

        h = get_kernel('rmsnorm')(x, layer['ffn_norm'], eps=eps)
        x = x + get_kernel('swiglu_mlp')(h, layer['w_gate'], layer['w_up'], layer['w_down'])

    x = get_kernel('rmsnorm')(x, weights['final_norm'], eps=eps)
    logits = x @ weights['lm_head']
    return logits, new_kv_cache


def init_kv_cache(config, batch_size, max_seq_len, num_layers=None):
    """Allocate empty KV cache for autoregressive decode."""
    n_layers = num_layers or config['eval_layers']
    Hkv = config['num_kv_heads']
    Dh = config['head_dim']
    cache = []
    for _ in range(n_layers):
        cache_k = jnp.zeros((batch_size, Hkv, max_seq_len, Dh), dtype=jnp.bfloat16)
        cache_v = jnp.zeros((batch_size, Hkv, max_seq_len, Dh), dtype=jnp.bfloat16)
        cache.append((cache_k, cache_v))
    return cache
