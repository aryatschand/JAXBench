#!/usr/bin/env python3
"""Standalone TPU evaluation script — runs on TPU VM without module dependencies.

This script is self-contained: it includes all configs, kernels, models,
and evaluation logic in a single file so it can be SCP'd to a TPU VM.

Usage:
    PJRT_DEVICE=TPU python3 tpu_eval_standalone.py --model llama3_8b
    PJRT_DEVICE=TPU python3 tpu_eval_standalone.py --all
    PJRT_DEVICE=TPU python3 tpu_eval_standalone.py --model llama3_8b --optimized rmsnorm
"""

import argparse
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# ===================================================================
# CONFIGS
# ===================================================================

LLAMA3_8B = {
    'name': 'llama3_8b',
    'display_name': 'Llama-3.1-8B',
    'model_type': 'llama3',
    'vocab_size': 128256,
    'd_model': 4096,
    'num_layers': 32,
    'eval_layers': 8,
    'num_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'ffn_dim': 14336,
    'rope_theta': 500_000.0,
    'rms_norm_eps': 1e-6,
    'block_kernels': {'rmsnorm': 2, 'rope': 1, 'gqa_attention': 1, 'swiglu_mlp': 1},
    'forward_kernels': {'token_embed': 1, 'rmsnorm': 1},
}

GLA_1_3B = {
    'name': 'gla_1_3b',
    'display_name': 'GLA-1.3B',
    'model_type': 'gla',
    'vocab_size': 50304,
    'd_model': 2048,
    'num_layers': 24,
    'eval_layers': 24,
    'num_heads': 16,
    'head_dim': 128,
    'gate_dim': 16,
    'ffn_dim': 5632,
    'rms_norm_eps': 1e-6,
    'block_kernels': {'rmsnorm': 2, 'gated_linear_attention': 1, 'swiglu_mlp': 1},
    'forward_kernels': {'token_embed': 1, 'rmsnorm': 1},
}

MAMBA2_2_7B = {
    'name': 'mamba2_2_7b',
    'display_name': 'Mamba-2-2.7B',
    'model_type': 'mamba2',
    'vocab_size': 50304,
    'd_model': 2560,
    'num_layers': 64,
    'eval_layers': 32,
    'num_heads': 64,
    'head_dim': 80,
    'd_state': 128,
    'd_conv': 4,
    'expand': 2,
    'rms_norm_eps': 1e-6,
    'block_kernels': {'rmsnorm': 1, 'ssd_attention': 1},
    'forward_kernels': {'token_embed': 1, 'rmsnorm': 1},
}

ALL_MODELS = {'llama3_8b': LLAMA3_8B, 'gla_1_3b': GLA_1_3B, 'mamba2_2_7b': MAMBA2_2_7B}

# ===================================================================
# KERNEL REGISTRY
# ===================================================================

_vanilla_registry = {}
_active_registry = {}

def _register_vanilla(name, fn):
    _vanilla_registry[name] = fn
    _active_registry[name] = fn

def get_kernel(name):
    return _active_registry[name]

def swap_kernel(name, fn):
    _active_registry[name] = fn

def reset_kernels():
    _active_registry.update(_vanilla_registry)

def list_kernels():
    return {n: (_active_registry[n] is not _vanilla_registry[n]) for n in _vanilla_registry}

# ===================================================================
# VANILLA KERNELS
# ===================================================================

def vanilla_rmsnorm(x, weight, eps=1e-6):
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / rms).astype(x.dtype) * weight

def vanilla_rope(q, k, positions, theta=500_000.0):
    D = q.shape[-1]
    half_d = D // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, half_d, dtype=jnp.float32) / half_d))
    angles = jnp.outer(positions.astype(jnp.float32), freqs)
    cos = jnp.cos(angles)[None, :, None, :]
    sin = jnp.sin(angles)[None, :, None, :]
    def rotate(x):
        x1, x2 = x[..., :half_d].astype(jnp.float32), x[..., half_d:].astype(jnp.float32)
        return jnp.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1).astype(x.dtype)
    return rotate(q), rotate(k)

def vanilla_gqa_attention(query, key, value, mask=None):
    B, S, Hq, D = query.shape
    Hkv = key.shape[2]
    G = Hq // Hkv
    key = jnp.repeat(key, G, axis=2)
    value = jnp.repeat(value, G, axis=2)
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    if mask is None:
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn = jnp.where(mask, attn, jnp.finfo(query.dtype).min)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(query.dtype)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    return out.transpose(0, 2, 1, 3)

def vanilla_swiglu_mlp(x, w_gate, w_up, w_down):
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down

def vanilla_gated_linear_attention(query, key, value, gate_logits):
    B, H, S, D = query.shape
    gate = jax.nn.sigmoid(gate_logits)
    log_gate = jnp.log(gate + 1e-8)
    log_gate_cumsum = jnp.cumsum(log_gate, axis=-1)
    M = jnp.exp(log_gate_cumsum[:, :, :, None] - log_gate_cumsum[:, :, None, :])
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.float32))
    M = M * causal[None, None, :, :]
    scores = jnp.einsum('bhsd,bhtd->bhst', query.astype(jnp.float32), key.astype(jnp.float32))
    scores = scores * M
    norm = jnp.maximum(jnp.sum(jnp.abs(scores), axis=-1, keepdims=True), 1.0)
    scores = scores / norm
    return jnp.einsum('bhst,bhtd->bhsd', scores.astype(query.dtype), value)

def vanilla_ssd_attention(query, key, value, A_log):
    B, H, S, D = query.shape
    a = jax.nn.sigmoid(A_log.astype(jnp.float32))
    log_a = jnp.log(a + 1e-8)
    log_a_cumsum = jnp.cumsum(log_a, axis=-1)
    L = jnp.exp(log_a_cumsum[:, :, :, None] - log_a_cumsum[:, :, None, :])
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.float32))
    L = L * causal[None, None, :, :]
    scores = jnp.einsum('bhsd,bhtd->bhst', query.astype(jnp.float32), key.astype(jnp.float32))
    scores = scores * L
    scores_sum = jnp.sum(scores, axis=-1, keepdims=True)
    scores_sum = jnp.where(jnp.abs(scores_sum) < 1e-6, 1.0, scores_sum)
    scores = scores / jnp.maximum(jnp.abs(scores_sum), 1.0)
    return jnp.einsum('bhst,bhtd->bhsd', scores.astype(query.dtype), value)

def vanilla_token_embed(token_ids, embed_table):
    D = embed_table.shape[1]
    return embed_table[token_ids] * (D ** 0.5)

# Register
for name, fn in [
    ('rmsnorm', vanilla_rmsnorm), ('rope', vanilla_rope),
    ('gqa_attention', vanilla_gqa_attention), ('swiglu_mlp', vanilla_swiglu_mlp),
    ('gated_linear_attention', vanilla_gated_linear_attention),
    ('ssd_attention', vanilla_ssd_attention), ('token_embed', vanilla_token_embed),
]:
    _register_vanilla(name, fn)

# ===================================================================
# PALLAS KERNELS
# ===================================================================

try:
    from jax.experimental import pallas as pl

    def _rmsnorm_kernel_tiled(x_ref, w_ref, o_ref, *, eps):
        """Pallas kernel: fused RMSNorm for a block of rows.

        Operates on (block_rows, D) tiles. Weight is (1, D) broadcast.
        """
        x = x_ref[...].astype(jnp.float32)   # (block_rows, D)
        w = w_ref[0, :]                        # (D,) — take row 0 of (1, D)
        sq = x * x
        mean_sq = jnp.mean(sq, axis=-1, keepdims=True)  # (block_rows, 1)
        rsqrt_val = jax.lax.rsqrt(mean_sq + eps)
        normed = x * rsqrt_val
        o_ref[...] = (normed * w).astype(o_ref.dtype)

    def pallas_rmsnorm(x, weight, eps=1e-6):
        """Pallas-optimized RMSNorm with 2D BlockSpec tiling for TPU Mosaic."""
        orig_shape = x.shape
        if x.ndim == 3:
            B, S, D = x.shape
            x_2d = x.reshape(B * S, D)
        else:
            x_2d = x
            D = x.shape[-1]

        BS = x_2d.shape[0]
        block_rows = min(8, BS)
        n_blocks = BS // block_rows

        # Handle remainder if BS not divisible by block_rows
        if BS % block_rows != 0:
            pad_rows = block_rows - (BS % block_rows)
            x_2d = jnp.pad(x_2d, ((0, pad_rows), (0, 0)))
            n_blocks = x_2d.shape[0] // block_rows

        # Reshape weight to 2D for Mosaic compatibility
        w_2d = weight[None, :]  # (1, D)

        out_2d = pl.pallas_call(
            partial(_rmsnorm_kernel_tiled, eps=eps),
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            in_specs=[
                pl.BlockSpec((block_rows, D), lambda i: (i, 0)),
                pl.BlockSpec((1, D), lambda i: (0, 0)),  # broadcast weight
            ],
            out_specs=pl.BlockSpec((block_rows, D), lambda i: (i, 0)),
            grid=(n_blocks,),
        )(x_2d, w_2d)

        # Remove padding and reshape back
        out_2d = out_2d[:BS]
        return out_2d.reshape(orig_shape)

    AVAILABLE_PALLAS = {'rmsnorm': pallas_rmsnorm}
except Exception:
    AVAILABLE_PALLAS = {}

# ===================================================================
# LLAMA3 MODEL
# ===================================================================

def llama3_init_weights(config, rng):
    n_layers = config['eval_layers']
    D, Hq, Hkv, Dh, FFN, V = (config['d_model'], config['num_heads'],
        config['num_kv_heads'], config['head_dim'], config['ffn_dim'], config['vocab_size'])
    def make(rng, shape, scale=0.02):
        return jax.random.normal(rng, shape, dtype=jnp.bfloat16) * scale
    keys = jax.random.split(rng, n_layers * 7 + 3)
    ki = iter(range(len(keys)))
    layers = []
    for _ in range(n_layers):
        layers.append({
            'attn_norm': jnp.ones(D, dtype=jnp.bfloat16),
            'wq': make(keys[next(ki)], (D, Hq * Dh)),
            'wk': make(keys[next(ki)], (D, Hkv * Dh)),
            'wv': make(keys[next(ki)], (D, Hkv * Dh)),
            'wo': make(keys[next(ki)], (Hq * Dh, D)),
            'ffn_norm': jnp.ones(D, dtype=jnp.bfloat16),
            'w_gate': make(keys[next(ki)], (D, FFN)),
            'w_up': make(keys[next(ki)], (D, FFN)),
            'w_down': make(keys[next(ki)], (FFN, D)),
        })
    return {
        'embed': make(keys[next(ki)], (V, D)),
        'final_norm': jnp.ones(D, dtype=jnp.bfloat16),
        'lm_head': make(keys[next(ki)], (D, V)),
        'layers': layers,
    }

def llama3_prefill(weights, token_ids, config):
    B, S = token_ids.shape
    Hq, Hkv, Dh = config['num_heads'], config['num_kv_heads'], config['head_dim']
    eps = config['rms_norm_eps']
    x = get_kernel('token_embed')(token_ids, weights['embed'])
    positions = jnp.arange(S)
    for layer in weights['layers']:
        h = get_kernel('rmsnorm')(x, layer['attn_norm'], eps=eps)
        q = (h @ layer['wq']).reshape(B, S, Hq, Dh)
        k = (h @ layer['wk']).reshape(B, S, Hkv, Dh)
        v = (h @ layer['wv']).reshape(B, S, Hkv, Dh)
        q, k = get_kernel('rope')(q, k, positions, theta=config['rope_theta'])
        attn_out = get_kernel('gqa_attention')(q, k, v).reshape(B, S, Hq * Dh)
        x = x + attn_out @ layer['wo']
        h = get_kernel('rmsnorm')(x, layer['ffn_norm'], eps=eps)
        x = x + get_kernel('swiglu_mlp')(h, layer['w_gate'], layer['w_up'], layer['w_down'])
    x = get_kernel('rmsnorm')(x, weights['final_norm'], eps=eps)
    return x @ weights['lm_head']

# ===================================================================
# GLA MODEL
# ===================================================================

def gla_init_weights(config, rng):
    n_layers = config['eval_layers']
    D, H, Dh, FFN, V = (config['d_model'], config['num_heads'],
        config['head_dim'], config['ffn_dim'], config['vocab_size'])
    def make(rng, shape, scale=0.02):
        return jax.random.normal(rng, shape, dtype=jnp.bfloat16) * scale
    keys = jax.random.split(rng, n_layers * 8 + 3)
    ki = iter(range(len(keys)))
    layers = []
    for _ in range(n_layers):
        layers.append({
            'attn_norm': jnp.ones(D, dtype=jnp.bfloat16),
            'wq': make(keys[next(ki)], (D, H * Dh)),
            'wk': make(keys[next(ki)], (D, H * Dh)),
            'wv': make(keys[next(ki)], (D, H * Dh)),
            'wo': make(keys[next(ki)], (H * Dh, D)),
            'w_gate': make(keys[next(ki)], (D, H)),
            'ffn_norm': jnp.ones(D, dtype=jnp.bfloat16),
            'w_ffn_gate': make(keys[next(ki)], (D, FFN)),
            'w_ffn_up': make(keys[next(ki)], (D, FFN)),
            'w_ffn_down': make(keys[next(ki)], (FFN, D)),
        })
    return {
        'embed': make(keys[next(ki)], (V, D)),
        'final_norm': jnp.ones(D, dtype=jnp.bfloat16),
        'lm_head': make(keys[next(ki)], (D, V)),
        'layers': layers,
    }

def gla_prefill(weights, token_ids, config):
    B, S = token_ids.shape
    H, Dh = config['num_heads'], config['head_dim']
    eps = config['rms_norm_eps']
    x = get_kernel('token_embed')(token_ids, weights['embed'])
    for layer in weights['layers']:
        h = get_kernel('rmsnorm')(x, layer['attn_norm'], eps=eps)
        q = (h @ layer['wq']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        k = (h @ layer['wk']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        v = (h @ layer['wv']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        gate_logits = (h @ layer['w_gate']).transpose(0, 2, 1)
        attn_out = get_kernel('gated_linear_attention')(q, k, v, gate_logits)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, H * Dh)
        x = x + attn_out @ layer['wo']
        h = get_kernel('rmsnorm')(x, layer['ffn_norm'], eps=eps)
        x = x + get_kernel('swiglu_mlp')(h, layer['w_ffn_gate'], layer['w_ffn_up'], layer['w_ffn_down'])
    x = get_kernel('rmsnorm')(x, weights['final_norm'], eps=eps)
    return x @ weights['lm_head']

# ===================================================================
# MAMBA-2 MODEL
# ===================================================================

def _causal_conv1d(x, weight, bias):
    B, S, D = x.shape
    K = weight.shape[1]
    x_padded = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0)))
    out = jnp.zeros_like(x)
    for k in range(K):
        out = out + x_padded[:, k:k + S, :] * weight[:, k][None, None, :]
    return out + bias[None, None, :]

def mamba2_init_weights(config, rng):
    n_layers = config['eval_layers']
    D, H = config['d_model'], config['num_heads']
    expand = config['expand']
    d_inner = D * expand
    V = config['vocab_size']
    d_conv = config['d_conv']
    def make(rng, shape, scale=0.02):
        return jax.random.normal(rng, shape, dtype=jnp.bfloat16) * scale
    keys = jax.random.split(rng, n_layers * 7 + 3)
    ki = iter(range(len(keys)))
    layers = []
    for _ in range(n_layers):
        layers.append({
            'norm': jnp.ones(D, dtype=jnp.bfloat16),
            'in_proj': make(keys[next(ki)], (D, 2 * d_inner)),
            'conv_weight': make(keys[next(ki)], (d_inner, d_conv)),
            'conv_bias': jnp.zeros(d_inner, dtype=jnp.bfloat16),
            'w_b': make(keys[next(ki)], (d_inner, d_inner)),
            'w_c': make(keys[next(ki)], (d_inner, d_inner)),
            'A_log': jnp.full(H, -4.0, dtype=jnp.float32),
            'w_dt': make(keys[next(ki)], (d_inner, H)),
            'dt_bias': jnp.full(H, -2.0, dtype=jnp.float32),
            'out_proj': make(keys[next(ki)], (d_inner, D)),
        })
    return {
        'embed': make(keys[next(ki)], (V, D)),
        'final_norm': jnp.ones(D, dtype=jnp.bfloat16),
        'lm_head': make(keys[next(ki)], (D, V)),
        'layers': layers,
    }

def mamba2_prefill(weights, token_ids, config):
    B, S = token_ids.shape
    H = config['num_heads']
    eps = config['rms_norm_eps']
    d_inner = config['d_model'] * config['expand']
    Dh = d_inner // H
    x = get_kernel('token_embed')(token_ids, weights['embed'])
    for layer in weights['layers']:
        h = get_kernel('rmsnorm')(x, layer['norm'], eps=eps)
        proj = h @ layer['in_proj']
        x_path, z_gate = proj[:, :, :d_inner], proj[:, :, d_inner:]
        x_conv = _causal_conv1d(x_path, layer['conv_weight'], layer['conv_bias'])
        x_conv = jax.nn.silu(x_conv)
        key = (x_conv @ layer['w_b']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        query = (x_conv @ layer['w_c']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        value = x_conv.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        dt = x_conv @ layer['w_dt'] + layer['dt_bias']
        A_log_eff = (layer['A_log'][None, None, :] * jax.nn.softplus(dt)).transpose(0, 2, 1)
        ssd_out = get_kernel('ssd_attention')(query, key, value, A_log_eff)
        ssd_out = ssd_out.transpose(0, 2, 1, 3).reshape(B, S, d_inner)
        out = (ssd_out * jax.nn.silu(z_gate)) @ layer['out_proj']
        x = x + out
    x = get_kernel('rmsnorm')(x, weights['final_norm'], eps=eps)
    return x @ weights['lm_head']

# ===================================================================
# EVALUATION
# ===================================================================

INIT_FNS = {'llama3': llama3_init_weights, 'gla': gla_init_weights, 'mamba2': mamba2_init_weights}
PREFILL_FNS = {'llama3': llama3_prefill, 'gla': gla_prefill, 'mamba2': mamba2_prefill}

def profile_kernels(config, batch_size=1, seq_len=2048, num_warmup=5, num_iters=50):
    D = config['d_model']
    n_layers = config['eval_layers']
    dtype = jnp.bfloat16
    kernel_benchmarks = {}

    # RMSNorm
    x_n = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, D), dtype=dtype)
    w_n = jnp.ones(D, dtype=dtype)
    kernel_benchmarks['rmsnorm'] = (jax.jit(lambda x, w: get_kernel('rmsnorm')(x, w, eps=config['rms_norm_eps'])), (x_n, w_n))

    # Token embed
    toks = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, config['vocab_size'])
    etab = jax.random.normal(jax.random.PRNGKey(2), (config['vocab_size'], D), dtype=dtype) * 0.02
    kernel_benchmarks['token_embed'] = (jax.jit(get_kernel('token_embed')), (toks, etab))

    if config['model_type'] == 'llama3':
        Hq, Hkv, Dh, FFN = config['num_heads'], config['num_kv_heads'], config['head_dim'], config['ffn_dim']
        q = jax.random.normal(jax.random.PRNGKey(3), (batch_size, seq_len, Hq, Dh), dtype=dtype)
        k = jax.random.normal(jax.random.PRNGKey(4), (batch_size, seq_len, Hkv, Dh), dtype=dtype)
        v = jax.random.normal(jax.random.PRNGKey(5), (batch_size, seq_len, Hkv, Dh), dtype=dtype)
        pos = jnp.arange(seq_len)
        kernel_benchmarks['rope'] = (jax.jit(lambda q, k: get_kernel('rope')(q, k, pos, theta=config['rope_theta'])), (q, k))
        kernel_benchmarks['gqa_attention'] = (jax.jit(get_kernel('gqa_attention')), (q, k, v))
        x_m = jax.random.normal(jax.random.PRNGKey(6), (batch_size, seq_len, D), dtype=dtype)
        wg = jax.random.normal(jax.random.PRNGKey(7), (D, FFN), dtype=dtype) * 0.02
        wu = jax.random.normal(jax.random.PRNGKey(8), (D, FFN), dtype=dtype) * 0.02
        wd = jax.random.normal(jax.random.PRNGKey(9), (FFN, D), dtype=dtype) * 0.02
        kernel_benchmarks['swiglu_mlp'] = (jax.jit(get_kernel('swiglu_mlp')), (x_m, wg, wu, wd))

    elif config['model_type'] == 'gla':
        H, Dh, FFN = config['num_heads'], config['head_dim'], config['ffn_dim']
        q = jax.random.normal(jax.random.PRNGKey(3), (batch_size, H, seq_len, Dh), dtype=dtype)
        k = jax.random.normal(jax.random.PRNGKey(4), (batch_size, H, seq_len, Dh), dtype=dtype)
        v = jax.random.normal(jax.random.PRNGKey(5), (batch_size, H, seq_len, Dh), dtype=dtype)
        gate = jax.random.normal(jax.random.PRNGKey(6), (batch_size, H, seq_len), dtype=jnp.float32)
        kernel_benchmarks['gated_linear_attention'] = (jax.jit(get_kernel('gated_linear_attention')), (q, k, v, gate))
        x_m = jax.random.normal(jax.random.PRNGKey(7), (batch_size, seq_len, D), dtype=dtype)
        wg = jax.random.normal(jax.random.PRNGKey(8), (D, FFN), dtype=dtype) * 0.02
        wu = jax.random.normal(jax.random.PRNGKey(9), (D, FFN), dtype=dtype) * 0.02
        wd = jax.random.normal(jax.random.PRNGKey(10), (FFN, D), dtype=dtype) * 0.02
        kernel_benchmarks['swiglu_mlp'] = (jax.jit(get_kernel('swiglu_mlp')), (x_m, wg, wu, wd))

    elif config['model_type'] == 'mamba2':
        H = config['num_heads']
        d_inner = config['d_model'] * config['expand']
        Dh = d_inner // H
        q = jax.random.normal(jax.random.PRNGKey(3), (batch_size, H, seq_len, Dh), dtype=dtype)
        k = jax.random.normal(jax.random.PRNGKey(4), (batch_size, H, seq_len, Dh), dtype=dtype)
        v = jax.random.normal(jax.random.PRNGKey(5), (batch_size, H, seq_len, Dh), dtype=dtype)
        A = jax.random.normal(jax.random.PRNGKey(6), (batch_size, H, seq_len), dtype=jnp.float32) * 0.5 - 4.0
        kernel_benchmarks['ssd_attention'] = (jax.jit(get_kernel('ssd_attention')), (q, k, v, A))

    results = {}
    for name, (fn, inputs) in kernel_benchmarks.items():
        for _ in range(num_warmup):
            out = fn(*inputs)
            (out[0] if isinstance(out, tuple) else out).block_until_ready()
        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            out = fn(*inputs)
            (out[0] if isinstance(out, tuple) else out).block_until_ready()
            times.append(time.perf_counter() - t0)
        times_ms = np.array(times) * 1000
        avg_ms = float(np.mean(times_ms))
        block_count = config['block_kernels'].get(name, 0)
        forward_count = config['forward_kernels'].get(name, 0)
        total_calls = block_count * config['eval_layers'] + forward_count
        results[name] = {
            'time_ms': round(avg_ms, 4),
            'std_ms': round(float(np.std(times_ms)), 4),
            'calls_per_forward': total_calls,
            'total_ms_per_forward': round(avg_ms * total_calls, 4),
        }
    return results

def benchmark_prefill(config, weights, batch_size=1, seq_len=2048, num_warmup=3, num_iters=20):
    prefill_fn = PREFILL_FNS[config['model_type']]
    tokens = jax.random.randint(jax.random.PRNGKey(42), (batch_size, seq_len), 0, config['vocab_size'])
    fn = jax.jit(lambda t: prefill_fn(weights, t, config))
    for _ in range(num_warmup):
        out = fn(tokens)
        jax.tree.map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(tokens)
        jax.tree.map(lambda x: x.block_until_ready(), out)
        times.append(time.perf_counter() - t0)
    times_ms = np.array(times) * 1000
    avg_ms = float(np.mean(times_ms))
    tps = (batch_size * seq_len) / (avg_ms / 1000)
    return {'time_ms': round(avg_ms, 4), 'std_ms': round(float(np.std(times_ms)), 4),
            'tokens_per_sec': round(tps, 1)}

def compute_predictions(kernel_profile, total_forward_ms):
    predictions = {}
    for name, kdata in kernel_profile.items():
        kt = kdata['total_ms_per_forward']
        frac = kt / total_forward_ms if total_forward_ms > 0 else 0
        speedups = {}
        for factor in [1.5, 2.0, 3.0, 5.0]:
            new_total = total_forward_ms - kt + kt / factor
            model_speedup = total_forward_ms / new_total if new_total > 0 else 1.0
            speedups[f'{factor}x'] = {
                'model_speedup': round(model_speedup, 4),
                'tokens_pct_increase': round((model_speedup - 1.0) * 100, 2),
            }
        predictions[name] = {'fraction': round(frac, 4), 'total_ms': kt, 'speedups': speedups}
    return predictions

def evaluate_model(model_name, seq_len=2048, optimized_kernels=None):
    config = ALL_MODELS[model_name].copy()
    init_fn = INIT_FNS[config['model_type']]

    print(f"\n{'=' * 70}")
    print(f"  {config['display_name']} Inference Benchmark (TPU v6e-1)")
    print(f"  Layers: {config['eval_layers']} (of {config['num_layers']}), Seq: {seq_len}")
    print(f"{'=' * 70}")

    reset_kernels()

    print("\n[1/4] Initializing weights...")
    weights = init_fn(config, jax.random.PRNGKey(0))
    jax.tree.map(lambda x: x.block_until_ready(), weights)
    print(f"  Allocated on {jax.devices()[0]}")

    print("\n[2/4] Profiling kernels...")
    kp = profile_kernels(config, seq_len=seq_len)
    theo_total = sum(k['total_ms_per_forward'] for k in kp.values())
    print(f"\n  {'Kernel':<30} {'Time/call':>10} {'Calls':>6} {'Total':>10} {'%':>7}")
    print(f"  {'-' * 65}")
    for name, kd in sorted(kp.items(), key=lambda x: x[1]['total_ms_per_forward'], reverse=True):
        frac = kd['total_ms_per_forward'] / theo_total * 100 if theo_total > 0 else 0
        print(f"  {name:<30} {kd['time_ms']:>8.4f}ms {kd['calls_per_forward']:>6} "
              f"{kd['total_ms_per_forward']:>8.4f}ms {frac:>5.1f}%")
    print(f"  {'Theoretical total':<30} {'':>10} {'':>6} {theo_total:>8.4f}ms")

    print("\n[3/4] Benchmarking prefill...")
    baseline = benchmark_prefill(config, weights, seq_len=seq_len)
    print(f"  {baseline['tokens_per_sec']:,.0f} tokens/s ({baseline['time_ms']:.2f}ms for {seq_len} tokens)")

    print("\n[4/4] Speedup predictions...")
    preds = compute_predictions(kp, baseline['time_ms'])
    print(f"\n  {'Kernel':<30} {'% of fwd':>8}  {'2x→tok/s':>10}  {'3x→tok/s':>10}  {'5x→tok/s':>10}")
    print(f"  {'-' * 72}")
    for name, p in sorted(preds.items(), key=lambda x: x[1]['fraction'], reverse=True):
        f = p['fraction'] * 100
        s2 = p['speedups']['2.0x']['tokens_pct_increase']
        s3 = p['speedups']['3.0x']['tokens_pct_increase']
        s5 = p['speedups']['5.0x']['tokens_pct_increase']
        print(f"  {name:<30} {f:>6.1f}%   {'+' if s2>=0 else ''}{s2:>8.1f}%  "
              f"{'+' if s3>=0 else ''}{s3:>8.1f}%   {'+' if s5>=0 else ''}{s5:>8.1f}%")

    result = {
        'model': model_name, 'config': {k: config[k] for k in ['eval_layers', 'num_layers', 'd_model']},
        'seq_len': seq_len, 'kernel_profile': kp, 'baseline': baseline, 'predictions': preds,
    }

    # Optimized kernel test
    if optimized_kernels:
        print(f"\n{'=' * 70}")
        print(f"  Optimized Kernel Comparison: {', '.join(optimized_kernels)}")
        print(f"{'=' * 70}")
        for kn in optimized_kernels:
            if kn in AVAILABLE_PALLAS:
                swap_kernel(kn, AVAILABLE_PALLAS[kn])
                print(f"  Swapped {kn} → Pallas")
            else:
                print(f"  WARNING: No Pallas kernel for '{kn}'")

        print("\n  Re-profiling kernels...")
        opt_kp = profile_kernels(config, seq_len=seq_len)
        for kn in optimized_kernels:
            if kn in opt_kp and kn in kp:
                old, new = kp[kn]['time_ms'], opt_kp[kn]['time_ms']
                print(f"  {kn}: {old:.4f}ms → {new:.4f}ms ({old/new:.2f}x)")

        print("\n  Re-benchmarking prefill...")
        opt_baseline = benchmark_prefill(config, weights, seq_len=seq_len)
        imp = (opt_baseline['tokens_per_sec'] / baseline['tokens_per_sec'] - 1) * 100
        print(f"  {baseline['tokens_per_sec']:,.0f} → {opt_baseline['tokens_per_sec']:,.0f} tokens/s "
              f"({'+' if imp>=0 else ''}{imp:.1f}%)")
        result['optimized'] = {'kernels': optimized_kernels, 'profile': opt_kp, 'baseline': opt_baseline}
        reset_kernels()

    print(f"\n{'=' * 70}\n  Done.\n{'=' * 70}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Inference speedup evaluation on TPU")
    parser.add_argument('--model', choices=list(ALL_MODELS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--optimized', nargs='+', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Specify --model or --all")

    models = list(ALL_MODELS.keys()) if args.all else [args.model]
    all_results = {}
    for m in models:
        all_results[m] = evaluate_model(m, seq_len=args.seq_len, optimized_kernels=args.optimized)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        # Print JSON summary to stdout
        print("\n" + json.dumps({m: {
            'tokens_per_sec': r['baseline']['tokens_per_sec'],
            'time_ms': r['baseline']['time_ms'],
        } for m, r in all_results.items()}, indent=2))

if __name__ == '__main__':
    main()
