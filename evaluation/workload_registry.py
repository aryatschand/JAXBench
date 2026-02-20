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
    category: str  # attention, mlp, rope, moe, norm
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
# Llama 3.1 — GQA Attention
# ============================================================================

def _llama3_gqa_inputs(config):
    """Generate inputs for Llama3 GQA attention."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = config['batch'], config['seq_len']
    Hq, Hkv, D = config['num_query_heads'], config['num_kv_heads'], config['head_dim']
    query = jax.random.normal(k1, (B, S, Hq, D), dtype=jnp.bfloat16)
    key_t = jax.random.normal(k2, (B, S, Hkv, D), dtype=jnp.bfloat16)
    value = jax.random.normal(k3, (B, S, Hkv, D), dtype=jnp.bfloat16)
    return (query, key_t, value)


def _llama3_gqa_baseline(query, key, value, num_kv_heads):
    """Llama3 GQA baseline implementation."""
    B, S, Hq, D = query.shape
    G = Hq // num_kv_heads
    key = jnp.repeat(key[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    value = jnp.repeat(value[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    return out.transpose(0, 2, 1, 3)


_gqa_configs = [
    ('llama3_8b_gqa', 'llama3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 32, 'num_kv_heads': 8, 'head_dim': 128}),
    ('llama3_70b_gqa', 'llama3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 64, 'num_kv_heads': 8, 'head_dim': 128}),
    ('llama3_405b_gqa', 'llama3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 128, 'num_kv_heads': 8, 'head_dim': 128}),
]

for _name, _model, _cfg in _gqa_configs:
    _c = _cfg.copy()
    _kv = _c['num_kv_heads']
    register_workload(WorkloadConfig(
        name=_name, model=_model, category='attention', config=_c,
        input_generator=lambda c=_c: _llama3_gqa_inputs(c),
        baseline_fn=lambda q, k, v, kv=_kv: _llama3_gqa_baseline(q, k, v, num_kv_heads=kv),
        rtol=1e-2, atol=1e-2,
    ))


# ============================================================================
# Llama 3.1 — SwiGLU MLP
# ============================================================================

def _llama3_swiglu_inputs(config):
    """Generate inputs for Llama3 SwiGLU MLP."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B, S, E, M = config['batch'], config['seq_len'], config['emb_dim'], config['mlp_dim']
    x = jax.random.normal(k1, (B, S, E), dtype=jnp.bfloat16)
    gate = jax.random.normal(k2, (E, M), dtype=jnp.bfloat16) * 0.02
    up = jax.random.normal(k3, (E, M), dtype=jnp.bfloat16) * 0.02
    down = jax.random.normal(k4, (M, E), dtype=jnp.bfloat16) * 0.02
    return (x, gate, up, down)


def _llama3_swiglu_baseline(x, gate_kernel, up_kernel, down_kernel):
    """Llama3 SwiGLU MLP baseline."""
    gate = jax.nn.silu(jnp.dot(x, gate_kernel))
    up = jnp.dot(x, up_kernel)
    return jnp.dot(gate * up, down_kernel)


_swiglu_configs = [
    ('llama3_8b_swiglu', 'llama3', {'batch': 1, 'seq_len': 2048, 'emb_dim': 4096, 'mlp_dim': 14336}),
    ('llama3_70b_swiglu', 'llama3', {'batch': 1, 'seq_len': 2048, 'emb_dim': 8192, 'mlp_dim': 28672}),
    ('llama3_405b_swiglu', 'llama3', {'batch': 1, 'seq_len': 2048, 'emb_dim': 16384, 'mlp_dim': 53248}),
]

for _name, _model, _cfg in _swiglu_configs:
    _c = _cfg.copy()
    register_workload(WorkloadConfig(
        name=_name, model=_model, category='mlp', config=_c,
        input_generator=lambda c=_c: _llama3_swiglu_inputs(c),
        baseline_fn=_llama3_swiglu_baseline,
        rtol=1e-2, atol=1e-2,
    ))


# ============================================================================
# Llama 3.1 — RoPE
# ============================================================================

def _llama3_rope_inputs(config):
    """Generate inputs for Llama3 RoPE."""
    key = jax.random.PRNGKey(42)
    B, S, H, D = config['batch'], config['seq_len'], config['num_heads'], config['head_dim']
    x = jax.random.normal(key, (B, S, H, D), dtype=jnp.bfloat16)
    return (x,)


def _llama3_rope_baseline(x, theta=500000.0):
    """Llama3 RoPE baseline."""
    B, S, H, D = x.shape
    dim_pairs = D // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, dim_pairs, dtype=jnp.float32) / dim_pairs))
    positions = jnp.arange(S, dtype=jnp.float32)
    angles = jnp.outer(positions, freqs)
    cos = jnp.cos(angles).astype(x.dtype)[None, :, None, :]
    sin = jnp.sin(angles).astype(x.dtype)[None, :, None, :]
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


_rope_configs = [
    ('llama3_8b_rope', 'llama3', {'batch': 1, 'seq_len': 2048, 'num_heads': 32, 'head_dim': 128}),
    ('llama3_70b_rope', 'llama3', {'batch': 1, 'seq_len': 2048, 'num_heads': 64, 'head_dim': 128}),
]

for _name, _model, _cfg in _rope_configs:
    _c = _cfg.copy()
    register_workload(WorkloadConfig(
        name=_name, model=_model, category='rope', config=_c,
        input_generator=lambda c=_c: _llama3_rope_inputs(c),
        baseline_fn=lambda x: _llama3_rope_baseline(x, theta=500000.0),
        rtol=1e-2, atol=1e-2,
    ))


# ============================================================================
# Llama 3.1 — RMSNorm
# ============================================================================

def _llama3_rmsnorm_inputs(config):
    """Generate inputs for Llama3 RMSNorm."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    B, S, E = config['batch'], config['seq_len'], config['emb_dim']
    x = jax.random.normal(k1, (B, S, E), dtype=jnp.bfloat16)
    scale = jax.random.normal(k2, (E,), dtype=jnp.bfloat16) * 0.1 + 1.0
    return (x, scale)


def _llama3_rmsnorm_baseline(x, scale, eps=1e-6):
    """Llama3 RMSNorm baseline."""
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * scale


_rmsnorm_configs = [
    ('llama3_8b_rmsnorm', 'llama3', {'batch': 1, 'seq_len': 2048, 'emb_dim': 4096}),
    ('llama3_70b_rmsnorm', 'llama3', {'batch': 1, 'seq_len': 2048, 'emb_dim': 8192}),
]

for _name, _model, _cfg in _rmsnorm_configs:
    _c = _cfg.copy()
    register_workload(WorkloadConfig(
        name=_name, model=_model, category='norm', config=_c,
        input_generator=lambda c=_c: _llama3_rmsnorm_inputs(c),
        baseline_fn=_llama3_rmsnorm_baseline,
        rtol=1e-2, atol=1e-2,
    ))


# ============================================================================
# Gemma 3 — Sliding Window Attention
# ============================================================================

def _gemma3_attn_inputs(config):
    """Generate inputs for Gemma3 attention."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = config['batch'], config['seq_len']
    Hq, Hkv, D = config['num_query_heads'], config['num_kv_heads'], config['head_dim']
    query = jax.random.normal(k1, (B, S, Hq, D), dtype=jnp.bfloat16)
    key_t = jax.random.normal(k2, (B, S, Hkv, D), dtype=jnp.bfloat16)
    value = jax.random.normal(k3, (B, S, Hkv, D), dtype=jnp.bfloat16)
    return (query, key_t, value)


def _gemma3_sliding_baseline(query, key, value, num_kv_heads, window_size=4096, soft_cap=50.0):
    """Gemma3 sliding window attention with QK norm and soft capping."""
    B, S, Hq, D = query.shape
    G = Hq // num_kv_heads
    query = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + 1e-6)
    key = key / (jnp.linalg.norm(key, axis=-1, keepdims=True) + 1e-6)
    key = jnp.repeat(key[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    value = jnp.repeat(value[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    attn = soft_cap * jnp.tanh(attn / soft_cap)
    pos = jnp.arange(S)
    dist = pos[:, None] - pos[None, :]
    mask = (dist >= 0) & (dist < window_size)
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    return out.transpose(0, 2, 1, 3)


def _gemma3_global_baseline(query, key, value, num_kv_heads, soft_cap=50.0):
    """Gemma3 global causal attention with QK norm and soft capping."""
    B, S, Hq, D = query.shape
    G = Hq // num_kv_heads
    query = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + 1e-6)
    key = key / (jnp.linalg.norm(key, axis=-1, keepdims=True) + 1e-6)
    key = jnp.repeat(key[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    value = jnp.repeat(value[:, :, :, None, :], G, axis=3).reshape(B, S, Hq, D)
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)
    scale = D ** -0.5
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    attn = soft_cap * jnp.tanh(attn / soft_cap)
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    return out.transpose(0, 2, 1, 3)


_gemma3_sw_configs = [
    ('gemma3_4b_sliding_window_attn', 'gemma3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 8, 'num_kv_heads': 4, 'head_dim': 256, 'window_size': 4096}),
    ('gemma3_12b_sliding_window_attn', 'gemma3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 16, 'num_kv_heads': 8, 'head_dim': 256, 'window_size': 4096}),
    ('gemma3_27b_sliding_window_attn', 'gemma3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 32, 'num_kv_heads': 16, 'head_dim': 144, 'window_size': 4096}),
]

for _name, _model, _cfg in _gemma3_sw_configs:
    _c = _cfg.copy()
    _kv = _c['num_kv_heads']
    register_workload(WorkloadConfig(
        name=_name, model=_model, category='attention', config=_c,
        input_generator=lambda c=_c: _gemma3_attn_inputs(c),
        baseline_fn=lambda q, k, v, kv=_kv: _gemma3_sliding_baseline(q, k, v, num_kv_heads=kv),
        rtol=1e-2, atol=1e-2,
    ))

_gemma3_global_configs = [
    ('gemma3_4b_global_attn', 'gemma3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 8, 'num_kv_heads': 4, 'head_dim': 256}),
    ('gemma3_12b_global_attn', 'gemma3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 16, 'num_kv_heads': 8, 'head_dim': 256}),
    ('gemma3_27b_global_attn', 'gemma3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 32, 'num_kv_heads': 16, 'head_dim': 144}),
]

for _name, _model, _cfg in _gemma3_global_configs:
    _c = _cfg.copy()
    _kv = _c['num_kv_heads']
    register_workload(WorkloadConfig(
        name=_name, model=_model, category='attention', config=_c,
        input_generator=lambda c=_c: _gemma3_attn_inputs(c),
        baseline_fn=lambda q, k, v, kv=_kv: _gemma3_global_baseline(q, k, v, num_kv_heads=kv),
        rtol=1e-2, atol=1e-2,
    ))


# ============================================================================
# Mixtral — Sparse MoE
# ============================================================================

def _mixtral_moe_inputs(config):
    """Generate inputs for Mixtral sparse MoE."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    B, S, E, M = config['batch'], config['seq_len'], config['emb_dim'], config['mlp_dim']
    N = config['num_experts']
    x = jax.random.normal(keys[0], (B, S, E), dtype=jnp.bfloat16)
    router = jax.random.normal(keys[1], (E, N), dtype=jnp.bfloat16) * 0.02
    gate_k = jax.random.normal(keys[2], (N, E, M), dtype=jnp.bfloat16) * 0.02
    up_k = jax.random.normal(keys[3], (N, E, M), dtype=jnp.bfloat16) * 0.02
    down_k = jax.random.normal(keys[4], (N, M, E), dtype=jnp.bfloat16) * 0.02
    return (x, router, gate_k, up_k, down_k)


def _mixtral_moe_baseline(x, router_weights, expert_gate, expert_up, expert_down, num_experts_per_tok):
    """Mixtral sparse MoE baseline (einsum version)."""
    B, S, E = x.shape
    N = router_weights.shape[-1]
    K = num_experts_per_tok
    logits = jnp.dot(x, router_weights)
    top_k_logits, top_k_idx = jax.lax.top_k(logits, K)
    probs = jax.nn.softmax(top_k_logits, axis=-1)
    gate_out = jax.nn.silu(jnp.einsum('bse,nem->bsnm', x, expert_gate))
    up_out = jnp.einsum('bse,nem->bsnm', x, expert_up)
    expert_outputs = jnp.einsum('bsnm,nme->bsne', gate_out * up_out, expert_down)
    one_hot = jax.nn.one_hot(top_k_idx, N)
    expert_weights = (one_hot * probs[..., None]).sum(axis=2)
    return jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights)


_mixtral_configs = [
    ('mixtral_8x7b_moe', 'mixtral', {'batch': 1, 'seq_len': 2048, 'emb_dim': 4096, 'mlp_dim': 14336, 'num_experts': 8, 'num_experts_per_tok': 2}),
    ('mixtral_8x22b_moe', 'mixtral', {'batch': 1, 'seq_len': 2048, 'emb_dim': 6144, 'mlp_dim': 16384, 'num_experts': 8, 'num_experts_per_tok': 2}),
]

for _name, _model, _cfg in _mixtral_configs:
    _c = _cfg.copy()
    _k = _c['num_experts_per_tok']
    register_workload(WorkloadConfig(
        name=_name, model=_model, category='moe', config=_c,
        input_generator=lambda c=_c: _mixtral_moe_inputs(c),
        baseline_fn=lambda x, rw, ge, ue, de, k=_k: _mixtral_moe_baseline(x, rw, ge, ue, de, num_experts_per_tok=k),
        rtol=1e-2, atol=1e-2,
    ))


# ============================================================================
# DeepSeek V3 — Multi-head Latent Attention (MLA)
# ============================================================================

def _deepseek_mla_inputs(config):
    """Generate inputs for DeepSeek V3 MLA."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 8)
    C = config
    B, S, E = C['batch'], C['seq_len'], C['emb_dim']
    H = C['num_heads']
    ql, kvl = C['q_lora_rank'], C['kv_lora_rank']
    nope, rope, vd = C['qk_nope_head_dim'], C['qk_rope_head_dim'], C['v_head_dim']
    x = jax.random.normal(keys[0], (B, S, E), dtype=jnp.bfloat16)
    q_down = jax.random.normal(keys[1], (E, ql), dtype=jnp.bfloat16) * 0.02
    q_up = jax.random.normal(keys[2], (ql, H * (nope + rope)), dtype=jnp.bfloat16) * 0.02
    kv_down = jax.random.normal(keys[3], (E, kvl + rope), dtype=jnp.bfloat16) * 0.02
    k_up = jax.random.normal(keys[4], (kvl, H * nope), dtype=jnp.bfloat16) * 0.02
    v_up = jax.random.normal(keys[5], (kvl, H * vd), dtype=jnp.bfloat16) * 0.02
    o_proj = jax.random.normal(keys[6], (H * vd, E), dtype=jnp.bfloat16) * 0.02
    return (x, q_down, q_up, kv_down, k_up, v_up, o_proj)


def _deepseek_mla_baseline(x, q_down, q_up, kv_down, k_up, v_up, o_proj,
                            num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                            kv_lora_rank, rope_theta):
    """DeepSeek V3 MLA baseline."""
    B, S, E = x.shape
    H, nope, rope_d, vd, kvl = num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank
    # Query
    q = jnp.dot(jnp.dot(x, q_down), q_up).reshape(B, S, H, nope + rope_d)
    q_nope, q_rope = q[..., :nope], q[..., nope:]
    # KV
    kv = jnp.dot(x, kv_down)
    k_latent, k_rope_raw = kv[..., :kvl], kv[..., kvl:]
    k_nope = jnp.dot(k_latent, k_up).reshape(B, S, H, nope)
    # RoPE
    freqs = 1.0 / (rope_theta ** (jnp.arange(0, rope_d, 2, dtype=jnp.float32) / rope_d))
    angles = jnp.outer(jnp.arange(S, dtype=jnp.float32), freqs)
    cos = jnp.cos(angles).astype(x.dtype)
    sin = jnp.sin(angles).astype(x.dtype)
    k_rope = jnp.broadcast_to(k_rope_raw[:, :, None, :], (B, S, H, rope_d))

    def _apply(t, c, s):
        t1, t2 = t[..., ::2], t[..., 1::2]
        c, s = c[None, :, None, :], s[None, :, None, :]
        return jnp.stack([t1 * c - t2 * s, t1 * s + t2 * c], axis=-1).reshape(t.shape)

    q_rope = _apply(q_rope, cos, sin)
    k_rope = _apply(k_rope, cos, sin)
    v = jnp.dot(k_latent, v_up).reshape(B, S, H, vd)
    # Attention
    q_full = jnp.concatenate([q_nope, q_rope], axis=-1).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    hd = nope + rope_d
    attn = jnp.einsum('bhqd,bhkd->bhqk', q_full, k_full) * (hd ** -0.5)
    mask = jnp.tril(jnp.ones((S, S)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * vd)
    return jnp.dot(out, o_proj)


_mla_config = {
    'batch': 1, 'seq_len': 2048, 'emb_dim': 7168, 'num_heads': 128,
    'q_lora_rank': 1536, 'kv_lora_rank': 512,
    'qk_nope_head_dim': 128, 'qk_rope_head_dim': 64, 'v_head_dim': 128,
    'rope_theta': 10000,
}
register_workload(WorkloadConfig(
    name='deepseek_v3_mla', model='deepseek_v3', category='attention', config=_mla_config,
    input_generator=lambda: _deepseek_mla_inputs(_mla_config),
    baseline_fn=lambda x, qd, qu, kvd, ku, vu, op: _deepseek_mla_baseline(
        x, qd, qu, kvd, ku, vu, op,
        num_heads=128, qk_nope_head_dim=128, qk_rope_head_dim=64,
        v_head_dim=128, kv_lora_rank=512, rope_theta=10000),
    rtol=1e-2, atol=1e-2,
))


# ============================================================================
# DeepSeek V3 — YaRN RoPE
# ============================================================================

def _deepseek_yarn_inputs(config):
    """Generate inputs for DeepSeek V3 YaRN RoPE."""
    key = jax.random.PRNGKey(42)
    B, S, H, D = config['batch'], config['seq_len'], config['num_heads'], config['head_dim']
    x = jax.random.normal(key, (B, S, H, D), dtype=jnp.bfloat16)
    return (x,)


def _deepseek_yarn_baseline(x, rope_theta, max_pos, orig_max_pos, rope_factor, beta_fast, mscale):
    """DeepSeek V3 YaRN RoPE baseline."""
    import math
    B, S, H, D = x.shape
    dim_pairs = D // 2
    base_freqs = 1.0 / (rope_theta ** (jnp.arange(0, dim_pairs, dtype=jnp.float32) / dim_pairs))
    wavelengths = 2 * math.pi / base_freqs
    low_wl = orig_max_pos / beta_fast
    high_wl = float(orig_max_pos)
    smooth = jnp.clip((wavelengths - high_wl) / (low_wl - high_wl), 0.0, 1.0)
    yarn_freqs = base_freqs * (rope_factor ** (1 - smooth)) / rope_factor
    angles = jnp.outer(jnp.arange(S, dtype=jnp.float32), yarn_freqs)
    cos = jnp.cos(angles).astype(x.dtype)[None, :, None, :]
    sin = jnp.sin(angles).astype(x.dtype)[None, :, None, :]
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = jnp.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], axis=-1).reshape(B, S, H, D)
    return out * mscale


_yarn_config = {
    'batch': 1, 'seq_len': 8192, 'num_heads': 128, 'head_dim': 64,
    'rope_theta': 10000, 'max_position_embeddings': 163840,
    'original_max_position_embeddings': 4096, 'rope_factor': 40,
    'beta_fast': 32, 'mscale': 1.0,
}
register_workload(WorkloadConfig(
    name='deepseek_v3_yarn_rope', model='deepseek_v3', category='rope', config=_yarn_config,
    input_generator=lambda: _deepseek_yarn_inputs(_yarn_config),
    baseline_fn=lambda x: _deepseek_yarn_baseline(x, 10000, 163840, 4096, 40, 32, 1.0),
    rtol=1e-2, atol=1e-2,
))


# ============================================================================
# DeepSeek V3 — MoE with Shared Experts
# ============================================================================

def _deepseek_moe_inputs(config):
    """Generate inputs for DeepSeek V3 MoE."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)
    C = config
    B, S, E, M = C['batch'], C['seq_len'], C['emb_dim'], C['mlp_dim']
    SM, N = C['shared_mlp_dim'], C['num_experts']
    x = jax.random.normal(keys[0], (B, S, E), dtype=jnp.bfloat16)
    rw = jax.random.normal(keys[1], (E, N), dtype=jnp.bfloat16) * 0.02
    rb = jnp.zeros((N,), dtype=jnp.bfloat16)
    sg = jax.random.normal(keys[2], (E, SM), dtype=jnp.bfloat16) * 0.02
    su = jax.random.normal(keys[3], (E, SM), dtype=jnp.bfloat16) * 0.02
    sd = jax.random.normal(keys[4], (SM, E), dtype=jnp.bfloat16) * 0.02
    eg = jax.random.normal(keys[5], (N, E, M), dtype=jnp.bfloat16) * 0.02
    eu = jax.random.normal(keys[6], (N, E, M), dtype=jnp.bfloat16) * 0.02
    ed = jax.random.normal(keys[7], (N, M, E), dtype=jnp.bfloat16) * 0.02
    return (x, rw, rb, sg, su, sd, eg, eu, ed)


def _deepseek_moe_baseline(x, rw, rb, sg, su, sd, eg, eu, ed, num_experts_per_tok, scaling_factor):
    """DeepSeek V3 MoE baseline."""
    B, S, E = x.shape
    N = rw.shape[-1]
    K = num_experts_per_tok
    sf = scaling_factor
    s_out = jnp.dot(jax.nn.silu(jnp.dot(x, sg)) * jnp.dot(x, su), sd)
    scores = jax.nn.sigmoid(jnp.dot(x, rw)) + rb
    top_k_scores, top_k_idx = jax.lax.top_k(scores, K)
    probs = top_k_scores / (top_k_scores.sum(axis=-1, keepdims=True) + 1e-6)
    gate_out = jax.nn.silu(jnp.einsum('bse,nem->bsnm', x, eg))
    up_out = jnp.einsum('bse,nem->bsnm', x, eu)
    expert_outputs = jnp.einsum('bsnm,nme->bsne', gate_out * up_out, ed)
    one_hot = jax.nn.one_hot(top_k_idx, N)
    expert_weights = (one_hot * probs[..., None]).sum(axis=2)
    r_out = jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights) * sf
    return s_out + r_out


_ds_moe_config = {
    'batch': 1, 'seq_len': 2048, 'emb_dim': 7168, 'mlp_dim': 2048,
    'num_experts': 256, 'num_experts_per_tok': 8,
    'shared_experts': 1, 'shared_mlp_dim': 18432, 'routed_scaling_factor': 2.5,
}
register_workload(WorkloadConfig(
    name='deepseek_v3_moe', model='deepseek_v3', category='moe', config=_ds_moe_config,
    input_generator=lambda: _deepseek_moe_inputs(_ds_moe_config),
    baseline_fn=lambda x, rw, rb, sg, su, sd, eg, eu, ed: _deepseek_moe_baseline(
        x, rw, rb, sg, su, sd, eg, eu, ed, num_experts_per_tok=8, scaling_factor=2.5),
    rtol=1e-2, atol=1e-2,
))


# ============================================================================
# Llama 3.1 — Token Embedding
# ============================================================================

def _llama3_token_embed_inputs(config):
    """Generate inputs for token embedding lookup."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    B, S, V, E = config['batch'], config['seq_len'], config['vocab_size'], config['emb_dim']
    token_ids = jax.random.randint(k1, (B, S), 0, V)
    embed_table = jax.random.normal(k2, (V, E), dtype=jnp.bfloat16) * 0.02
    return (token_ids, embed_table)


def _llama3_token_embed_baseline(token_ids, embed_table, emb_dim=4096):
    """Token embedding: lookup + scale."""
    return embed_table[token_ids] * (emb_dim ** 0.5)


_token_embed_config = {
    'batch': 1, 'seq_len': 2048, 'vocab_size': 128256, 'emb_dim': 4096,
}
register_workload(WorkloadConfig(
    name='llama3_8b_token_embed', model='llama3', category='embed', config=_token_embed_config,
    input_generator=lambda: _llama3_token_embed_inputs(_token_embed_config),
    baseline_fn=lambda ids, tbl: _llama3_token_embed_baseline(ids, tbl, emb_dim=4096),
    rtol=1e-2, atol=1e-2,
))


# ============================================================================
# Qwen 3 — GQA Attention
# ============================================================================

def _qwen3_gqa_inputs(config):
    """Generate inputs for Qwen3 GQA attention."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, S = config['batch'], config['seq_len']
    Hq, Hkv, D = config['num_query_heads'], config['num_kv_heads'], config['head_dim']
    query = jax.random.normal(k1, (B, S, Hq, D), dtype=jnp.bfloat16)
    key_t = jax.random.normal(k2, (B, S, Hkv, D), dtype=jnp.bfloat16)
    value = jax.random.normal(k3, (B, S, Hkv, D), dtype=jnp.bfloat16)
    return (query, key_t, value)


_qwen3_gqa_configs = [
    ('qwen3_8b_gqa', 'qwen3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 32, 'num_kv_heads': 8, 'head_dim': 128, 'emb_dim': 4096}),
    ('qwen3_14b_gqa', 'qwen3', {'batch': 1, 'seq_len': 2048, 'num_query_heads': 40, 'num_kv_heads': 8, 'head_dim': 128, 'emb_dim': 5120}),
]

for _name, _model, _cfg in _qwen3_gqa_configs:
    _c = _cfg.copy()
    _kv = _c['num_kv_heads']
    register_workload(WorkloadConfig(
        name=_name, model=_model, category='attention', config=_c,
        input_generator=lambda c=_c: _qwen3_gqa_inputs(c),
        baseline_fn=lambda q, k, v, kv=_kv: _llama3_gqa_baseline(q, k, v, num_kv_heads=kv),
        rtol=1e-2, atol=1e-2,
    ))


# ============================================================================
# Qwen 3 — SwiGLU MLP
# ============================================================================

_qwen3_swiglu_config = {
    'batch': 1, 'seq_len': 2048, 'emb_dim': 4096, 'mlp_dim': 14336,
}
register_workload(WorkloadConfig(
    name='qwen3_8b_swiglu', model='qwen3', category='mlp', config=_qwen3_swiglu_config,
    input_generator=lambda: _llama3_swiglu_inputs(_qwen3_swiglu_config),
    baseline_fn=_llama3_swiglu_baseline,
    rtol=1e-2, atol=1e-2,
))


# ============================================================================
# Qwen 3 MoE-30B — Sparse MoE with Shared Expert
# ============================================================================

def _qwen3_moe_inputs(config):
    """Generate inputs for Qwen3 MoE."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)
    C = config
    B, S, E, M = C['batch'], C['seq_len'], C['emb_dim'], C['mlp_dim']
    SM, N = C['shared_mlp_dim'], C['num_experts']
    x = jax.random.normal(keys[0], (B, S, E), dtype=jnp.bfloat16)
    rw = jax.random.normal(keys[1], (E, N), dtype=jnp.bfloat16) * 0.02
    sg = jax.random.normal(keys[2], (E, SM), dtype=jnp.bfloat16) * 0.02
    su = jax.random.normal(keys[3], (E, SM), dtype=jnp.bfloat16) * 0.02
    sd = jax.random.normal(keys[4], (SM, E), dtype=jnp.bfloat16) * 0.02
    eg = jax.random.normal(keys[5], (N, E, M), dtype=jnp.bfloat16) * 0.02
    eu = jax.random.normal(keys[6], (N, E, M), dtype=jnp.bfloat16) * 0.02
    ed = jax.random.normal(keys[7], (N, M, E), dtype=jnp.bfloat16) * 0.02
    return (x, rw, sg, su, sd, eg, eu, ed)


def _qwen3_moe_baseline(x, rw, sg, su, sd, eg, eu, ed, num_experts_per_tok):
    """Qwen3 MoE with shared expert baseline."""
    B, S, E = x.shape
    N = rw.shape[-1]
    K = num_experts_per_tok
    s_out = jnp.dot(jax.nn.silu(jnp.dot(x, sg)) * jnp.dot(x, su), sd)
    scores = jax.nn.sigmoid(jnp.dot(x, rw))
    top_k_scores, top_k_idx = jax.lax.top_k(scores, K)
    probs = top_k_scores / (top_k_scores.sum(axis=-1, keepdims=True) + 1e-6)
    gate_out = jax.nn.silu(jnp.einsum('bse,nem->bsnm', x, eg))
    up_out = jnp.einsum('bse,nem->bsnm', x, eu)
    expert_outputs = jnp.einsum('bsnm,nme->bsne', gate_out * up_out, ed)
    one_hot = jax.nn.one_hot(top_k_idx, N)
    expert_weights = (one_hot * probs[..., None]).sum(axis=2)
    r_out = jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights)
    return s_out + r_out


_qwen3_moe_config = {
    'batch': 1, 'seq_len': 2048, 'emb_dim': 2048, 'mlp_dim': 1024,
    'num_experts': 128, 'num_experts_per_tok': 8, 'shared_mlp_dim': 4096,
}
register_workload(WorkloadConfig(
    name='qwen3_moe_30b_moe', model='qwen3', category='moe', config=_qwen3_moe_config,
    input_generator=lambda: _qwen3_moe_inputs(_qwen3_moe_config),
    baseline_fn=lambda x, rw, sg, su, sd, eg, eu, ed: _qwen3_moe_baseline(
        x, rw, sg, su, sd, eg, eu, ed, num_experts_per_tok=8),
    rtol=1e-2, atol=1e-2,
))


# ============================================================================
# Llama 4 Scout/Maverick — GQA Attention
# ============================================================================

_llama4_scout_gqa_config = {
    'batch': 1, 'seq_len': 2048, 'num_query_heads': 48, 'num_kv_heads': 8,
    'head_dim': 128, 'emb_dim': 5120,
}
register_workload(WorkloadConfig(
    name='llama4_scout_gqa', model='llama4', category='attention', config=_llama4_scout_gqa_config,
    input_generator=lambda: _llama3_gqa_inputs(_llama4_scout_gqa_config),
    baseline_fn=lambda q, k, v: _llama3_gqa_baseline(q, k, v, num_kv_heads=8),
    rtol=1e-2, atol=1e-2,
))


# ============================================================================
# Llama 4 Scout — Sparse MoE (top-1)
# ============================================================================

def _llama4_moe_inputs(config):
    """Generate inputs for Llama 4 MoE."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)
    C = config
    B, S, E, M = C['batch'], C['seq_len'], C['emb_dim'], C['mlp_dim']
    N = C['num_experts']
    x = jax.random.normal(keys[0], (B, S, E), dtype=jnp.bfloat16)
    rw = jax.random.normal(keys[1], (E, N), dtype=jnp.bfloat16) * 0.02
    eg = jax.random.normal(keys[2], (N, E, M), dtype=jnp.bfloat16) * 0.02
    eu = jax.random.normal(keys[3], (N, E, M), dtype=jnp.bfloat16) * 0.02
    ed = jax.random.normal(keys[4], (N, M, E), dtype=jnp.bfloat16) * 0.02
    return (x, rw, eg, eu, ed)


def _llama4_moe_baseline(x, rw, eg, eu, ed, num_experts_per_tok):
    """Llama 4 MoE with top-1 routing."""
    B, S, E = x.shape
    N = rw.shape[-1]
    K = num_experts_per_tok
    logits = jnp.dot(x, rw)
    top_k_logits, top_k_idx = jax.lax.top_k(logits, K)
    probs = jax.nn.softmax(top_k_logits, axis=-1)
    gate_out = jax.nn.silu(jnp.einsum('bse,nem->bsnm', x, eg))
    up_out = jnp.einsum('bse,nem->bsnm', x, eu)
    expert_outputs = jnp.einsum('bsnm,nme->bsne', gate_out * up_out, ed)
    one_hot = jax.nn.one_hot(top_k_idx, N)
    expert_weights = (one_hot * probs[..., None]).sum(axis=2)
    return jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights)


_llama4_moe_configs = [
    ('llama4_scout_moe', 'llama4', {'batch': 1, 'seq_len': 2048, 'emb_dim': 5120, 'mlp_dim': 8192, 'num_experts': 16, 'num_experts_per_tok': 1}),
    ('llama4_maverick_moe', 'llama4', {'batch': 1, 'seq_len': 2048, 'emb_dim': 5120, 'mlp_dim': 4096, 'num_experts': 128, 'num_experts_per_tok': 1}),
]

for _name, _model, _cfg in _llama4_moe_configs:
    _c = _cfg.copy()
    _k = _c['num_experts_per_tok']
    register_workload(WorkloadConfig(
        name=_name, model=_model, category='moe', config=_c,
        input_generator=lambda c=_c: _llama4_moe_inputs(c),
        baseline_fn=lambda x, rw, eg, eu, ed, k=_k: _llama4_moe_baseline(x, rw, eg, eu, ed, num_experts_per_tok=k),
        rtol=1e-2, atol=1e-2,
    ))


# ============================================================================
# Llama 4 Scout — RoPE
# ============================================================================

_llama4_rope_config = {
    'batch': 1, 'seq_len': 2048, 'num_heads': 48, 'head_dim': 128,
}
register_workload(WorkloadConfig(
    name='llama4_scout_rope', model='llama4', category='rope', config=_llama4_rope_config,
    input_generator=lambda: _llama3_rope_inputs(_llama4_rope_config),
    baseline_fn=lambda x: _llama3_rope_baseline(x, theta=500000.0),
    rtol=1e-2, atol=1e-2,
))
