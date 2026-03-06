"""GLA-1.3B — Gated Linear Attention transformer in pure JAX.

Architecture: Gated Linear Attention + SwiGLU MLP + RMSNorm
No RoPE — GLA uses data-dependent forget gates instead of positional encoding.

Supports:
  - Prefill: parallel quadratic form
  - Decode: recurrent form with (D, D) state per head per layer
"""

import jax
import jax.numpy as jnp
from inference_speedup.kernels import get_kernel


def init_weights(config, rng, num_layers=None):
    """Initialize random weights matching GLA-1.3B architecture."""
    n_layers = num_layers or config['eval_layers']
    D = config['d_model']
    H = config['num_heads']
    Dh = config['head_dim']
    FFN = config['ffn_dim']
    V = config['vocab_size']

    def make(rng, shape, scale=0.02):
        return jax.random.normal(rng, shape, dtype=jnp.bfloat16) * scale

    keys = jax.random.split(rng, n_layers * 8 + 3)
    ki = iter(range(len(keys)))

    layers = []
    for _ in range(n_layers):
        layer = {
            'attn_norm': jnp.ones(D, dtype=jnp.bfloat16),
            'wq': make(keys[next(ki)], (D, H * Dh)),
            'wk': make(keys[next(ki)], (D, H * Dh)),
            'wv': make(keys[next(ki)], (D, H * Dh)),
            'wo': make(keys[next(ki)], (H * Dh, D)),
            'w_gate': make(keys[next(ki)], (D, H)),     # gate logits per head
            'ffn_norm': jnp.ones(D, dtype=jnp.bfloat16),
            'w_ffn_gate': make(keys[next(ki)], (D, FFN)),
            'w_ffn_up': make(keys[next(ki)], (D, FFN)),
            'w_ffn_down': make(keys[next(ki)], (FFN, D)),
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
    """Full forward pass for prefill using parallel GLA form.

    Returns:
        logits: (B, S, V)
        states: list of (B, H, D, D) recurrent states per layer (for decode)
    """
    B, S = token_ids.shape
    H = config['num_heads']
    Dh = config['head_dim']
    eps = config['rms_norm_eps']

    x = get_kernel('token_embed')(token_ids, weights['embed'])

    states = []
    for layer in weights['layers']:
        h = get_kernel('rmsnorm')(x, layer['attn_norm'], eps=eps)

        # QKV projections -> (B, H, S, Dh)
        q = (h @ layer['wq']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        k = (h @ layer['wk']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        v = (h @ layer['wv']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

        # Gate logits: (B, S, H) -> (B, H, S)
        gate_logits = (h @ layer['w_gate']).transpose(0, 2, 1)

        # Gated linear attention (parallel form)
        attn_out = get_kernel('gated_linear_attention')(q, k, v, gate_logits)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, H * Dh)

        x = x + attn_out @ layer['wo']

        # Compute recurrent state for decode: S = Σ_t Π_{j>t} g_j * k_t^T v_t
        # For simplicity, just compute the final state from the last position
        gate = jax.nn.sigmoid(gate_logits)  # (B, H, S)
        log_gate = jnp.log(gate + 1e-8)
        log_gate_cumsum = jnp.cumsum(log_gate, axis=-1)  # (B, H, S)
        # Decay weights from each position to end: exp(cumsum[-1] - cumsum[t])
        decay_to_end = jnp.exp(log_gate_cumsum[:, :, -1:] - log_gate_cumsum)
        # Weighted KV: Σ_t decay[t] * k_t^T v_t
        k_weighted = k * decay_to_end[:, :, :, None]  # (B, H, S, D)
        state = jnp.einsum('bhsd,bhse->bhde',
                           k_weighted.astype(jnp.float32),
                           v.astype(jnp.float32))
        states.append(state)

        # MLP
        h = get_kernel('rmsnorm')(x, layer['ffn_norm'], eps=eps)
        x = x + get_kernel('swiglu_mlp')(
            h, layer['w_ffn_gate'], layer['w_ffn_up'], layer['w_ffn_down']
        )

    x = get_kernel('rmsnorm')(x, weights['final_norm'], eps=eps)
    logits = x @ weights['lm_head']
    return logits, states


def decode_step(weights, token_id, states, config):
    """Single autoregressive decode step using recurrent form.

    Args:
        weights: weight dict
        token_id: (B, 1) int32
        states: list of (B, H, Dh, Dh) recurrent states
        config: model config dict
    Returns:
        logits: (B, 1, V), updated_states
    """
    B = token_id.shape[0]
    H = config['num_heads']
    Dh = config['head_dim']
    eps = config['rms_norm_eps']

    x = get_kernel('token_embed')(token_id, weights['embed'])

    new_states = []
    for i, layer in enumerate(weights['layers']):
        h = get_kernel('rmsnorm')(x, layer['attn_norm'], eps=eps)

        q = (h @ layer['wq']).reshape(B, 1, H, Dh).transpose(0, 2, 1, 3)
        k = (h @ layer['wk']).reshape(B, 1, H, Dh).transpose(0, 2, 1, 3)
        v = (h @ layer['wv']).reshape(B, 1, H, Dh).transpose(0, 2, 1, 3)
        gate_logits = (h @ layer['w_gate']).transpose(0, 2, 1)  # (B, H, 1)

        attn_out, new_state = get_kernel('gla_decode')(
            q, k, v, gate_logits, states[i]
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, 1, H * Dh)
        x = x + attn_out @ layer['wo']
        new_states.append(new_state)

        h = get_kernel('rmsnorm')(x, layer['ffn_norm'], eps=eps)
        x = x + get_kernel('swiglu_mlp')(
            h, layer['w_ffn_gate'], layer['w_ffn_up'], layer['w_ffn_down']
        )

    x = get_kernel('rmsnorm')(x, weights['final_norm'], eps=eps)
    logits = x @ weights['lm_head']
    return logits, new_states


def init_states(config, batch_size, num_layers=None):
    """Allocate zero recurrent states for decode."""
    n_layers = num_layers or config['eval_layers']
    H = config['num_heads']
    Dh = config['head_dim']
    return [
        jnp.zeros((batch_size, H, Dh, Dh), dtype=jnp.float32)
        for _ in range(n_layers)
    ]
