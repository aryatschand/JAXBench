"""Mamba-2-2.7B — State Space Duality model in pure JAX.

Architecture: Mamba-2 SSD blocks + RMSNorm
Each block: in_proj -> conv1d -> SSD -> out_proj (gate path multiplied in).
No separate MLP — the Mamba block combines mixing and channel expansion.

Supports:
  - Prefill: parallel quadratic form
  - Decode: recurrent form with (D, D) SSM state per head per layer
"""

import jax
import jax.numpy as jnp
from inference_speedup.kernels import get_kernel


def init_weights(config, rng, num_layers=None):
    """Initialize random weights matching Mamba-2-2.7B architecture."""
    n_layers = num_layers or config['eval_layers']
    D = config['d_model']
    H = config['num_heads']
    expand = config['expand']
    d_inner = D * expand  # expanded inner dimension
    Dh = d_inner // H     # per-head dimension in expanded space
    V = config['vocab_size']
    d_conv = config['d_conv']

    def make(rng, shape, scale=0.02):
        return jax.random.normal(rng, shape, dtype=jnp.bfloat16) * scale

    keys = jax.random.split(rng, n_layers * 7 + 3)
    ki = iter(range(len(keys)))

    layers = []
    for _ in range(n_layers):
        layer = {
            'norm': jnp.ones(D, dtype=jnp.bfloat16),
            # in_proj: D -> 2*d_inner (x path + z gate path)
            'in_proj': make(keys[next(ki)], (D, 2 * d_inner)),
            # Conv1d weights for the x path
            'conv_weight': make(keys[next(ki)], (d_inner, d_conv)),
            'conv_bias': jnp.zeros(d_inner, dtype=jnp.bfloat16),
            # B, C projections: d_inner -> d_inner (reshaped to H heads)
            'w_b': make(keys[next(ki)], (d_inner, d_inner)),  # -> key (B matrix)
            'w_c': make(keys[next(ki)], (d_inner, d_inner)),  # -> query (C matrix)
            # A_log: learnable decay (per head)
            'A_log': jnp.full(H, -4.0, dtype=jnp.float32),
            # dt projection: d_inner -> H (one dt per head)
            'w_dt': make(keys[next(ki)], (d_inner, H)),
            'dt_bias': jnp.full(H, -2.0, dtype=jnp.float32),
            # out_proj: d_inner -> D
            'out_proj': make(keys[next(ki)], (d_inner, D)),
        }
        layers.append(layer)

    weights = {
        'embed': make(keys[next(ki)], (V, D)),
        'final_norm': jnp.ones(D, dtype=jnp.bfloat16),
        'lm_head': make(keys[next(ki)], (D, V)),
        'layers': layers,
    }
    return weights


def _causal_conv1d(x, weight, bias):
    """Simple causal 1D convolution (left-padded).

    Args:
        x: (B, S, D_inner)
        weight: (D_inner, K) — kernel of width K
        bias: (D_inner,)
    Returns:
        (B, S, D_inner)
    """
    B, S, D = x.shape
    K = weight.shape[1]
    # Left-pad
    x_padded = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0)))
    # Depthwise conv via scan over kernel positions
    out = jnp.zeros_like(x)
    for k in range(K):
        out = out + x_padded[:, k:k + S, :] * weight[:, k][None, None, :]
    return out + bias[None, None, :]


def prefill(weights, token_ids, config):
    """Full forward pass for prefill using parallel SSD form.

    Returns:
        logits: (B, S, V)
        states: list of (B, H, Dh, Dh) SSM states per layer
        conv_states: list of (B, d_conv-1, d_inner) conv buffer per layer
    """
    B, S = token_ids.shape
    H = config['num_heads']
    eps = config['rms_norm_eps']
    d_inner = config['d_model'] * config['expand']
    Dh = d_inner // H  # per-head dim in expanded space

    x = get_kernel('token_embed')(token_ids, weights['embed'])

    ssm_states = []
    conv_states = []
    for layer in weights['layers']:
        h = get_kernel('rmsnorm')(x, layer['norm'], eps=eps)

        # In projection: split into x_path and z_gate
        proj = h @ layer['in_proj']  # (B, S, 2*d_inner)
        x_path = proj[:, :, :d_inner]
        z_gate = proj[:, :, d_inner:]

        # Causal conv1d on x_path
        x_conv = _causal_conv1d(x_path, layer['conv_weight'], layer['conv_bias'])
        x_conv = jax.nn.silu(x_conv)

        # Save conv state (last d_conv-1 positions) for decode
        d_conv = config['d_conv']
        conv_state = x_path[:, -(d_conv - 1):, :]  # (B, d_conv-1, d_inner)
        conv_states.append(conv_state)

        # Project to B (key), C (query) for SSD — both (B, H, S, Dh)
        key = (x_conv @ layer['w_b']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        query = (x_conv @ layer['w_c']).reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
        # Value is x_conv itself, reshaped to (B, H, S, Dh)
        value = x_conv.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

        # Compute dt (discretization step) -> used as input-dependent A_log
        dt = x_conv @ layer['w_dt'] + layer['dt_bias']  # (B, S, H)
        # A_log_effective = A_log_base * softplus(dt)
        A_log_eff = layer['A_log'][None, None, :] * jax.nn.softplus(dt)  # (B, S, H)
        A_log_eff = A_log_eff.transpose(0, 2, 1)  # (B, H, S)

        # SSD attention (parallel form)
        ssd_out = get_kernel('ssd_attention')(query, key, value, A_log_eff)
        ssd_out = ssd_out.transpose(0, 2, 1, 3).reshape(B, S, d_inner)

        # Gate and output projection
        out = (ssd_out * jax.nn.silu(z_gate)) @ layer['out_proj']
        x = x + out

        # Compute SSM state from last few positions for decode
        a = jax.nn.sigmoid(A_log_eff.astype(jnp.float32))
        log_a = jnp.log(a + 1e-8)
        log_a_cumsum = jnp.cumsum(log_a, axis=-1)
        decay_to_end = jnp.exp(log_a_cumsum[:, :, -1:] - log_a_cumsum)
        k_weighted = key * decay_to_end[:, :, :, None]
        state = jnp.einsum('bhsd,bhse->bhde',
                           k_weighted.astype(jnp.float32),
                           value.astype(jnp.float32))
        ssm_states.append(state)

    x = get_kernel('rmsnorm')(x, weights['final_norm'], eps=eps)
    logits = x @ weights['lm_head']
    return logits, ssm_states, conv_states


def decode_step(weights, token_id, ssm_states, conv_states, config):
    """Single autoregressive decode step using recurrent SSD form.

    Args:
        weights: weight dict
        token_id: (B, 1) int32
        ssm_states: list of (B, H, Dh, Dh) SSM states
        conv_states: list of (B, d_conv-1, d_inner) conv buffers
        config: model config dict
    Returns:
        logits, new_ssm_states, new_conv_states
    """
    B = token_id.shape[0]
    H = config['num_heads']
    eps = config['rms_norm_eps']
    d_inner = config['d_model'] * config['expand']
    Dh = d_inner // H
    d_conv = config['d_conv']

    x = get_kernel('token_embed')(token_id, weights['embed'])

    new_ssm_states = []
    new_conv_states = []
    for i, layer in enumerate(weights['layers']):
        h = get_kernel('rmsnorm')(x, layer['norm'], eps=eps)

        proj = h @ layer['in_proj']
        x_path = proj[:, :, :d_inner]   # (B, 1, d_inner)
        z_gate = proj[:, :, d_inner:]

        # Update conv state: shift buffer and append new
        old_conv = conv_states[i]  # (B, d_conv-1, d_inner)
        new_conv = jnp.concatenate([old_conv[:, 1:, :], x_path], axis=1)
        new_conv_states.append(new_conv)

        # Apply conv: dot product of buffer+current with kernel weights
        conv_input = jnp.concatenate([new_conv, x_path], axis=1)  # (B, d_conv, d_inner)
        # Depthwise conv: sum over kernel dimension
        x_conv = jnp.sum(conv_input * layer['conv_weight'].T[None, :, :], axis=1, keepdims=True)
        x_conv = x_conv + layer['conv_bias'][None, None, :]
        x_conv = jax.nn.silu(x_conv)  # (B, 1, d_inner)

        # Project to B (key), C (query)
        key = (x_conv @ layer['w_b']).reshape(B, 1, H, Dh).transpose(0, 2, 1, 3)
        query = (x_conv @ layer['w_c']).reshape(B, 1, H, Dh).transpose(0, 2, 1, 3)
        value = x_conv.reshape(B, 1, H, Dh).transpose(0, 2, 1, 3)

        # dt -> A_log
        dt = x_conv @ layer['w_dt'] + layer['dt_bias']  # (B, 1, H)
        A_log_eff = layer['A_log'][None, None, :] * jax.nn.softplus(dt)
        A_log_eff = A_log_eff.transpose(0, 2, 1)  # (B, H, 1)

        # Recurrent SSD step
        ssd_out, new_state = get_kernel('ssd_decode')(
            query, key, value, A_log_eff, ssm_states[i]
        )
        ssd_out = ssd_out.transpose(0, 2, 1, 3).reshape(B, 1, d_inner)
        new_ssm_states.append(new_state)

        out = (ssd_out * jax.nn.silu(z_gate)) @ layer['out_proj']
        x = x + out

    x = get_kernel('rmsnorm')(x, weights['final_norm'], eps=eps)
    logits = x @ weights['lm_head']
    return logits, new_ssm_states, new_conv_states


def init_states(config, batch_size, num_layers=None):
    """Allocate zero SSM states and conv buffers for decode."""
    n_layers = num_layers or config['eval_layers']
    H = config['num_heads']
    d_inner = config['d_model'] * config['expand']
    Dh = d_inner // H
    d_conv = config['d_conv']

    ssm_states = [
        jnp.zeros((batch_size, H, Dh, Dh), dtype=jnp.float32)
        for _ in range(n_layers)
    ]
    conv_states = [
        jnp.zeros((batch_size, d_conv - 1, d_inner), dtype=jnp.bfloat16)
        for _ in range(n_layers)
    ]
    return ssm_states, conv_states
