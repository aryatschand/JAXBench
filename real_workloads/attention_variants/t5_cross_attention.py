"""Cross-Attention — T5-Base encoder-decoder (google/t5-v1_1-base).

Encoder-decoder cross-attention where Q comes from the decoder and K, V come
from the encoder output. No relative position bias (only in self-attention).

Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_flax_t5.py
Paper: "Exploring the Limits of Transfer Learning with T5" (Raffel et al., 2020)
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 't5_base_cross_attention',
    'model': 'T5-Base',
    'operator': 'cross_attention',
    'batch': 1,
    'encoder_seq_len': 2048,
    'decoder_seq_len': 512,
    'num_heads': 12,
    'head_dim': 64,
    'd_model': 768,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (decoder_query, encoder_key, encoder_value)."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    S_enc, S_dec = CONFIG['encoder_seq_len'], CONFIG['decoder_seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    dec_query = jax.random.normal(k1, (B, S_dec, H, D), dtype=dtype)
    enc_key = jax.random.normal(k2, (B, S_enc, H, D), dtype=dtype)
    enc_value = jax.random.normal(k3, (B, S_enc, H, D), dtype=dtype)
    return dec_query, enc_key, enc_value


def workload(dec_query, enc_key, enc_value):
    """T5 cross-attention: decoder queries attend to encoder key-values.

    No causal mask (all encoder positions visible).
    No relative position bias (only used in self-attention layers).
    T5 uses unscaled dot products (historical design choice).
    """
    B, S_dec, H, D = dec_query.shape
    S_enc = enc_key.shape[1]
    q = dec_query.transpose(0, 2, 1, 3)   # (B, H, S_dec, D)
    k = enc_key.transpose(0, 2, 1, 3)     # (B, H, S_enc, D)
    v = enc_value.transpose(0, 2, 1, 3)   # (B, H, S_enc, D)
    # T5 uses unscaled attention (no 1/sqrt(d_k))
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k)  # (B, H, S_dec, S_enc)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(dec_query.dtype)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)  # (B, H, S_dec, D)
    return out.transpose(0, 2, 1, 3)


def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    B = CONFIG['batch']
    S_dec, S_enc = CONFIG['decoder_seq_len'], CONFIG['encoder_seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    # QK^T: B*H*S_dec*S_enc*D * 2 (matmul)
    # AV:   B*H*S_dec*S_enc*D * 2 (matmul)
    flops = B * H * S_dec * S_enc * D * 4
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 2),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
