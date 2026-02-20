"""YaRN RoPE (context extension) — DeepSeek V3 671B. Extracted from MaxText."""
import jax
import jax.numpy as jnp
from functools import partial
import math

CONFIG = {
    'name': 'deepseek_v3_yarn_rope',
    'model': 'DeepSeek-V3-671B',
    'operator': 'yarn_rope',
    'batch': 1,
    'seq_len': 8192,
    'num_heads': 128,
    'head_dim': 64,
    'rope_theta': 10000,
    'max_position_embeddings': 163840,
    'original_max_position_embeddings': 4096,
    'rope_factor': 40,
    'beta_fast': 32,
    'mscale': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (x,) tensor."""
    key = jax.random.PRNGKey(42)
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    x = jax.random.normal(key, (B, S, H, D), dtype=dtype)
    return (x,)


def workload(x):
    """YaRN RoPE: frequency scaling for context extension, interleaved format."""
    C = CONFIG
    B, S, H, D = x.shape
    dim_pairs = D // 2
    theta = C['rope_theta']
    base_freqs = 1.0 / (theta ** (jnp.arange(0, dim_pairs, dtype=jnp.float32) / dim_pairs))
    # YaRN frequency scaling
    wavelengths = 2 * math.pi / base_freqs
    low_freq_wavelen = C['original_max_position_embeddings'] / C['beta_fast']
    high_freq_wavelen = float(C['original_max_position_embeddings'])
    smooth = jnp.clip(
        (wavelengths - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen),
        0.0, 1.0,
    )
    factor = C['rope_factor']
    yarn_freqs = base_freqs * (factor ** (1 - smooth)) / factor
    # Compute angles
    pos = jnp.arange(S, dtype=jnp.float32)
    angles = jnp.outer(pos, yarn_freqs)
    cos = jnp.cos(angles).astype(x.dtype)[None, :, None, :]
    sin = jnp.sin(angles).astype(x.dtype)[None, :, None, :]
    # Interleaved format
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    out = jnp.stack([rot_even, rot_odd], axis=-1).reshape(B, S, H, D)
    return out * C['mscale']


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
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    total_bytes = B * S * H * D * 2 * 2
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(total_bytes / (avg / 1000) / 1e12, 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
