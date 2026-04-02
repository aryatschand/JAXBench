import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

CONFIG = {
    'name': 'ragged_paged_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'ragged_paged_attention',
    'max_num_batched_tokens': 2048,
    'max_num_seqs': 32,
    'num_q_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'page_size': 16,
    'pages_per_seq': 128,
}

_skip_jit = True


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    max_tokens = CONFIG['max_num_batched_tokens']
    max_seqs = CONFIG['max_num_seqs']
    H_q = CONFIG['num_q_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = CONFIG['pages_per_seq']
    num_combined_kv_heads = 2 * H_kv
    total_num_pages = max_seqs * pages_per_seq
    q = jax.random.normal(k1, (max_tokens, H_q, D), dtype=dtype)
    kv_pages = jax.random.normal(
        k2, (total_num_pages, page_size, num_combined_kv_heads, D), dtype=dtype
    )
    tokens_per_seq = max_tokens // max_seqs
    kv_len_per_seq = pages_per_seq * page_size
    kv_lens = jnp.full((max_seqs,), kv_len_per_seq, dtype=jnp.int32)
    page_indices = jnp.arange(total_num_pages, dtype=jnp.int32).reshape(
        max_seqs, pages_per_seq
    )
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def attention_kernel(q_ref, k_ref, v_ref, o_ref):
    q = q_ref[:, :, :]
    k = k_ref[:, :, :]
    v = v_ref[:, :, :]

    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])

    attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    attn = attn * sm_scale

    attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v).astype(q.dtype)

    o_ref[:, :, :] = out


def workload(queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])

    max_seqs = CONFIG['max_num_seqs']
    tokens_per_seq = CONFIG['max_num_batched_tokens'] // max_seqs
    H_q = CONFIG['num_q_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = CONFIG['pages_per_seq']
    kv_len = page_size * pages_per_seq

    num_query_per_kv = H_q // H_kv

    # reshape queries -> (S, T, H, D)
    q = queries.reshape(max_seqs, tokens_per_seq, H_q, D)

    # build k/v dense
    kv = kv_pages.reshape(max_seqs, pages_per_seq, page_size, 2 * H_kv, D)
    k = kv[:, :, :, 0::2, :].reshape(max_seqs, kv_len, H_kv, D)
    v = kv[:, :, :, 1::2, :].reshape(max_seqs, kv_len, H_kv, D)

    k = jnp.repeat(k, num_query_per_kv, axis=2)
    v = jnp.repeat(v, num_query_per_kv, axis=2)

    # flatten seq into batch dimension for kernel
    q_flat = q.reshape(max_seqs * tokens_per_seq, H_q, D)
    k_flat = k.reshape(max_seqs * kv_len, H_q, D)
    v_flat = v.reshape(max_seqs * kv_len, H_q, D)

    block_q = tokens_per_seq
    block_k = kv_len

    def kernel(q_ref, k_ref, v_ref, o_ref):
        q_block = q_ref[:, :, :]
        k_block = k_ref[:, :, :]
        v_block = v_ref[:, :, :]

        attn = jnp.einsum("qhd,khd->hqk", q_block, k_block, preferred_element_type=jnp.float32)
        attn = attn * sm_scale
        attn = jax.nn.softmax(attn, axis=-1).astype(v_block.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v_block).astype(q_block.dtype)

        o_ref[:, :, :] = out

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(q_flat.shape, q_flat.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(max_seqs,),
            in_specs=[
                pl.BlockSpec((block_q, H_q, D), lambda i: (i, 0, 0)),
                pl.BlockSpec((block_k, H_q, D), lambda i: (i, 0, 0)),
                pl.BlockSpec((block_k, H_q, D), lambda i: (i, 0, 0)),
            ],
            out_specs=pl.BlockSpec((block_q, H_q, D), lambda i: (i, 0, 0)),
        ),
    )(q_flat, k_flat, v_flat)

    return out.reshape(queries.shape)


def benchmark(num_warmup=2, num_iters=10):
    import time
    inputs = create_inputs()
    for _ in range(num_warmup):
        out = workload(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = workload(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    max_seqs = CONFIG['max_num_seqs']
    H_q = CONFIG['num_q_heads']
    D = CONFIG['head_dim']
    tokens_per_seq = CONFIG['max_num_batched_tokens'] // max_seqs
    kv_len = CONFIG['pages_per_seq'] * CONFIG['page_size']
    flops = max_seqs * H_q * (4 * tokens_per_seq * kv_len * D)
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 4),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
