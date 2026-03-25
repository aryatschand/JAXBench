"""JAX reference for Pallas ragged paged attention — Llama-3.1-8B mixed prefill+decode.

Adapted from ref_ragged_paged_attention in
jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py.

Not jit-compatible: uses data-dependent slicing on per-sequence boundaries.
"""

import math

import jax
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

CONFIG = {
    'name': 'pallas_ragged_paged_attention_llama8b',
    'model': 'Llama-3.1-8B',
    'operator': 'pallas_ragged_paged_attention',
    'max_num_batched_tokens': 2048,
    'max_num_seqs': 32,
    'num_q_heads': 32,
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


def workload(queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    """Reference ragged paged attention from upstream JAX.

    Processes each sequence independently with data-dependent slicing.
    Must be run eagerly (not under jax.jit).
    """
    sm_scale = 1.0 / math.sqrt(CONFIG['head_dim'])
    mask_value = DEFAULT_MASK_VALUE
    _, _, num_combined_kv_heads, head_dim = kv_pages.shape
    num_kv_heads = num_combined_kv_heads // 2
    num_q_heads = queries.shape[1]
    num_query_per_kv = num_q_heads // num_kv_heads

    outputs = []
    for i in range(num_seqs[0]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        indices = page_indices[i]

        q = queries[q_start:q_end]
        k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]

        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)

        attn = jnp.einsum(
            "qhd,khd->hqk", q, k, preferred_element_type=jnp.float32
        )
        attn *= sm_scale

        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1
        )
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        attn += jnp.where(mask, mask_value, 0.0)

        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)
