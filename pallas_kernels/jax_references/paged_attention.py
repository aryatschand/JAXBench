"""JAX reference for Pallas paged attention — Llama-3.1-70B decode dimensions.

Adapted from grouped_query_attention_reference in
jax/experimental/pallas/ops/tpu/paged_attention/util.py
and _reconstruct_kv from
jax/tests/pallas/tpu_paged_attention_kernel_test.py.
"""

import jax
import jax.numpy as jnp

MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

CONFIG = {
    'name': 'pallas_paged_attention_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_paged_attention',
    'batch': 32,
    'num_q_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'page_size': 16,
    'pages_per_seq': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch']
    H_q = CONFIG['num_q_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = CONFIG['pages_per_seq']
    total_num_pages = B * pages_per_seq
    q = jax.random.normal(k1, (B, H_q, D), dtype=dtype)
    k_pages = jax.random.normal(k2, (H_kv, total_num_pages, page_size, D), dtype=dtype)
    v_pages = jax.random.normal(k3, (H_kv, total_num_pages, page_size, D), dtype=dtype)
    seq_len = pages_per_seq * page_size
    lengths = jnp.full((B,), seq_len, dtype=jnp.int32)
    page_indices = jnp.arange(total_num_pages, dtype=jnp.int32).reshape(B, pages_per_seq)
    return q, k_pages, v_pages, lengths, page_indices


def _reconstruct_kv(page_indices, pages):
    """Upstream _reconstruct_kv from tpu_paged_attention_kernel_test.py."""
    num_kv_heads, _, _, head_dim = pages.shape

    def per_sequence_page_gather(pages, page_indices):
        return jnp.take(pages, page_indices, 1)

    gathered = jax.vmap(per_sequence_page_gather, in_axes=(None, 0))(
        pages, page_indices
    )
    batch_size = page_indices.shape[0]
    return gathered.reshape(batch_size, num_kv_heads, -1, head_dim)


def workload(queries, k_pages, v_pages, seq_lens, page_indices):
    """Upstream grouped_query_attention_reference from paged_attention/util.py."""
    k = _reconstruct_kv(page_indices, k_pages)
    v = _reconstruct_kv(page_indices, v_pages)

    batch_size, num_q_heads, head_dim = queries.shape
    _, num_kv_heads, max_seq_len, _ = k.shape

    num_groups = num_q_heads // num_kv_heads
    queries = queries.reshape(batch_size, num_kv_heads, num_groups, head_dim)

    qk = jnp.einsum(
        "bhgd,bhtd->bhgt",
        queries.astype(jnp.float32),
        k.astype(jnp.float32),
    )

    mask = jnp.arange(max_seq_len)[None] < seq_lens[:, None]
    qk += jnp.where(mask, 0.0, MASK_VALUE)[:, None, None, :]

    probs = jax.nn.softmax(qk, axis=-1).astype(v.dtype)

    attention = jnp.einsum("bhgt,bhtd->bhgd", probs, v)
    return attention.reshape(batch_size, num_q_heads, head_dim)
