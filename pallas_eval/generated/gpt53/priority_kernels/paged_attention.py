import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_70b_paged_attention',
    'model': 'Llama-3.1-70B',
    'operator': 'paged_attention',
    'num_seqs': 32,
    'max_seq_len': 2048,
    'num_query_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 128,
    'page_size': 16,
    'pages_per_seq': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    num_seqs = CONFIG['num_seqs']
    max_seq_len = CONFIG['max_seq_len']
    num_q_heads = CONFIG['num_query_heads']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = CONFIG['pages_per_seq']
    total_pages = num_seqs * pages_per_seq

    max_num_tokens = num_seqs
    queries = jax.random.normal(keys[0], (max_num_tokens, num_q_heads, head_dim), dtype=dtype)
    k_pages = jax.random.normal(keys[1], (total_pages, page_size, num_kv_heads, head_dim), dtype=dtype) * 0.02
    v_pages = jax.random.normal(keys[2], (total_pages, page_size, num_kv_heads, head_dim), dtype=dtype) * 0.02

    kv_lens = jnp.full((num_seqs,), max_seq_len, dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)

    return queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens


def workload(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    num_seqs = CONFIG['num_seqs']
    num_q_heads = CONFIG['num_query_heads']
    num_kv_heads = CONFIG['num_kv_heads']
    head_dim = CONFIG['head_dim']
    page_size = CONFIG['page_size']
    pages_per_seq = CONFIG['pages_per_seq']
    num_q_per_kv = num_q_heads // num_kv_heads
    sm_scale = head_dim ** -0.5
    max_seq_len = pages_per_seq * page_size

    def kernel(q_ref, k_ref, v_ref, kv_len_ref, page_idx_ref, o_ref):
        seq = pl.program_id(0)
        head = pl.program_id(1)

        q = q_ref[seq, head, :].astype(jnp.float32)
        kv_head = head // num_q_per_kv
        kv_len = kv_len_ref[seq]

        def body(t, state):
            m, l, acc = state

            page = t // page_size
            offset = t % page_size
            page_id = page_idx_ref[seq, page]

            k = k_ref[page_id, offset, kv_head, :].astype(jnp.float32)
            v = v_ref[page_id, offset, kv_head, :].astype(jnp.float32)

            score = jnp.dot(q, k) * sm_scale
            score = jnp.where(t < kv_len, score, -1e30)

            new_m = jnp.maximum(m, score)
            exp_m = jnp.exp(m - new_m)
            exp_s = jnp.exp(score - new_m)

            l = l * exp_m + exp_s
            acc = acc * exp_m + exp_s * v

            return (new_m, l, acc)

        init = (-1e30, 0.0, jnp.zeros((head_dim,), dtype=jnp.float32))
        m, l, acc = jax.lax.fori_loop(0, max_seq_len, body, init)

        out = (acc / l).astype(o_ref.dtype)
        o_ref[seq, head, :] = out

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_seqs, num_q_heads, head_dim), queries.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_seqs, num_q_heads),
            in_specs=[
                pl.BlockSpec(queries.shape, lambda i, j: (0, 0, 0)),
                pl.BlockSpec(k_pages.shape, lambda i, j: (0, 0, 0, 0)),
                pl.BlockSpec(v_pages.shape, lambda i, j: (0, 0, 0, 0)),
                pl.BlockSpec(kv_lens.shape, lambda i, j: (0,)),
                pl.BlockSpec(page_indices.shape, lambda i, j: (0, 0)),
            ],
            out_specs=pl.BlockSpec((num_seqs, num_q_heads, head_dim), lambda i, j: (0, 0, 0)),
        ),
    )(queries, k_pages, v_pages, kv_lens, page_indices)


def benchmark(num_warmup=5, num_iters=100):
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
    num_seqs = CONFIG['num_seqs']
    num_q_heads = CONFIG['num_query_heads']
    max_seq_len = CONFIG['max_seq_len']
    head_dim = CONFIG['head_dim']
    flops = num_seqs * num_q_heads * (4 * max_seq_len * head_dim)
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
