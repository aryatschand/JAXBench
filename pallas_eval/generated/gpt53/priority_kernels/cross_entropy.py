import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'llama3_8b_cross_entropy',
    'model': 'Llama-3.1-8B',
    'operator': 'fused_cross_entropy',
    'batch_tokens': 4096,
    'hidden_dim': 4096,
    'vocab_size': 128256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    B, H, V = CONFIG['batch_tokens'], CONFIG['hidden_dim'], CONFIG['vocab_size']
    hidden = jax.random.normal(k1, (B, H), dtype=dtype)
    weight = jax.random.normal(k2, (H, V), dtype=dtype) * 0.02
    labels = jax.random.randint(k3, (B,), 0, V)
    return hidden, weight, labels


def matmul_kernel(x_ref, w_ref, o_ref):
    x = x_ref[:, :]          # (BM, H)
    w = w_ref[:, :]          # (H, BN)
    acc = jnp.zeros((x.shape[0], w.shape[1]), dtype=jnp.float32)

    def body(k, acc):
        xk = x[:, k:k+128]
        wk = w[k:k+128, :]
        return acc + jnp.dot(xk.astype(jnp.float32), wk.astype(jnp.float32))

    acc = jax.lax.fori_loop(0, x.shape[1], body, acc)
    o_ref[:, :] = acc.astype(o_ref.dtype)


def pallas_matmul(x, w):
    B, H = x.shape
    _, V = w.shape

    BM = 128
    BN = 128

    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((B, V), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(B // BM, V // BN),
            in_specs=[
                pl.BlockSpec((BM, H), lambda i, j: (i, 0)),
                pl.BlockSpec((H, BN), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
        ),
    )(x, w)


def workload(hidden, weight, labels):
    logits = pallas_matmul(hidden, weight)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


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
    B, H, V = CONFIG['batch_tokens'], CONFIG['hidden_dim'], CONFIG['vocab_size']
    flops = B * H * V * 2 + B * V * 3
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
