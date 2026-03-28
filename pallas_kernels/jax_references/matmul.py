"""JAX reference for Pallas matmul — Llama-3.1-70B FFN dimensions."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'pallas_matmul_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'pallas_matmul',
    'M': 8192,
    'K': 8192,
    'N': 28672,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    x = jax.random.normal(k1, (M, K), dtype=dtype)
    y = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    return x, y


def workload(x, y):
    return jnp.dot(x, y)
