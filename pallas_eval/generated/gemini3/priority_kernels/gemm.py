` or `grid=(..., K//BK)`, that would be much more complex because TPU Pallas doesn't support atomic adds. To do a 3D grid reduction, one has to use `scratch_memory` and a separate reduction step, which is notoriously difficult and often slower than just looping inside the kernel.
Actually, looping inside the kernel IS the standard way to do GEMM in Pallas TPU. The accumulator is kept in registers/VMEM.
I will stick to the inner loop approach. It is robust, fast, and fits the constraints perfectly.

Let's do a final sanity check on the code structure.
```python
"""Dense bf16 GEMM — Llama-70B FFN dimensions.

Baseline dense matrix multiplication at Llama-3.1-70B hidden-to-FFN scale.
A = (8192, 8192), B = (8192, 28672) — matches hidden_dim -> mlp_dim projection.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    'name': 'gemm_llama70b',
    'model': 'Llama-3.1-70B',
    'operator': 'dense_matmul',
    'M': 8192,
    'K': 8192,
    'N': 28672,
}

def create_inputs(dtype=jnp.bfloat16):
    """Returns (A, B) matrices."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    A = jax.random.normal(k1, (M, K), dtype=dtype)
    B = jax.random.normal(k2, (K, N), dtype=dtype) * 0.02
    return A, B

def gemm_kernel(a_ref, b_ref, c_ref):
    BM = a_ref.shape[0]
    BN = b_ref.shape[1]
    K = a_ref.shape[1]
    BK = 512
    
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    
    for i in range(K // BK):
        a_tile = a_ref[:, i * BK : (i + 1) * BK]
        b_tile = b_ref[i * BK : (i + 1) * BK, :]
        acc = acc + jnp.dot(a_tile, b_tile, preferred_element_type=jnp
