"""
JAXBench Level 2 - Gemm_Max_Subtract_GELU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.nn import gelu

def epilogue_dim1_kernel(x_ref, o_ref):
    # Load the entire block into VMEM
    x = x_ref[...]
    
    # Fused epilogue operations
    x = jnp.max(x, axis=1, keepdims=True)
    x = x - jnp.mean(x, axis=1, keepdims=True)
    
    # Inline GELU to ensure compatibility within Pallas
    x = 0.5 * x * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))
    
    # Store the result
    o_ref[...] = x

def pallas_epilogue_dim1(x):
    B, F = x.shape
    
    # Choose a block size that fits in VMEM and divides B
    block_B = 32
    while block_B > 8 and B % block_B != 0:
        block_B //= 2
        
    # Fallback to JAX if B is not divisible by a reasonable block size
    if B % block_B != 0:
        x = jnp.max(x, axis=1, keepdims=True)
        x = x - jnp.mean(x, axis=1, keepdims=True)
        return gelu(x)

    grid = (B // block_B,)
    
    return pl.pallas_call(
        epilogue_dim1_kernel,
        out_shape=jax.ShapeDtypeStruct((B, 1), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[pl.BlockSpec((block_B, F), lambda i: (i, 0))],
            out_specs=pl.BlockSpec((block_B, 1), lambda i: (i, 0))
        )
    )(x)

class Model:
    def __init__(self, in_features, out_features, max_dim):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        self.max_dim = max_dim

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Highly optimized MXU matmul via standard JAX
        x = jnp.matmul(x, self.weight) + self.bias
        
        # Use custom Pallas kernel for the memory-bound epilogue if max_dim == 1
        if self.max_dim == 1:
            return pallas_epilogue_dim1(x)
        else:
            # Fallback for other max_dim values
            x = jnp.max(x, axis=self.max_dim, keepdims=True)
            x = x - jnp.mean(x, axis=1, keepdims=True)
            x = gelu(x)
            return x

batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, max_dim]
