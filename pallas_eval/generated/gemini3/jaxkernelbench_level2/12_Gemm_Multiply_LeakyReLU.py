"""
JAXBench Level 2 - Gemm_Multiply_LeakyReLU
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gemm_mult_leaky_relu_kernel(x_ref, w_ref, b_ref, mult_ref, slope_ref, o_ref):
    x_val = x_ref[...]
    w_val = w_ref[...]
    b_val = b_ref[...]
    mult_val = mult_ref[...]
    slope_val = slope_ref[...]
    
    # Matmul with f32 accumulation
    acc = jnp.dot(x_val, w_val, preferred_element_type=jnp.float32)
    
    # Add bias
    acc = acc + b_val
    
    # Multiply by scalar
    acc = acc * mult_val
    
    # LeakyReLU activation
    out = jnp.where(acc >= 0, acc, acc * slope_val)
    
    # Write back to HBM
    o_ref[...] = out.astype(o_ref.dtype)

class Model:
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Reshape 1D tensors and scalars to 2D as required by Pallas TPU
        bias_2d = self.bias.reshape(1, -1)
        mult_2d = jnp.asarray(self.multiplier).reshape(1, 1)
        slope_2d = jnp.asarray(self.negative_slope).reshape(1, 1)
        
        # Define block sizes (powers of 2, multiples of 128 for f32/bf16)
        BM = min(x.shape[0], 256)
        BN = min(self.weight.shape[1], 256)
        BK = self.weight.shape[0]
        
        grid_shape = (x.shape[0] // BM, self.weight.shape[1] // BN)
        out_shape = jax.ShapeDtypeStruct((x.shape[0], self.weight.shape[1]), x.dtype)
        
        return pl.pallas_call(
            gemm_mult_leaky_relu_kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, BK), lambda i, j: (i, 0)),
                    pl.BlockSpec((BK, BN), lambda i, j: (0, j)),
                    pl.BlockSpec((1, BN), lambda i, j: (0, j)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(x, self.weight, bias_2d, mult_2d, slope_2d)

batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
