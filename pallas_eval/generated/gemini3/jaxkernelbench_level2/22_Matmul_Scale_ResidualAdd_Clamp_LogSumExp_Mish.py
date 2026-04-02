"""
JAXBench Level 2 - Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        self.weight = jnp.zeros((hidden_size, input_size))
        self.bias = jnp.zeros(hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Perform the heavy matrix multiplication using JAX's highly optimized MXU path
        x_matmul = jnp.matmul(x, self.matmul_weight.T)
        
        # Reshape bias to 2D to satisfy Pallas constraints
        bias_2d = self.matmul_bias[None, :]
        
        # Capture scalars for the Pallas kernel closure
        scale = self.scale_factor
        cmin = self.clamp_min
        cmax = self.clamp_max
        
        def epilogue_kernel(x_ref, bias_ref, o_ref):
            x_val = x_ref[...]
            bias_val = bias_ref[...]
            
            # Add bias
            bias_rep = pltpu.repeat(bias_val, x_val.shape[0], axis=0)
            x_val = x_val + bias_rep
            
            # Scale and Residual Add: x = x * scale; x = x + x  =>  x = x * (scale * 2.0)
            x_val = x_val * (scale * 2.0)
            
            # Clamp
            x_val = jnp.clip(x_val, cmin, cmax)
            
            # LogSumExp over axis 1
            max_val = jnp.max(x_val, axis=1, keepdims=True)
            max_val_rep = pltpu.repeat(max_val, x_val.shape[1], axis=1)
            sum_exp = jnp.sum(jnp.exp(x_val - max_val_rep), axis=1, keepdims=True)
            lse = jnp.log(sum_exp) + max_val
            
            # Mish and final multiply:
            # softplus_x = logaddexp(lse, 0.0)
            # mish_x = lse * tanh(softplus_x)
            # out = lse * mish_x  =>  lse^2 * tanh(softplus_x)
            lse_sq = lse * lse
            softplus_x = jnp.logaddexp(lse, 0.0)
            out = lse_sq * jnp.tanh(softplus_x)
            
            o_ref[...] = out

        block_m = 128
        block_n = x_matmul.shape[1]
        grid_m = x_matmul.shape[0] // block_m
        
        out = pl.pallas_call(
            epilogue_kernel,
            out_shape=jax.ShapeDtypeStruct((x_matmul.shape[0], 1), x_matmul.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m,),
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i: (i, 0)),
                    pl.BlockSpec((1, block_n), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, 1), lambda i: (i, 0)),
            ),
        )(x_matmul, bias_2d)
        
        return out

batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
