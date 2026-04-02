import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_kernel(x_ref, w_ref, b_ref, d_ref, o_ref):
    x = x_ref[:, :]
    w = w_ref[:, :]
    b = b_ref[:, :]
    d = d_ref[:, :]
    out = jnp.dot(x, w) + b
    out = out / d
    out = jnn.gelu(out)
    o_ref[:, :] = out

class Model:
    def __init__(self, input_size, output_size, divisor):
        self.weight = jnp.zeros((input_size, output_size))
        self.bias = jnp.zeros((1, output_size))
        self.divisor = jnp.array([[divisor]])

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == "bias":
                setattr(self, name.replace('.', '_'), jnp.array(value).reshape(1, -1))
            else:
                setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        B, I = x.shape
        O = self.weight.shape[1]

        block_m = 128
        block_n = 128

        grid = (B // block_m, O // block_n)

        return pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((B, O), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, I), lambda i, j: (i, 0)),
                    pl.BlockSpec((I, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x, self.weight, self.bias, self.divisor)

batch_size = 1024
input_size = 8192 
output_size = 8192
divisor = 10.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, output_size, divisor]
