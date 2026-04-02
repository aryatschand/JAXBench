import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, input_size, hidden_size, scaling_factor):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scaling_factor = scaling_factor

        key = jax.random.PRNGKey(0)
        self.weight = jax.random.normal(key, (hidden_size, input_size))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        # Precompute summed weights to reduce GEMM -> GEMV
        w_sum = jnp.sum(self.weight, axis=0, keepdims=True)  # (1, input_size)

        batch_size, input_size = x.shape
        block_m = 128  # must divide batch_size

        def kernel(x_ref, w_ref, o_ref):
            x_block = x_ref[:, :]                  # (block_m, K)
            w_vec = w_ref[0, :]                   # (K,)
            acc = jnp.sum(x_block * w_vec, axis=1, keepdims=True)
            o_ref[:] = acc * (self.scaling_factor / 2.0)

        grid_m = batch_size // block_m

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((batch_size, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m,),
                in_specs=[
                    pl.BlockSpec((block_m, input_size), lambda i: (i, 0)),
                    pl.BlockSpec((1, input_size), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, 1), lambda i: (i, 0)),
            ),
        )(x, w_sum)

batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 1.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, input_size))
    return [x]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
