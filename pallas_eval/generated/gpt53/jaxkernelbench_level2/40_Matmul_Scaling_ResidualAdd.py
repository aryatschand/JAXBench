import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, scaling_factor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((1, out_features))
        self.scaling_factor = jnp.array([[scaling_factor]], dtype=jnp.float32)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            val = jnp.array(value)
            if name == "bias":
                val = val.reshape(1, -1)
            setattr(self, name.replace('.', '_'), val)

    def forward(self, x):
        def kernel(x_ref, w_ref, b_ref, s_ref, o_ref):
            acc = jnp.matmul(
                x_ref[:, :],
                w_ref[:, :],
                preferred_element_type=jnp.float32
            )
            y = acc + b_ref[:, :]
            o_ref[:, :] = y * s_ref[0, 0] + y

        B, K = x.shape
        K2, N = self.weight.shape

        bm = 128
        bn = 128

        grid = (B // bm, N // bn)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, K), lambda i, j: (i, 0)),   # x
                    pl.BlockSpec((K, bn), lambda i, j: (0, j)),   # w
                    pl.BlockSpec((1, bn), lambda i, j: (0, j)),   # bias
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),    # scaling
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x, self.weight, self.bias, self.scaling_factor)

batch_size = 16384
in_features = 4096
out_features = 4096
scaling_factor = 0.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
