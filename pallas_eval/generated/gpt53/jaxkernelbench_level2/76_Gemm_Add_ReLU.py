import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def gemm_add_relu_kernel(x_ref, w_ref, b_ref, o_ref):
    x_block = x_ref[:, :]
    w_block = w_ref[:, :]
    acc = jnp.matmul(x_block, w_block, preferred_element_type=jnp.float32)
    acc = acc + b_ref[:, :]
    o_ref[:, :] = jnp.maximum(acc, 0)


class Model:
    def __init__(self, in_features, out_features, bias_shape):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        batch_size, in_features = x.shape
        out_features = self.weight.shape[1]

        block_m = 128
        block_n = 128

        grid_m = batch_size // block_m
        grid_n = out_features // block_n

        bias_2d = self.bias.reshape(1, out_features)

        return pl.pallas_call(
            gemm_add_relu_kernel,
            out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_m, grid_n),
                in_specs=[
                    pl.BlockSpec((block_m, in_features), lambda i, j: (i, 0)),
                    pl.BlockSpec((in_features, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x, self.weight, bias_2d)


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, bias_shape]
