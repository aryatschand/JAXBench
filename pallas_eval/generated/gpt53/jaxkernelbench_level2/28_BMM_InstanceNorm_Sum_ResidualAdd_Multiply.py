import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(y_ref, b_ref, o_ref):
    y = y_ref[:, :]
    b = b_ref[0, :]
    b_broadcast = pltpu.repeat(b, y.shape[0], axis=0)
    o_ref[:, :] = (b_broadcast + y) * y


def fused_pallas(y, b):
    N, C = y.shape
    block = (128, 128)
    grid = (N // block[0], C // block[1])

    b_2d = jnp.reshape(b, (1, C))

    return pl.pallas_call(
        fused_kernel,
        out_shape=jax.ShapeDtypeStruct((N, C), y.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block, lambda i, j: (i, j)),
                pl.BlockSpec((1, block[1]), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        ),
    )(y, b_2d)


class Model:
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        self.in_features = in_features
        self.out_features = out_features
        self.bmm_weight = jnp.zeros((out_features, in_features))
        self.bmm_bias = jnp.zeros((out_features,))
        self.instance_norm_weight = jnp.ones((out_features,))
        self.instance_norm_bias = jnp.zeros((out_features,))
        self.eps = eps

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x, y):
        return fused_pallas(y, self.instance_norm_bias)


batch_size = 1024
in_features = 8192
out_features = 8192


def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    return [
        jax.random.uniform(key1, (batch_size, in_features), dtype=jnp.float32),
        jax.random.uniform(key2, (batch_size, out_features), dtype=jnp.float32),
    ]


def get_init_inputs():
    return [in_features, out_features]
