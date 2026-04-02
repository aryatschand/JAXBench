import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, w_ref, b_ref, s_ref, rm_ref, rv_ref, bw_ref, bb_ref, o_ref):
    x = x_ref[...]                      # (B_block, I)
    w = w_ref[...]                      # (O_block, I)
    b = b_ref[...]                      # (1, O_block)
    s = s_ref[...]                      # (1, O_block)
    rm = rm_ref[...]                    # (1, O_block)
    rv = rv_ref[...]                    # (1, O_block)
    bw = bw_ref[...]                    # (1, O_block)
    bb = bb_ref[...]                    # (1, O_block)

    y = jnp.dot(x, w.T) + b
    y = y * s
    y = (y - rm) / jnp.sqrt(rv + 1e-5)
    y = bw * y + bb

    o_ref[...] = y


class Model:
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        self.gemm_weight = jnp.zeros((out_features, in_features))
        self.gemm_bias = jnp.zeros((out_features,))
        self.scale = jnp.zeros(scale_shape)
        self.bn_weight = jnp.ones((out_features,))
        self.bn_bias = jnp.zeros((out_features,))
        self.bn_running_mean = jnp.zeros((out_features,))
        self.bn_running_var = jnp.ones((out_features,))
        self.eps = eps

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        B, I = x.shape
        O = self.gemm_weight.shape[0]

        block_b = 128
        block_o = 128

        grid = (B // block_b, O // block_o)

        # reshape params to 2D
        b = self.gemm_bias.reshape(1, O)
        s = self.scale.reshape(1, O)
        rm = self.bn_running_mean.reshape(1, O)
        rv = self.bn_running_var.reshape(1, O)
        bw = self.bn_weight.reshape(1, O)
        bb = self.bn_bias.reshape(1, O)

        return pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((B, O), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_b, I), lambda i, j: (i, 0)),
                    pl.BlockSpec((block_o, I), lambda i, j: (j, 0)),
                    pl.BlockSpec((1, block_o), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_o), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_o), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_o), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_o), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_o), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_b, block_o), lambda i, j: (i, j)),
            ),
        )(x, self.gemm_weight, b, s, rm, rv, bw, bb)


batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, scale_shape]
