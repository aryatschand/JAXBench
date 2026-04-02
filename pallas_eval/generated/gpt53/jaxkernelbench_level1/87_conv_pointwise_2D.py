import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        weight_shape = (1, 1, in_channels, out_channels)
        self.conv1d_weight = jnp.zeros(weight_shape)
        self.conv1d_bias = jnp.zeros(out_channels) if bias else None
        self.use_bias = bias

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        N, H, W, C = x.shape
        OC = self.conv1d_weight.shape[-1]

        # Reshape to 2D: (N*H*W, C)
        x2d = x.reshape((N * H * W, C))
        w2d = self.conv1d_weight.reshape((C, OC))

        M = x2d.shape[0]
        K = C
        Nout = OC

        bm = min(128, M)
        bn = min(128, Nout)
        bk = 128

        def kernel(x_ref, w_ref, o_ref):
            acc = jnp.zeros((bm, bn), dtype=jnp.float32)

            def body(k, acc):
                k_start = k * bk
                x_block = x_ref[:, k_start:k_start + bk]
                w_block = w_ref[k_start:k_start + bk, :]
                acc = acc + jnp.matmul(x_block, w_block, preferred_element_type=jnp.float32)
                return acc

            num_k = K // bk
            acc = jax.lax.fori_loop(0, num_k, body, acc)
            o_ref[...] = acc.astype(o_ref.dtype)

        y2d = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, Nout), x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(M // bm, Nout // bn),
                in_specs=[
                    pl.BlockSpec((bm, bk), lambda i, j: (i, 0)),
                    pl.BlockSpec((bk, bn), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x2d, w2d)

        if self.conv1d_bias is not None:
            y2d = y2d + self.conv1d_bias

        # Reshape back NHWC -> NCHW
        y = y2d.reshape((N, H, W, OC))
        y = jnp.transpose(y, (0, 3, 1, 2))
        return y


# Test code - REDUCED SIZE for memory
batch_size = 4
in_channels = 64
out_channels = 128
width = 512
height = 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]
