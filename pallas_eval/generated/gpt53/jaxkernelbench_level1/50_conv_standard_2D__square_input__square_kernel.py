import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, num_classes=1000):
        self.conv1_weight = None
        self.conv1_bias = None
        self.weight_shape = (96, 3, 11, 11)
        self.bias_shape = (96,)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        weight = jnp.transpose(self.conv1_weight, (2, 3, 1, 0))

        N, H, W, C = x.shape
        KH, KW, CI, CO = weight.shape

        OH = (H + 2 * 2 - KH) // 4 + 1
        OW = (W + 2 * 2 - KW) // 4 + 1

        block = (1, 11, 11, 32)
        grid = (N, OH // 11, OW // 11, CO // 32)

        def kernel(x_ref, w_ref, b_ref, o_ref):
            x_block = x_ref[...]
            w_full = w_ref[...]
            b_full = b_ref[...]

            acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)

            def body_kh(kh, acc):
                def body_kw(kw, acc):
                    x_slice = x_block[:, kh:kh+11, kw:kw+11, :]
                    w_slice = w_full[kh, kw, :, :]
                    acc = acc + jnp.einsum('bhwc,co->bhwo', x_slice, w_slice)
                    return acc
                acc = jax.lax.fori_loop(0, KW, body_kw, acc)
                return acc

            acc = jax.lax.fori_loop(0, KH, body_kh, acc)

            acc = acc + b_full.reshape(1, 1, 1, -1)
            o_ref[...] = acc.astype(o_ref.dtype)

        y = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, OH, OW, CO), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda n, i, j, k: (n, i, j, 0)),
                    pl.BlockSpec(weight.shape, lambda n, i, j, k: (0, 0, 0, 0)),
                    pl.BlockSpec((CO,), lambda n, i, j, k: (0,)),
                ],
                out_specs=pl.BlockSpec(block, lambda n, i, j, k: (n, i, j, k)),
            ),
        )(x, weight, self.conv1_bias if self.conv1_bias is not None else jnp.zeros((CO,), x.dtype))

        y = jnp.transpose(y, (0, 3, 1, 2))
        return y

batch_size = 256
num_classes = 1000

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, 3, 224, 224))]

def get_init_inputs():
    return [num_classes]
