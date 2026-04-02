import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.kernel_size
        OH = H - K + 1
        OW = W - K + 1

        # NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        # HWIO
        w = jnp.transpose(self.weight, (2, 3, 1, 0))

        NHW = N * OH * OW
        OC = self.out_channels

        x_flat = x_nhwc.reshape(N, H, W, C)

        def kernel_fn(x_ref, w_ref, b_ref, o_ref):
            pid = pl.program_id(axis=0)

            n = pid // (OH * OW)
            rem = pid % (OH * OW)
            h = rem // OW
            w_out = rem % OW

            acc = jnp.zeros((OC,), dtype=jnp.float32)

            for kh in range(K):
                for kw in range(K):
                    x_val = x_ref[n, h + kh, w_out + kw, :]  # (C,)
                    w_val = w_ref[kh, kw, :, :]              # (C, OC)
                    acc += jnp.dot(x_val, w_val)

            acc = acc + b_ref[:]

            # HardSwish
            acc = acc * jnp.minimum(jnp.maximum(acc + 3.0, 0.0), 6.0) / 6.0
            # ReLU
            acc = jnp.maximum(acc, 0.0)

            o_ref[pid, :] = acc

        block = (1, OC)

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((NHW, OC), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(NHW,),
                in_specs=[
                    pl.BlockSpec(x_flat.shape, lambda i: (0, 0, 0, 0)),
                    pl.BlockSpec(w.shape, lambda i: (0, 0, 0, 0)),
                    pl.BlockSpec((OC,), lambda i: (0,)),
                ],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x_flat, w, self.bias)

        out = out.reshape(N, OH, OW, OC)
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
