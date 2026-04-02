import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        self.weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.bias_conv = jnp.zeros(out_channels)
        self.scaling_factor = scaling_factor
        self.bias = jnp.zeros(bias_shape)
        self.pool_kernel_size = pool_kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.weight.shape[2]
        OC = self.weight.shape[0]
        P = self.pool_kernel_size

        # NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))

        Hc = H - K + 1
        Wc = W - K + 1
        Hp = Hc // P
        Wp = Wc // P

        # reshape for 2D requirement
        x_flat = x_nhwc.reshape(N * H * W, C)
        bias_nhwc = jnp.transpose(self.bias, (1, 2, 0)).reshape(1, OC)

        def kernel_fn(x_ref, w_ref, bc_ref, b_ref, o_ref):
            pid = pl.program_id(0)

            rows = o_ref.shape[0]
            cols = o_ref.shape[1]

            for idx in range(rows * cols):
                r = idx // cols
                c = idx % cols

                n = r // Hp
                hp = r % Hp

                wp = c // OC
                oc = c % OC

                max_val = -jnp.inf

                for ph in range(P):
                    for pw in range(P):
                        h = hp * P + ph
                        w = wp * P + pw

                        acc = 0.0
                        for kh in range(K):
                            for kw in range(K):
                                for ic in range(C):
                                    xi = x_ref[(n * H + (h + kh)) * W + (w + kw), ic]
                                    wi = w_ref[kh, kw, ic, oc]
                                    acc += xi * wi

                        acc = acc + bc_ref[oc]
                        acc = jnp.tanh(acc)
                        acc = acc * self.scaling_factor
                        acc = acc + b_ref[0, oc]

                        max_val = jnp.maximum(max_val, acc)

                o_ref[r, c] = max_val

        out_shape = (N * Hp, Wp * OC)

        block = (128, 128)
        grid = (out_shape[0] // block[0],)

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block[0] * W, C), lambda i: (i, 0)),
                    pl.BlockSpec((K, K, C, OC), lambda i: (0, 0, 0, 0)),
                    pl.BlockSpec((OC,), lambda i: (0,)),
                    pl.BlockSpec((1, OC), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x_flat, kernel, self.bias_conv, bias_nhwc)

        out = out.reshape(N, Hp, Wp, OC)
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 4

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
