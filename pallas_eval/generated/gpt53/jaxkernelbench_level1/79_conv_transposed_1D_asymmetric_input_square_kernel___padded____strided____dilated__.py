import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def conv_transpose_kernel(x_ref, w_ref, o_ref, stride, pad):
    x = x_ref[...]      # (B*W_in, C_in)
    w = w_ref[...]      # (K_eff, C_out, C_in)

    BWi, Cin = x.shape
    K, Cout, _ = w.shape
    BWo, _ = o_ref.shape

    pid = pl.program_id(axis=0)
    start = pid * o_ref.shape[0]
    end = start + o_ref.shape[0]

    out = jnp.zeros((o_ref.shape[0], Cout), dtype=x.dtype)

    def body(i, acc):
        out_idx = start + i

        def inner(j, acc2):
            in_idx = (out_idx + pad - j) // stride
            valid = (in_idx >= 0) & (in_idx < BWi) & ((out_idx + pad - j) % stride == 0)

            x_val = jnp.where(valid[:, None], x[in_idx], 0.0)
            w_val = w[j]

            acc2 = acc2 + jnp.dot(x_val, w_val.T)
            return acc2

        acc = jax.lax.fori_loop(0, K, inner, acc)
        return acc

    out = jax.lax.fori_loop(0, o_ref.shape[0], body, out)

    o_ref[...] = out


class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size))
        if bias:
            self.bias = jnp.zeros((out_channels,))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        self._kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 1))  # (N, W, C)
        N, W, Cin = x.shape

        kernel = jnp.transpose(self.weight, (2, 1, 0))  # (K, Cout, Cin)

        if self.dilation > 1:
            K = kernel.shape[0]
            Kd = K + (K - 1) * (self.dilation - 1)
            dilated = jnp.zeros((Kd, kernel.shape[1], kernel.shape[2]), dtype=kernel.dtype)
            idx = jnp.arange(K) * self.dilation
            dilated = dilated.at[idx].set(kernel)
            kernel = dilated
            K_eff = Kd
        else:
            K_eff = self._kernel_size

        pad = K_eff - 1 - self.padding

        W_out = (W - 1) * self.stride - 2 * self.padding + K_eff

        x2 = x.reshape((N * W, Cin))
        o2 = jnp.zeros((N * W_out, kernel.shape[1]), dtype=x.dtype)

        block = (1024, kernel.shape[1])
        grid = (o2.shape[0] // block[0],)

        o2 = pl.pallas_call(
            lambda x_ref, w_ref, o_ref: conv_transpose_kernel(
                x_ref, w_ref, o_ref, self.stride, pad
            ),
            out_shape=jax.ShapeDtypeStruct(o2.shape, o2.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block[0], Cin), lambda i: (i, 0)),
                    pl.BlockSpec(kernel.shape, lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x2, kernel)

        out = o2.reshape((N, W_out, -1))

        if self.bias_flag:
            out = out + self.bias[None, None, :]

        out = jnp.transpose(out, (0, 2, 1))
        return out

    @property
    def kernel_size(self):
        return self.weight.shape[2]


def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (16, 32, 131072))
    return [x]


def get_init_inputs():
    return [32, 64, 3, 2, 1, 2]
