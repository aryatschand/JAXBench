import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        kernel_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = jnp.zeros(kernel_shape)
        if bias:
            self.bias = jnp.zeros(out_channels)
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        kernel = jnp.flip(kernel, axis=(0, 1, 2))

        batch_size, d_in, h_in, w_in, in_ch = x.shape

        if self.stride > 1:
            d_dil = d_in + (d_in - 1) * (self.stride - 1)
            h_dil = h_in + (h_in - 1) * (self.stride - 1)
            w_dil = w_in + (w_in - 1) * (self.stride - 1)
            x_dil = jnp.zeros((batch_size, d_dil, h_dil, w_dil, in_ch), dtype=x.dtype)
            x_dil = x_dil.at[:, ::self.stride, ::self.stride, ::self.stride, :].set(x)
        else:
            x_dil = x
            d_dil, h_dil, w_dil = d_in, h_in, w_in

        eff_k = self.dilation * (self.kernel_size - 1) + 1
        pad = eff_k - 1 - self.padding

        d_out = d_dil + 2 * pad - eff_k + 1
        h_out = h_dil + 2 * pad - eff_k + 1
        w_out = w_dil + 2 * pad - eff_k + 1
        out_ch = kernel.shape[3]

        total_voxels = batch_size * d_out * h_out * w_out

        def kernel_fn(x_ref, k_ref, o_ref):
            idx = pl.program_id(axis=0)

            b = idx // (d_out * h_out * w_out)
            rem = idx % (d_out * h_out * w_out)
            d = rem // (h_out * w_out)
            rem = rem % (h_out * w_out)
            h = rem // w_out
            w = rem % w_out

            acc = jnp.zeros((out_ch,), dtype=jnp.float32)

            for kd in range(self.kernel_size):
                for kh in range(self.kernel_size):
                    for kw in range(self.kernel_size):
                        id_in = d - pad + kd * self.dilation
                        ih_in = h - pad + kh * self.dilation
                        iw_in = w - pad + kw * self.dilation

                        valid = (id_in >= 0) & (id_in < d_dil) & \
                                (ih_in >= 0) & (ih_in < h_dil) & \
                                (iw_in >= 0) & (iw_in < w_dil)

                        x_val = jnp.where(
                            valid,
                            x_ref[b, id_in, ih_in, iw_in, :],
                            jnp.zeros((in_ch,), dtype=x_ref.dtype)
                        )

                        k_val = k_ref[kd, kh, kw, :, :]  # (out_ch, in_ch)
                        acc = acc + jnp.dot(k_val, x_val)

            o_ref[0, :] = acc.astype(o_ref.dtype)

        out = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((total_voxels, out_ch), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(total_voxels,),
                in_specs=[
                    pl.BlockSpec(x_dil.shape, lambda i: (0, 0, 0, 0, 0)),
                    pl.BlockSpec(kernel.shape, lambda i: (0, 0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, out_ch), lambda i: (i, 0)),
            ),
        )(x_dil, kernel)

        out = out.reshape((batch_size, d_out, h_out, w_out, out_ch))

        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, 1, 1, -1)

        out = jnp.transpose(out, (0, 4, 1, 2, 3))
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (16, 32, 16, 32, 32))
    return [x]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 2]
