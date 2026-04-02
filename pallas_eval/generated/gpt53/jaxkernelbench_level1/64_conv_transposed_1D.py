import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        key = jax.random.PRNGKey(0)
        weight_shape = (in_channels, out_channels, kernel_size)
        k = 1.0 / (in_channels * kernel_size)
        weight = jax.random.uniform(key, weight_shape) * (2.0 * jnp.sqrt(k)) - jnp.sqrt(k)
        self.weight = jnp.transpose(weight, (2, 1, 0))

        if bias:
            self.bias = jax.random.uniform(key, (out_channels,)) * (2.0 * jnp.sqrt(k)) - jnp.sqrt(k)
        else:
            self.bias = None

        self.stride = stride
        self.pytorch_padding = padding
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv1d_transpose.weight':
                value = jnp.transpose(jnp.array(value), (2, 1, 0))
                setattr(self, 'weight', value)
            elif name == 'conv1d_transpose.bias':
                setattr(self, 'bias', jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 1))

        pad = self.kernel_size - 1 - self.pytorch_padding
        L_in = x.shape[1]
        L_out = (L_in - 1) * self.stride - 2 * pad + self.kernel_size + self.output_padding

        N, _, C_in = x.shape
        C_out = self.weight.shape[1]

        Nb, Lb, Cb = 8, 128, 128
        Nb = min(Nb, N)
        Lb = min(Lb, L_out)
        Cb = min(Cb, C_out)

        grid = (N // Nb, L_out // Lb, C_out // Cb)

        def kernel(x_ref, w_ref, o_ref):
            x_full = x_ref[...]
            w_full = w_ref[...]

            n_id = pl.program_id(0)
            l_id = pl.program_id(1)
            c_id = pl.program_id(2)

            n_start = n_id * Nb
            l_start = l_id * Lb
            c_start = c_id * Cb

            out_block = jnp.zeros((Nb, Lb, Cb), dtype=x_full.dtype)

            for k in range(self.kernel_size):
                l_out_idx = jnp.arange(Lb) + l_start
                i_float = (l_out_idx + pad - k) / self.stride
                valid = (i_float == jnp.floor(i_float))
                i_idx = i_float.astype(jnp.int32)

                valid = valid & (i_idx >= 0) & (i_idx < L_in)

                for cin in range(C_in):
                    x_vals = jnp.where(
                        valid[None, :, None],
                        x_full[n_start:n_start+Nb, i_idx, cin][:, :, None],
                        0.0
                    )
                    w_vals = w_full[k, c_start:c_start+Cb, cin][None, None, :]
                    out_block += x_vals * w_vals

            o_ref[...] = out_block

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, L_out, C_out), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(x.shape, lambda i, j, k: (0, 0, 0)),
                    pl.BlockSpec(self.weight.shape, lambda i, j, k: (0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((Nb, Lb, Cb), lambda i, j, k: (i, j, k)),
            ),
        )(x, self.weight)

        if self.bias is not None:
            out = out + self.bias

        out = jnp.transpose(out, (0, 2, 1))
        return out


batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
length = 65536

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, in_channels, length))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
