import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = jnp.zeros((1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        padding = ((self.kernel_size - 1 - self.padding,) * 2,) * 3

        out = jax.lax.conv_transpose(
            x_ndhwc,
            kernel,
            strides=(self.stride,) * 3,
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )
        out = jnp.transpose(out, (0, 4, 1, 2, 3))

        # Mean over depth (keepdim)
        out = jnp.mean(out, axis=2, keepdims=True)

        # Reshape for kernel: (N, C, 1, H, W) -> (N*H*W, C)
        N, C, _, H, W = out.shape
        out_2d = jnp.reshape(jnp.transpose(out, (0, 3, 4, 1, 2)), (N * H * W, C))

        bias = jnp.reshape(self.bias, (1, C))

        def kernel_fn(x_ref, b_ref, o_ref):
            x = x_ref[:, :]
            b = b_ref[:, :]

            x = x + b

            # softmax along channel axis
            x_max = jnp.max(x, axis=1, keepdims=True)
            x_exp = jnp.exp(x - x_max)
            x_sum = jnp.sum(x_exp, axis=1, keepdims=True)
            x_soft = x_exp / x_sum

            x_tanh = jnp.tanh(x_soft)
            o_ref[:, :] = x_tanh * self.scaling_factor

        block_m = min(out_2d.shape[0], 128)
        block_n = min(out_2d.shape[1], 128)

        grid = (out_2d.shape[0] // block_m, out_2d.shape[1] // block_n)

        result_2d = pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(out_2d.shape, out_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(out_2d, bias)

        # Reshape back to (N, C, 1, H, W)
        result = jnp.reshape(result_2d, (N, H, W, C, 1))
        result = jnp.transpose(result, (0, 3, 4, 1, 2))

        return result

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (16, 16, 32, 128, 128))]

def get_init_inputs():
    return [16, 64, 3, 1, 1, 2.0]
