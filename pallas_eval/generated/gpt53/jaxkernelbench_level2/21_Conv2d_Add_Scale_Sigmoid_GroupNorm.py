import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def groupnorm_kernel(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[:, :]
    w = w_ref[:, :]
    b = b_ref[:, :]
    mean = jnp.mean(x)
    var = jnp.var(x)
    x_hat = (x - mean) / jnp.sqrt(var + 1e-5)
    o_ref[:, :] = x_hat * w + b

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        self.conv_weight = jnp.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = jnp.zeros((out_channels,))
        self.bias = jnp.zeros(bias_shape)
        self.scale = jnp.zeros(scale_shape)
        self.group_norm_weight = jnp.ones((out_channels,))
        self.group_norm_bias = jnp.zeros((out_channels,))
        self.num_groups = num_groups
        self.out_channels = out_channels

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel = jnp.transpose(self.conv_weight, (2, 3, 1, 0))

        x = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        x = x + self.conv_bias.reshape(1, 1, 1, -1)
        x = jnp.transpose(x, (0, 3, 1, 2))

        x = x + self.bias
        x = x * self.scale
        x = jax.nn.sigmoid(x)

        N, C, H, W = x.shape
        G = self.num_groups
        K = (C // G) * H * W

        x = jnp.reshape(x, (N * G, K, 1))

        # Build per-position weight/bias
        channel_ids = jnp.arange(K) // (H * W)
        group_offsets = jnp.repeat(jnp.arange(G) * (C // G), K // (C // G))
        full_channel_ids = (channel_ids + group_offsets[:K])

        w = self.group_norm_weight[full_channel_ids]
        b = self.group_norm_bias[full_channel_ids]

        w = jnp.broadcast_to(w, (N * G, K)).reshape(N * G, K, 1)
        b = jnp.broadcast_to(b, (N * G, K)).reshape(N * G, K, 1)

        block = (K, 1)
        grid = (N * G,)

        out = pl.pallas_call(
            groupnorm_kernel,
            out_shape=jax.ShapeDtypeStruct((N * G, K, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i: (i, 0)),
                    pl.BlockSpec(block, lambda i: (i, 0)),
                    pl.BlockSpec(block, lambda i: (i, 0)),
                ],
                out_specs=pl.BlockSpec(block, lambda i: (i, 0)),
            ),
        )(x, w, b)

        out = jnp.reshape(out, (N, C, H, W))
        return out

batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]
