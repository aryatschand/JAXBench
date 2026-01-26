"""
JAXBench Level 2 - ConvTranspose3d_LayerNorm_GELU_Scaling
Translated from KernelBench PyTorch to JAX using bedrock/sonnet.
"""

"""
JAXBench Level 2 - Task 34: ConvTranspose3d_LayerNorm_GELU_Scaling
Manually implemented JAX version
"""

import jax
import jax.numpy as jnp
from jax import lax

class Model:
    """
    Model that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.eps = eps
        self.scaling_factor = scaling_factor

        # ConvTranspose3d weights - PyTorch shape: (in_channels, out_channels, kD, kH, kW)
        key = jax.random.PRNGKey(0)
        self.conv_transpose_weight = jax.random.normal(key, (in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = jnp.zeros((out_channels,)) if bias else None

        # LayerNorm parameters (normalizes over out_channels dimension)
        self.layer_norm_weight = jnp.ones((out_channels,))
        self.layer_norm_bias = jnp.zeros((out_channels,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        # x: (N, C, D, H, W) in PyTorch format
        # Convert to NDHWC for JAX
        x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC

        # ConvTranspose3d using manual approach matching level1/task77
        kernel = jnp.transpose(self.conv_transpose_weight, (2, 3, 4, 1, 0))
        kernel = jnp.flip(kernel, axis=(0, 1, 2))

        batch_size, d_in, h_in, w_in, channels = x.shape
        k = self.kernel_size

        if self.stride > 1:
            d_dilated = d_in + (d_in - 1) * (self.stride - 1)
            h_dilated = h_in + (h_in - 1) * (self.stride - 1)
            w_dilated = w_in + (w_in - 1) * (self.stride - 1)
            x_dilated = jnp.zeros((batch_size, d_dilated, h_dilated, w_dilated, channels), dtype=x.dtype)
            x_dilated = x_dilated.at[:, ::self.stride, ::self.stride, ::self.stride, :].set(x)
            x = x_dilated

        pad = k - 1 - self.padding
        jax_padding = ((pad, pad), (pad, pad), (pad, pad))

        x = lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding=jax_padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC')
        )

        if self.conv_transpose_bias is not None:
            x = x + self.conv_transpose_bias.reshape(1, 1, 1, 1, -1)

        # Convert back to NCDHW
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW

        # LayerNorm: nn.LayerNorm(out_channels) normalizes over the last dimension
        # With x shape (N, C, D, H, W), the last dim is W
        # After ConvTranspose3d, the shape is (N, 64, 32, 64, 64) where W=64=out_channels
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        # LayerNorm weight/bias are shape (out_channels,) and applied to the last dimension
        x = x * self.layer_norm_weight + self.layer_norm_bias

        # GELU activation
        x = jax.nn.gelu(x)

        # Scaling
        x = x * self.scaling_factor

        return x

batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]
