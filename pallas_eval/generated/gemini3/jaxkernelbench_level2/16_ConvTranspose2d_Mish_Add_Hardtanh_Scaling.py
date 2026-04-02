import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def elementwise_kernel(x_ref, bias_ref, add_value_ref, scale_ref, o_ref):
    x = x_ref[:, :]
    bias = bias_ref[:, :]
    add_val = add_value_ref[:, :]
    scale_val = scale_ref[:, :]
    
    x = x + bias
    softplus = jnp.log(1.0 + jnp.exp(x))
    x = x * jnp.tanh(softplus)
    x = x + add_val
    x = jnp.clip(x, -1.0, 1.0)
    x = x * scale_val
    
    o_ref[:, :] = x

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        self.weight = jnp.zeros((in_channels, out_channels, kernel_size, kernel_size))
        self.bias = jnp.zeros(out_channels)
        self.add_value = add_value
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # ConvTranspose2d
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))  # (in,out,H,W) -> (H,W,out,in)
        
        # Calculate padding
        pad_h = self.kernel_size - 1 - self.padding
        pad_w = self.kernel_size - 1 - self.padding
        conv_padding = ((pad_h, pad_h + self.output_padding), 
                        (pad_w, pad_w + self.output_padding))
        
        x_nhwc = jax.lax.conv_transpose(
            x_nhwc, kernel,
            strides=(self.stride, self.stride),
            padding=conv_padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC')
        )
        
        N, H_out, W_out, C = x_nhwc.shape
        M = N * H_out * W_out
        x_flat = x_nhwc.reshape((M, C))
        bias_2d = self.bias.reshape((1, C))
        add_value_arr = jnp.array([[self.add_value]], dtype=x_flat.dtype)
        scale_arr = jnp.array([[self.scale]], dtype=x_flat.dtype)
        
        bm = 1024
        bc = 64
        while C > bc and bc < 256:
            bc *= 2
            
        pad_m = (bm - (M % bm)) % bm
        pad_c = (bc - (C % bc)) % bc
        
        if pad_m > 0 or pad_c > 0:
            x_flat = jnp.pad(x_flat, ((0, pad_m), (0, pad_c)))
            bias_2d = jnp.pad(bias_2d, ((0, 0), (0, pad_c)))
            
        grid_shape = (x_flat.shape[0] // bm, x_flat.shape[1] // bc)
        
        out_flat = pl.pallas_call(
            elementwise_kernel,
            out_shape=jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((bm, bc), lambda i, j: (i, j)),
                    pl.BlockSpec((1, bc), lambda i, j: (0, j)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec((bm, bc), lambda i, j: (i, j)),
            )
        )(x_flat, bias_2d, add_value_arr, scale_arr)
        
        if pad_m > 0 or pad_c > 0:
            out_flat = out_flat[:M, :C]
            
        out_nhwc = out_flat.reshape((N, H_out, W_out, C))
        out_nchw = jnp.transpose(out_nhwc, (0, 3, 1, 2))
        
        return out_nchw

    @property
    def kernel_size(self):
        return self.weight.shape[2]

batch_size = 128
in_channels = 64
out_channels = 64
height = width = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_channels, height, width))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]
