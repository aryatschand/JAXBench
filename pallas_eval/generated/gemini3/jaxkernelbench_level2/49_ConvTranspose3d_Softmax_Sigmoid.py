import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_kernel(x_ref, bias_ref, o_ref):
    x = x_ref[...]
    bias = bias_ref[...]
    
    # Broadcast bias to match x
    bias = jnp.reshape(bias, (1, bias.shape[0]))
    x = x + bias
    
    # Softmax along the channel dimension (axis 1)
    x_max = jnp.max(x, axis=1, keepdims=True)
    x_exp = jnp.exp(x - x_max)
    x_sum = jnp.sum(x_exp, axis=1, keepdims=True)
    x_softmax = x_exp / x_sum
    
    # Sigmoid
    x_sigmoid = 1.0 / (1.0 + jnp.exp(-x_softmax))
    
    o_ref[...] = x_sigmoid

class Model:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        k = kernel_size
        if isinstance(k, int):
            k = (k, k, k)
        self.kernel_size = k
        self.weight = jnp.zeros((in_channels, out_channels, k[0], k[1], k[2]))
        if bias:
            self.bias = jnp.zeros((out_channels,))
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.has_bias = bias

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Convert from NCDHW to NDHWC
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose kernel from (in, out, D, H, W) to (D, H, W, out, in)
        kernel = jnp.transpose(self.weight, (2, 3, 4, 1, 0))
        
        # Calculate padding
        pad_d = self.kernel_size[0] - 1 - self.padding[0]
        pad_h = self.kernel_size[1] - 1 - self.padding[1]
        pad_w = self.kernel_size[2] - 1 - self.padding[2]
        padding = ((pad_d, pad_d + self.output_padding[0]),
                  (pad_h, pad_h + self.output_padding[1]),
                  (pad_w, pad_w + self.output_padding[2]))

        # 3D Transposed Convolution
        x = jax.lax.conv_transpose(
            x, kernel,
            strides=self.stride,
            padding=padding,
            dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'))

        # Flatten spatial and batch dimensions to fuse bias, softmax, and sigmoid
        N, D_out, H_out, W_out, C = x.shape
        M = N * D_out * H_out * W_out
        x_flat = x.reshape((M, C))
        
        # Pad dimensions to multiples of 128 for optimal TPU VMEM usage
        pad_C = (128 - (C % 128)) % 128
        pad_M = (128 - (M % 128)) % 128
        
        if pad_M > 0:
            x_flat = jnp.pad(x_flat, ((0, pad_M), (0, 0)), constant_values=0.0)
        if pad_C > 0:
            # Pad channels with -inf so they don't affect the softmax sum
            x_padded = jnp.pad(x_flat, ((0, 0), (0, pad_C)), constant_values=-jnp.inf)
        else:
            x_padded = x_flat
            
        if self.has_bias:
            if pad_C > 0:
                bias_padded = jnp.pad(self.bias, (0, pad_C), constant_values=-jnp.inf)
            else:
                bias_padded = self.bias
        else:
            bias_padded = jnp.full((C + pad_C,), -jnp.inf)
            bias_padded = bias_padded.at[:C].set(0.0)
            
        grid_M = (M + pad_M) // 128
        C_padded = C + pad_C
        
        # Execute fused Pallas kernel
        o_padded = pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct(x_padded.shape, x_padded.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid_M,),
                in_specs=[
                    pl.BlockSpec((128, C_padded), lambda i: (i, 0)),
                    pl.BlockSpec((C_padded,), lambda i: (0,))
                ],
                out_specs=pl.BlockSpec((128, C_padded), lambda i: (i, 0)),
            ),
        )(x_padded, bias_padded)
        
        # Slice away padding and reshape back to NDHWC
        o_flat = o_padded[:M, :C]
        o = o_flat.reshape((N, D_out, H_out, W_out, C))
        
        # Convert back from NDHWC to NCDHW
        o = jnp.transpose(o, (0, 4, 1, 2, 3))
        
        return o

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_channels, D, H, W))]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]
