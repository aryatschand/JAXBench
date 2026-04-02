import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_kernel(kernel_size, stride, use_bias, H_out, W_out):
    def conv_kernel(x_ref, w_ref, b_ref, o_ref):
        x = x_ref[...]
        w = w_ref[...]
        
        n_block = x_ref.shape[0]
        c_block = x_ref.shape[1]
        
        h_size = (H_out - 1) * stride + 1
        w_size = (W_out - 1) * stride + 1
        
        def body_fn(i, acc):
            kh = i // kernel_size
            kw = i % kernel_size
            
            x_sub = lax.dynamic_slice(x, (0, 0, kh, kw), (n_block, c_block, h_size, w_size))
            if stride == 1:
                x_slice = x_sub
            else:
                x_slice = lax.slice(x_sub, (0, 0, 0, 0), (n_block, c_block, h_size, w_size), (1, 1, stride, stride))
                
            w_slice = lax.dynamic_slice(w, (0, kh, kw), (c_block, 1, 1))
            w_slice = w_slice.reshape(1, c_block, 1, 1)
            
            return acc + x_slice * w_slice

        acc = jnp.zeros((n_block, c_block, H_out, W_out), dtype=jnp.float32)
        acc = lax.fori_loop(0, kernel_size * kernel_size, body_fn, acc)
        
        if use_bias:
            b = b_ref[...].reshape(1, c_block, 1, 1)
            acc += b
            
        o_ref[...] = acc.astype(o_ref.dtype)
    return conv_kernel

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.groups = in_channels
        
        key = jax.random.PRNGKey(0)
        weight_shape = (kernel_size, kernel_size, 1, in_channels)
        self.weight = jax.random.normal(key, weight_shape) * 0.02
        
        if bias:
            self.bias_param = jnp.zeros((out_channels,))
        else:
            self.bias_param = None

    def set_weights(self, weights_dict):
        w = weights_dict['conv2d.weight']
        w = jnp.transpose(w, (2, 3, 1, 0))
        self.weight = jnp.array(w)
        
        if self.use_bias:
            self.bias_param = jnp.array(weights_dict['conv2d.bias'])

    def forward(self, x):
        N, C, H_in, W_in = x.shape
        
        H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        H_req = (H_out - 1) * self.stride + self.kernel_size
        W_req = (W_out - 1) * self.stride + self.kernel_size
        
        pad_h_left = self.padding
        pad_h_right = max(0, H_req - (H_in + self.padding))
        pad_w_left = self.padding
        pad_w_right = max(0, W_req - (W_in + self.padding))
        
        x_padded = jnp.pad(x, ((0, 0), (0, 0), (pad_h_left, pad_h_right), (pad_w_left, pad_w_right)))
        x_padded = x_padded[:, :, :H_req, :W_req]
        
        w = jnp.transpose(self.weight, (3, 0, 1, 2))
        w = w.reshape(C, self.kernel_size, self.kernel_size)
        
        if self.use_bias:
            b = self.bias_param
        else:
            b = jnp.zeros((C,), dtype=x.dtype)
            
        block_n = 1
        block_c = 1
        for b_val in [32, 16, 8, 4, 2, 1]:
            if C % b_val == 0:
                if b_val * H_req * W_req <= 1000000:
                    block_c = b_val
                    break
                    
        grid = (N // block_n, C // block_c)
        
        kernel = get_kernel(self.kernel_size, self.stride, self.use_bias, H_out, W_out)
        
        out_shape = jax.ShapeDtypeStruct((N, C, H_out, W_out), x.dtype)
        
        in_specs = [
            pl.BlockSpec((block_n, block_c, H_req, W_req), lambda i, j: (i, j, 0, 0)),
            pl.BlockSpec((block_c, self.kernel_size, self.kernel_size), lambda i, j: (j, 0, 0)),
            pl.BlockSpec((block_c,), lambda i, j: (j,))
        ]
            
        out_specs = pl.BlockSpec((block_n, block_c, H_out, W_out), lambda i, j: (i, j, 0, 0))
        
        out = pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
            )
        )(x_padded, w, b)
        
        return out

batch_size = 64
in_channels = 128  
out_channels = 128
kernel_size = 3
width_in = 512
height_in = 256
stride = 1
padding = 0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height_in, width_in))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
