import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        
        weight_shape = (in_channels, 1, kernel_size, kernel_size)
        key = jax.random.PRNGKey(0)
        weight = jax.random.normal(key, weight_shape) * 0.02
        self.conv2d_weight = jnp.transpose(weight, (2, 3, 1, 0))
        
        if bias:
            self.conv2d_bias = jnp.zeros(in_channels)
        else:
            self.conv2d_bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                value = jnp.transpose(value, (2, 3, 1, 0))
            setattr(self, name.replace('.', '_'), jnp.array(value))
            
    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # NHWC
        
        N, H, W, C = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        
        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1
        
        x_pad = jnp.pad(x, ((0,0),(P,P),(P,P),(0,0)))
        
        BH = 128
        BW = 128
        
        def kernel(x_ref, w_ref, b_ref, o_ref):
            n = pl.program_id(0)
            h = pl.program_id(1)
            w = pl.program_id(2)
            
            h_start = h * BH
            w_start = w * BW
            
            x_block = x_ref[n, h_start:h_start+BH+K-1, w_start:w_start+BW+K-1, :]
            w_val = w_ref[:, :, :, :]  # K,K,1,C
            
            out = jnp.zeros((BH, BW, C), dtype=jnp.float32)
            
            for kh in range(K):
                for kw in range(K):
                    x_slice = x_block[kh:kh+BH, kw:kw+BW, :]
                    w_slice = w_val[kh, kw, 0, :]
                    out += x_slice * w_slice
            
            if b_ref is not None:
                out += b_ref[:]
            
            o_ref[n, h_start:h_start+BH, w_start:w_start+BW, :] = out
        
        grid = (
            N,
            H_out // BH,
            W_out // BW,
        )
        
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, H_out, W_out, C), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((1, H + 2*P, W + 2*P, C), lambda n, i, j: (n, 0, 0, 0)),
                    pl.BlockSpec((K, K, 1, C), lambda n, i, j: (0, 0, 0, 0)),
                    pl.BlockSpec((C,), lambda n, i, j: (0,)),
                ],
                out_specs=pl.BlockSpec((1, BH, BW, C), lambda n, i, j: (n, i, j, 0)),
            ),
        )(x_pad, self.conv2d_weight, self.conv2d_bias if self.conv2d_bias is not None else jnp.zeros((C,), dtype=x.dtype))
        
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
