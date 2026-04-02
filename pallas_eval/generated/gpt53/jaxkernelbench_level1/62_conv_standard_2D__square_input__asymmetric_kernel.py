import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        rng = jax.random.PRNGKey(0)
        k_h, k_w = kernel_size
        
        weight_shape = (out_channels, in_channels, k_h, k_w)
        weight = jax.random.normal(rng, weight_shape) * 0.02
        self.weight = jnp.transpose(weight, (2, 3, 1, 0))  # HWIO
        
        self.bias = None
        if bias:
            self.bias = jnp.zeros(out_channels)
            
        if isinstance(padding, int):
            self.padding = [(padding, padding), (padding, padding)]
        else:
            self.padding = [(p, p) for p in padding]
            
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.groups = groups

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
                setattr(self, 'weight', value)
            elif 'bias' in name:
                setattr(self, 'bias', jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # NHWC
        
        pad_h, pad_w = self.padding
        x = jnp.pad(x, ((0,0), pad_h, pad_w, (0,0)))
        
        N, H, W, C = x.shape
        KH, KW, IC, OC = self.weight.shape
        
        sh, sw = self.stride
        
        H_out = (H - (KH - 1) * self.dilation[0] - 1) // sh + 1
        W_out = (W - (KW - 1) * self.dilation[1] - 1) // sw + 1
        
        def kernel(x_ref, w_ref, o_ref):
            n = pl.program_id(0)
            h = pl.program_id(1)
            w = pl.program_id(2)
            oc = pl.program_id(3)
            
            acc = jnp.zeros((1,), dtype=jnp.float32)
            
            for kh in range(KH):
                for kw in range(KW):
                    ih = h * sh + kh * self.dilation[0]
                    iw = w * sw + kw * self.dilation[1]
                    
                    x_val = x_ref[n, ih, iw, :]
                    w_val = w_ref[kh, kw, :, oc]
                    
                    acc = acc + jnp.sum(x_val * w_val)
            
            if self.bias is not None:
                acc = acc + self.bias[oc]
            
            o_ref[n, h, w, oc] = acc.astype(o_ref.dtype)
        
        grid = (N, H_out, W_out, OC)
        
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((N, H_out, W_out, OC), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((1, H, W, C), lambda n, h, w, oc: (0, 0, 0, 0)),
                    pl.BlockSpec((KH, KW, IC, OC), lambda n, h, w, oc: (0, 0, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1, 1, 1), lambda n, h, w, oc: (n, h, w, oc)),
            ),
        )(x, self.weight)
        
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out


batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512

def get_inputs():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
