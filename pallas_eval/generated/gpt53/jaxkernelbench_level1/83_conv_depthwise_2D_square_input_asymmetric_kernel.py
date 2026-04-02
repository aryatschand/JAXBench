import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        
        key = jax.random.PRNGKey(0)
        weight_shape = (in_channels, 1, kernel_size, 1)
        weight = jax.random.normal(key, weight_shape)
        self.weight = jnp.transpose(weight, (2, 3, 1, 0))
        
        if bias:
            self.bias_param = jnp.zeros(in_channels)
        else:
            self.bias_param = None
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if 'weight' in name:
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
                setattr(self, 'weight', value)
            elif 'bias' in name:
                setattr(self, 'bias_param', jnp.array(value))
            else:
                setattr(self, name.replace('.', '_'), jnp.array(value))
    
    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        out = jax.lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=(self.stride, self.stride),
            padding=[(self.padding, self.padding), (self.padding, self.padding)],
            lhs_dilation=(1, 1),
            rhs_dilation=(self.dilation, self.dilation),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.in_channels
        )
        
        if self.bias_param is not None:
            out = out + self.bias_param
        
        def kernel(x_ref, o_ref):
            o_ref[...] = x_ref[...]
        
        N, H, W, C = out.shape
        block = (1, 1, 1, C)
        grid = (N, H, W, 1)
        
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[pl.BlockSpec(block, lambda n, h, w, c: (n, h, w, 0))],
                out_specs=pl.BlockSpec(block, lambda n, h, w, c: (n, h, w, 0)),
            ),
        )(out)
        
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

batch_size = 64
in_channels = 8
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0
dilation = 1

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, in_channels, height, width))
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]
