```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def add_bias_kernel(x_ref, bias_ref, o_ref):
    o_ref[...] = x_ref[...] + bias_ref[...]

def pallas_add_bias(x, bias):
    N, C, D, H, W = x.shape
    grid = (N, C, D)
    return pl.pallas_call(
        add_bias_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec((1, 1, 1, H, W), lambda n, c, d: (n, c, d, 0, 0)),
                pl.BlockSpec((1, 1, 1, 1, 1), lambda n, c, d: (0, c, 0, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, 1, 1, H, W), lambda n, c, d: (n, c, d, 0, 0)),
        )
    )(x, bias)

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        self.weight = jnp.zeros((in_channels, out_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        self.bias = jnp.zeros(out_channels) if bias else None
        self.stride = stride
        self.pytorch_padding = padding
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.groups = groups
        self.out_channels = out_channels
        
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        jax_padding = tuple(k - 1 - p for k, p in zip(self.kernel_size, self.pytorch_padding))
        
        if self.groups == 1:
            x_ndhwc = jnp.transpose(x, (0,
