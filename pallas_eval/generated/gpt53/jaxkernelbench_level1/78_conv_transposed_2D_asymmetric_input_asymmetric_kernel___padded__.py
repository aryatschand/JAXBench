import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def bias_add_kernel(x_ref, b_ref, o_ref):
    x = x_ref[...]
    b = b_ref[...]
    o_ref[...] = x + b

def pallas_bias_add(x, b):
    n, h, w, c = x.shape
    x2 = x.reshape(n * h, w * c)
    b2 = jnp.broadcast_to(b.reshape(1, c), (w, c)).reshape(1, w * c)

    block = (min(x2.shape[0], 128), min(x2.shape[1], 128))
    grid = (x2.shape[0] // block[0], x2.shape[1] // block[1])

    out2 = pl.pallas_call(
        bias_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                pl.BlockSpec(block, lambda i, j: (i, j)),
                pl.BlockSpec(block, lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec(block, lambda i, j: (i, j)),
        ),
    )(x2, b2)

    return out2.reshape(n, h, w, c)

class Model:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        key = jax.random.PRNGKey(0)
        weight_shape = (in_channels, out_channels, kernel_size[0], kernel_size[1])
        self.weight = jax.random.normal(key, weight_shape) * 0.02
        
        self.bias = None
        if bias:
            self.bias = jnp.zeros(out_channels)
            
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = ((kernel_size[0]-1-padding[0], kernel_size[0]-1-padding[0]),
                       (kernel_size[1]-1-padding[1], kernel_size[1]-1-padding[1]))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'conv_transpose2d.weight':
                value = jnp.transpose(jnp.array(value), (2, 3, 1, 0))
            elif name == 'conv_transpose2d.bias':
                value = jnp.array(value)
            setattr(self, name.replace('conv_transpose2d.', ''), jnp.array(value))

    def forward(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        out = lax.conv_transpose(
            x,
            self.weight,
            strides=self.stride,
            padding=self.padding,
            dimension_numbers=('NHWC', 'HWOI', 'NHWC'),
            transpose_kernel=True
        )

        if self.bias is not None:
            out = pallas_bias_add(out, self.bias)

        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (8, 32, 512, 1024))
    return [x]

def get_init_inputs():
    return [32, 32, (3, 7), (1, 1), (1, 3)]
