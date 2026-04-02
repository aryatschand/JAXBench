```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_mish_mish_kernel(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[...]
    w = w_ref[...]
    b = b_ref[...]  # shape is (1, BN) due to 2D constraint
    
    # Matrix multiplication
    acc = jnp.dot(x, w, preferred_element_type=jnp.float32)
    
    # Add bias (broadcasted along the batch dimension)
    b_bcast = pltpu.repeat(b, acc.shape[0], axis=0)
    acc = acc + b_bcast
    
    # Mish activation 1
    acc = acc * jnp.tanh(jax.nn.softplus(acc))
    
    # Mish activation 2
    acc = acc * jnp.tanh(jax.nn.softplus(acc))
    
    o_ref[...] = acc

class Model:
    def __init__(self, in_features, out_features):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        BM, BN = 128, 128
        grid_shape = (x.shape[0] // BM, self.weight.shape[1] // BN)
        
        # TPU constraint: All tensors must be at least 2D
        bias_2d = self.bias.reshape(1, -1)
        
        return pl.pallas_call(
            matmul_mish_mish_kernel,
            out_shape=jax.ShapeDtypeStruct((x.shape[0], self.weight.shape[1]), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, x.shape[1]), lambda i, j: (i, 0)),
                    pl.BlockSpec((self.weight.shape[0], BN), lambda i, j: (0, j)),
                    pl.BlockSpec((1, BN), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(x, self.weight, bias
