import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def forward_kernel(x_ref, w_ref, b_ref, sub_ref, mul_ref, o_ref):
    # Load blocks into VMEM
    x_val = x_ref[...]
    w_val = w_ref[...]
    
    # Matmul with f32 accumulator
    acc = jnp.dot(x_val, w_val, preferred_element_type=jnp.float32)
    
    # Add bias
    b_val = b_ref[...]
    acc = acc + b_val
    
    # Subtract
    sub_val = sub_ref[...]
    acc = acc - sub_val
    
    # Multiply
    mul_val = mul_ref[...]
    acc = acc * mul_val
    
    # ReLU
    acc = jax.nn.relu(acc)
    
    # Store result
    o_ref[...] = acc.astype(o_ref.dtype)

class Model:
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Reshape 1D tensors and scalars to 2D for Pallas
        b = self.bias.reshape(1, -1)
        sub = jnp.array([[self.subtract_value]], dtype=x.dtype)
        mul = jnp.array([[self.multiply_value]], dtype=x.dtype)
        
        # Block sizes
        BM = 128
        BN = 128
        
        grid_shape = (x.shape[0] // BM, self.weight.shape[1] // BN)
        
        return pl.pallas_call(
            forward_kernel,
            out_shape=jax.ShapeDtypeStruct((x.shape[0], self.weight.shape[1]), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, x.shape[1]), lambda i, j: (i, 0)),
                    pl.BlockSpec((self.weight.shape[0], BN), lambda i, j: (0, j)),
                    pl.BlockSpec((1, BN), lambda i, j: (0, j)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(x, self.weight, b, sub, mul)

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]
