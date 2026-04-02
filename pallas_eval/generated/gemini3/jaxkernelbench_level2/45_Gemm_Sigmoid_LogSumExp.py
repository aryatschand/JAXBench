```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def logsumexp_kernel(x_ref, o_ref):
    # Load the block of shape (256, 1024)
    x = x_ref[...]
    
    # Compute max for numerical stability
    max_x = jnp.max(x, axis=1, keepdims=True)
    
    # Compute exp(x - max)
    exp_x = jnp.exp(x - max_x)
    
    # Compute sum(exp)
    sum_exp = jnp.sum(exp_x, axis=1, keepdims=True)
    
    # Compute log(sum) + max, resulting in shape (256, 1)
    res = max_x + jnp.log(sum_exp)
    
    # Repeat to fill the (256, 128) block to satisfy TPU block size constraints
    # (minor dimension must be a multiple of 128 for f32)
    o_ref[...] = pltpu.repeat(res, 128, axis=1)

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with same shapes as PyTorch
        self.linear1_weight = jnp.zeros((hidden_size, input_size))
        self.linear1_bias = jnp.zeros(hidden_size)
        self.linear2_weight = jnp.zeros((output_size, hidden_size))
        self.linear2_bias = jnp.zeros(output_size)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # First linear layer
        x = jnp.matmul(x, self.linear1_weight.T) + self.linear1_bias

        # Sigmoid activation
        x = jax.nn.sigmoid(x)

        # Second linear layer
        x = jnp.matmul(x, self.linear2_weight.T) + self.linear2_bias

        # LogSumExp over features using a fused Pallas kernel
        batch_size, features = x.shape
        block_size = 256
        grid_shape = (batch_size // block_size,)
        
        out_padded = pl.pallas_call(
            logsumexp_kernel,
            out_shape=jax.ShapeDtypeStruct((batch_size, 128), x.dtype),
            grid_spec=pltpu.PrefetchScalarGrid
