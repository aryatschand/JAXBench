import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matvec_kernel(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[...]
    w = w_ref[...]
    b = b_ref[...]
    
    out = jnp.dot(x, w, preferred_element_type=jnp.float32) + b
    o_ref[...] = out

def forward_pallas(x, w, b):
    B_BLOCK = 128
    B_BLOCK = min(x.shape[0], B_BLOCK)
    grid_shape = (x.shape[0] // B_BLOCK,)
    
    return pl.pallas_call(
        matvec_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 1), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid_shape,
            in_specs=[
                pl.BlockSpec((B_BLOCK, x.shape[1]), lambda i: (i, 0)),
                pl.BlockSpec((x.shape[1], 1), lambda i: (0, 0)),
                pl.BlockSpec((1, 1), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((B_BLOCK, 1), lambda i: (i, 0)),
        )
    )(x, w, b)

class Model:
    def __init__(self, in_features, out_features):
        self.linear_weight = jnp.zeros((out_features, in_features))
        self.linear_bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        """
        Args:
            x: Input array of shape (batch_size, in_features)
        Returns:
            Array of shape (batch_size, 1)
        """
        # Mathematical simplification:
        # x @ W.T + b followed by sum(axis=1) is equivalent to:
        # x @ sum(W, axis=0) + sum(b)
        # The subsequent max, mean, and logsumexp over axis=1 on a shape of (batch_size, 1)
        # are all mathematical no-ops (identity functions).
        
        w_sum = jnp.sum(self.linear_weight, axis=0).reshape(-1, 1)
        b_sum = jnp.sum(self.linear_bias).reshape(1, 1)
        
        return forward_pallas(x, w_sum, b_sum)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]
