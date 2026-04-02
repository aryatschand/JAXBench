import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gemm_add_relu_kernel(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[...]
    w = w_ref[...]
    b = b_ref[...]
    
    acc = jnp.dot(x, w, preferred_element_type=jnp.float32)
    
    b_broadcast = pltpu.repeat(b, x.shape[0], axis=0)
    acc = acc + b_broadcast
    
    out = jax.nn.relu(acc)
    
    o_ref[...] = out.astype(o_ref.dtype)

def pallas_gemm_add_relu(x, w, b):
    BM = min(x.shape[0], 128)
    BN = min(w.shape[1], 256)
    BK = x.shape[1]
    
    grid_shape = (x.shape[0] // BM, w.shape[1] // BN)
    
    return pl.pallas_call(
        gemm_add_relu_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], w.shape[1]), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid_shape,
            in_specs=[
                pl.BlockSpec((BM, BK), lambda i, j: (i, 0)),
                pl.BlockSpec((BK, BN), lambda i, j: (0, j)),
                pl.BlockSpec((1, BN), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
        ),
    )(x, w, b)

class Model:
    def __init__(self, in_features, out_features, bias_shape):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(bias_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        b = self.bias.reshape(1, -1)
        return pallas_gemm_add_relu(x, self.weight, b)

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bias_shape]
