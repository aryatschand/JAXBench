import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM = 512
BN = 512
BK = 512

def fused_matmul_kernel(x_ref, w_ref, b_ref, s_ref, o_ref):
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    
    def body_fn(k, acc):
        x_block = x_ref[:, k * BK : (k + 1) * BK]
        w_block = w_ref[k * BK : (k + 1) * BK, :]
        return acc + jnp.dot(x_block, w_block, preferred_element_type=jnp.float32)
    
    iters = x_ref.shape[1] // BK
    acc = jax.lax.fori_loop(0, iters, body_fn, acc)
    
    bias = b_ref[...]
    acc = acc + bias
    
    s = s_ref[0, 0]
    acc = acc * s
    
    o_ref[...] = acc.astype(o_ref.dtype)

class Model:
    def __init__(self, in_features, out_features, scaling_factor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        s_val = jnp.array([[self.scaling_factor + 1.0]], dtype=x.dtype)
        bias_2d = jnp.expand_dims(self.bias, 0)
        
        grid_shape = (x.shape[0] // BM, self.weight.shape[1] // BN)
        
        out = pl.pallas_call(
            fused_matmul_kernel,
            out_shape=jax.ShapeDtypeStruct((x.shape[0], self.weight.shape[1]), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, self.weight.shape[0]), lambda i, j: (i, 0)),
                    pl.BlockSpec((self.weight.shape[0], BN), lambda i, j: (0, j)),
                    pl.Block
