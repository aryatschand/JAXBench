import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_div_gelu_kernel(x_ref, w_ref, b_ref, div_ref, o_ref):
    BM = x_ref.shape[0]
    BN = w_ref.shape[1]
    BK = 512
    
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    
    def body(k, acc):
        x_block = x_ref[:, k*BK : (k+1)*BK]
        w_block = w_ref[k*BK : (k+1)*BK, :]
        return acc + jnp.dot(x_block, w_block, preferred_element_type=jnp.float32)
    
    acc = jax.lax.fori_loop(0, x_ref.shape[1] // BK, body, acc)
    
    bias = b_ref[...]
    acc = acc + bias
    
    div = div_ref[...]
    acc = acc / div
    
    acc = jnn.gelu(acc)
    
    o_ref[...] = acc.astype(o_ref.dtype)

class Model:
    def __init__(self, input_size, output_size, divisor):
        self.weight = jnp.zeros((input_size, output_size))
        self.bias = jnp.zeros(output_size)
        self.divisor = divisor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        b = self.bias.reshape(1, -1)
        div = jnp.asarray(self.divisor, dtype=x.dtype).reshape(1, 1)
        
        BM = 128
        BN = 512
        
        grid_shape = (x.shape[0] // BM, self.weight.shape[1] // BN)
        
        out = pl.pallas_call(
            matmul_div_gelu_kernel,
            out_shape=jax.ShapeDtypeStruct((x.shape[0], self.weight.shape[1]), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, self.weight.shape[0]), lambda i, j: (i, 0)),
                    pl.BlockSpec((self.weight.shape[0], BN), lambda i, j: (0, j)),
                    pl.BlockSpec((1, BN), lambda i, j: (0, j)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(x, self.weight, b, div)
        return out

batch_size = 1024
input_size = 8192 
output_size = 8192
divisor = 10.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, output_size, divisor]
