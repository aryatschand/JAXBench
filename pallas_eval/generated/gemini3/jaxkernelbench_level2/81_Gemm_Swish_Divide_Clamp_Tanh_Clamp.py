import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = jnp.zeros((in_features, out_features))
        if bias:
            self.bias = jnp.zeros((out_features,))
        else:
            self.bias = None
            
    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        M = x.shape[0]
        K = x.shape[1]
        N = self.weight.shape[1]
        
        BM = 128
        BN = 128
        BK = 128
        
        grid_shape = (M // BM, N // BN)
        
        bias = self.bias if self.bias is not None else jnp.zeros((N,), dtype=x.dtype)
        
        def matmul_epilogue_kernel(x_ref, w_ref, b_ref, o_ref):
            acc = jnp.zeros((BM, BN), dtype=jnp.float32)
            
            def loop_body(i, acc_val):
                x_tile = x_ref[:, i*BK:(i+1)*BK]
                w_tile = w_ref[i*BK:(i+1)*BK, :]
                return acc_val + jnp.dot(x_tile, w_tile, preferred_element_type=jnp.float32)
                
            acc = jax.lax.fori_loop(0, K // BK, loop_body, acc)
            
            b = b_ref[:]
            acc = acc + b
            
            acc = acc * jax.nn.sigmoid(acc)
            acc = acc / 2.0
            acc = jnp.clip(acc, -1.0, 1.0)
            acc = jnp.tanh(acc)
            acc = jnp.clip(acc, -1.0, 1.0)
            
            o_ref[:, :] = acc.astype(x.dtype)

        return pl.pallas_call(
            matmul_epilogue_kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, BN), lambda i, j: (0, j)),
                    pl.BlockSpec((BN,), lambda i, j: (j,)),
                ],
                out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
            ),
        )(x, self.weight, bias)

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]
