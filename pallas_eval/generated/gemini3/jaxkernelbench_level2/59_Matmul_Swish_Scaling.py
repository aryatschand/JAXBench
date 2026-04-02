import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, scaling_factor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        M = x.shape[0]
        K = x.shape[1]
        N = self.weight.shape[1]
        
        M_block = min(M, 128)
        N_block = min(N, 512)
        K_block = min(K, 512)
        
        grid_shape = (M // M_block, N // N_block)
        
        bias_2d = self.bias.reshape(1, N)
        scaling_factor = self.scaling_factor
        
        def kernel(x_ref, w_ref, b_ref, o_ref):
            acc = jnp.zeros((M_block, N_block), dtype=jnp.float32)
            
            def loop_body(i, acc):
                x_val = x_ref[:, i * K_block : (i + 1) * K_block]
                w_val = w_ref[i * K_block : (i + 1) * K_block, :]
                return acc + jnp.dot(x_val, w_val, preferred_element_type=jnp.float32)
            
            acc = jax.lax.fori_loop(0, K // K_block, loop_body, acc)
            
            b_tile = b_ref[...]
            b_tile_broadcast = pltpu.repeat(b_tile, M_block, axis=0)
            acc = acc + b_tile_broadcast
            
            # Swish activation
            sig = jax.nn.sigmoid(acc)
            acc = acc * sig
            
            # Scaling
            acc = acc * scaling_factor
            
            o_ref[...] = acc.astype(o_ref.dtype)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((M_block, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, N_block), lambda i, j: (0, j)),
                    pl.BlockSpec((1, N_block), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((M_block, N_block), lambda i, j: (i, j)),
            ),
        )(x, self.weight, bias_2d)

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
