import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        self.weight = jnp.zeros((out_features, in_features))
        self.bias = jnp.zeros(out_features)
        
        self.group_norm_weight = jnp.ones(out_features)
        self.group_norm_bias = jnp.zeros(out_features)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        x = jnp.matmul(x, self.weight.T) + self.bias

        N, C = x.shape
        G = self.num_groups
        group_size = C // G

        x = x.reshape(N, G, group_size)
        gn_w = self.group_norm_weight.reshape(G, group_size)
        gn_b = self.group_norm_bias.reshape(G, group_size)

        def kernel(x_ref, w_ref, b_ref, o_ref):
            vals = x_ref[0, 0, :]
            mean = jnp.mean(vals)
            var = jnp.var(vals)
            norm = (vals - mean) / jnp.sqrt(var + 1e-5)
            out = norm * w_ref[0, :] + b_ref[0, :]
            out = jnp.clip(out, self.hardtanh_min, self.hardtanh_max)
            o_ref[0, 0, :] = out

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(N, G),
                in_specs=[
                    pl.BlockSpec((1, 1, group_size), lambda i, j: (i, j, 0)),
                    pl.BlockSpec((1, group_size), lambda i, j: (j, 0)),
                    pl.BlockSpec((1, group_size), lambda i, j: (j, 0)),
                ],
                out_specs=pl.BlockSpec((1, 1, group_size), lambda i, j: (i, j, 0)),
            ),
        )(x, gn_w, gn_b)

        return out.reshape(N, C)


batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]
