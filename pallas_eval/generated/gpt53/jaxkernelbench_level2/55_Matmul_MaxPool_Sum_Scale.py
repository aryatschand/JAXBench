import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        self.matmul_weight = jnp.zeros((out_features, in_features))
        self.matmul_bias = jnp.zeros((out_features, 1))
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == "matmul_bias":
                setattr(self, name.replace('.', '_'), jnp.array(value).reshape(-1, 1))
            else:
                setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        B, K = x.shape
        OF = self.matmul_weight.shape[0]
        ks = self.kernel_size

        x_2d = x
        w_2d = self.matmul_weight
        b_2d = self.matmul_bias

        def kernel(x_ref, w_ref, b_ref, o_ref):
            x_block = x_ref[...]          # (Br, K)
            w_full = w_ref[...]           # (OF, K)
            b_full = b_ref[...]           # (OF, 1)

            Br = x_block.shape[0]
            OF = w_full.shape[0]

            acc = jnp.zeros((Br,), dtype=x_block.dtype)

            num_groups = OF // ks

            def group_body(g, acc_val):
                maxv = jnp.full((Br,), -jnp.inf, dtype=x_block.dtype)

                def inner_body(t, max_val):
                    idx = g * ks + t
                    w_row = w_full[idx, :]
                    b_val = b_full[idx, 0]
                    dot = jnp.sum(x_block * w_row, axis=1) + b_val
                    return jnp.maximum(max_val, dot)

                maxv = jax.lax.fori_loop(0, ks, inner_body, maxv)
                return acc_val + maxv

            acc = jax.lax.fori_loop(0, num_groups, group_body, acc)
            o_ref[:, 0] = acc * self.scale_factor

        Br = min(B, 128)
        block = (Br, K)
        grid = (B // Br,)

        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, 1), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec(block, lambda i: (i, 0)),
                    pl.BlockSpec((OF, K), lambda i: (0, 0)),
                    pl.BlockSpec((OF, 1), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((Br, 1), lambda i: (i, 0)),
            ),
        )(x_2d, w_2d, b_2d)

        return jnp.squeeze(out, axis=1)


batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
