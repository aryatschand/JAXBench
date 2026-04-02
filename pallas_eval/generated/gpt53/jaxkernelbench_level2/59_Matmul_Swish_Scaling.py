import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_swish_scale_kernel(x_ref, w_ref, b_ref, o_ref, scaling_factor):
    x = x_ref[:, :]              # (B, K)
    w = w_ref[:, :]              # (K, N_block)
    b = b_ref[:]                 # (N_block,)

    out = jnp.matmul(x, w) + b
    out = out * jax.nn.sigmoid(out)
    out = out * scaling_factor

    o_ref[:, :] = out


class Model:
    def __init__(self, in_features, out_features, scaling_factor):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(out_features)
        self.scaling_factor = scaling_factor

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        B, K = x.shape
        _, N = self.weight.shape

        block_m = B
        block_n = 128  # must divide N

        grid = (N // block_n,)

        def kernel(x_ref, w_ref, b_ref, o_ref):
            pid = pl.program_id(axis=0)

            w_block = w_ref[:, pid * block_n:(pid + 1) * block_n]
            b_block = b_ref[pid * block_n:(pid + 1) * block_n]

            matmul_swish_scale_kernel(
                x_ref,
                w_block,
                b_block,
                o_ref[:, pid * block_n:(pid + 1) * block_n],
                self.scaling_factor,
            )

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((B, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, K), lambda i: (0, 0)),      # x
                    pl.BlockSpec((K, block_n), lambda i: (0, i)),      # w
                    pl.BlockSpec((block_n,), lambda i: (i,)),          # b
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i: (0, i)),
            ),
        )(x, self.weight, self.bias)


batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]
