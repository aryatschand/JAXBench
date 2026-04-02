import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gemm_scale_kernel(x_ref, w_ref, b_ref, s_ref, o_ref):
    acc = jnp.zeros_like(o_ref[...], dtype=jnp.float32)

    def k_loop(k, acc):
        x_block = x_ref[:, :]
        w_block = w_ref[:, :]
        acc = acc + jnp.dot(x_block, w_block.T)
        return acc

    acc = jax.lax.fori_loop(0, 1, k_loop, acc)
    out = acc + b_ref[None, :]
    out = out * s_ref[None, :]
    o_ref[...] = out.astype(o_ref.dtype)

def bn_kernel(x_ref, mean_ref, var_ref, w_ref, b_ref, o_ref, eps):
    x = x_ref[...]
    mean = mean_ref[...]
    var = var_ref[...]
    w = w_ref[...]
    b = b_ref[...]
    y = (x - mean) / jnp.sqrt(var + eps) * w + b
    o_ref[...] = y

class Model:
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)
        self.gemm_weight = jax.random.normal(key1, (out_features, in_features))
        self.gemm_bias = jax.random.normal(key2, (out_features,))
        self.scale = jax.random.normal(key3, scale_shape)

        self.bn_weight = jnp.ones((out_features,))
        self.bn_bias = jnp.zeros((out_features,))

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            jax_name = name.replace('.', '_')
            if hasattr(self, jax_name):
                setattr(self, jax_name, jnp.array(value))

    def forward(self, x):
        B, K = x.shape
        N = self.out_features

        block_m = 128
        block_n = 128

        grid = (B // block_m, N // block_n)

        out = pl.pallas_call(
            gemm_scale_kernel,
            out_shape=jax.ShapeDtypeStruct((B, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((block_n, K), lambda i, j: (j, 0)),
                    pl.BlockSpec((block_n,), lambda i, j: (j,)),
                    pl.BlockSpec((block_n,), lambda i, j: (j,)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x, self.gemm_weight, self.gemm_bias, self.scale)

        mean = jnp.mean(out, axis=0, keepdims=True)
        var = jnp.mean((out - mean) ** 2, axis=0, keepdims=True)

        out2 = pl.pallas_call(
            lambda xr, mr, vr, wr, br, or_: bn_kernel(xr, mr, vr, wr, br, or_, self.eps),
            out_shape=jax.ShapeDtypeStruct((B, N), out.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((block_n,), lambda i, j: (j,)),
                    pl.BlockSpec((block_n,), lambda i, j: (j,)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(out, mean, var, self.bn_weight, self.bn_bias)

        return out2

batch_size = 16384
in_features = 4096
out_features = 4096
scale_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, shape=(batch_size, in_features))
    return [x]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
