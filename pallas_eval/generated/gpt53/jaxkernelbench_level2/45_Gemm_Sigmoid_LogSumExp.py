import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gemm_sigmoid_kernel(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[:, :]
    w = w_ref[:, :]
    b = b_ref[:]

    acc = jnp.zeros((x.shape[0], w.shape[0]), dtype=jnp.float32)

    K = x.shape[1]
    tile_k = 128
    num_k = K // tile_k

    def body(k, acc):
        xk = x[:, k * tile_k:(k + 1) * tile_k]
        wk = w[:, k * tile_k:(k + 1) * tile_k]
        acc = acc + jnp.matmul(xk, wk.T)
        return acc

    acc = jax.lax.fori_loop(0, num_k, body, acc)
    acc = acc + b
    acc = 1 / (1 + jnp.exp(-acc))
    o_ref[:, :] = acc.astype(o_ref.dtype)


def gemm_kernel(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[:, :]
    w = w_ref[:, :]
    b = b_ref[:]

    acc = jnp.zeros((x.shape[0], w.shape[0]), dtype=jnp.float32)

    K = x.shape[1]
    tile_k = 128
    num_k = K // tile_k

    def body(k, acc):
        xk = x[:, k * tile_k:(k + 1) * tile_k]
        wk = w[:, k * tile_k:(k + 1) * tile_k]
        acc = acc + jnp.matmul(xk, wk.T)
        return acc

    acc = jax.lax.fori_loop(0, num_k, body, acc)
    acc = acc + b
    o_ref[:, :] = acc.astype(o_ref.dtype)


class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.linear1_weight = jnp.zeros((hidden_size, input_size))
        self.linear1_bias = jnp.zeros(hidden_size)
        self.linear2_weight = jnp.zeros((output_size, hidden_size))
        self.linear2_bias = jnp.zeros(output_size)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        B, I = x.shape
        H = self.linear1_weight.shape[0]
        O = self.linear2_weight.shape[0]

        block_b = 128
        block_h = 128
        block_o = 128

        grid1 = (B // block_b, H // block_h)
        grid2 = (B // block_b, O // block_o)

        def kernel1(x, w, b):
            return pl.pallas_call(
                gemm_sigmoid_kernel,
                out_shape=jax.ShapeDtypeStruct((B, H), x.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid1,
                    in_specs=[
                        pl.BlockSpec((block_b, I), lambda i, j: (i, 0)),
                        pl.BlockSpec((block_h, I), lambda i, j: (j, 0)),
                        pl.BlockSpec((block_h,), lambda i, j: (j,)),
                    ],
                    out_specs=pl.BlockSpec((block_b, block_h), lambda i, j: (i, j)),
                ),
            )(x, w, b)

        def kernel2(x, w, b):
            return pl.pallas_call(
                gemm_kernel,
                out_shape=jax.ShapeDtypeStruct((B, O), x.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=grid2,
                    in_specs=[
                        pl.BlockSpec((block_b, H), lambda i, j: (i, 0)),
                        pl.BlockSpec((block_o, H), lambda i, j: (j, 0)),
                        pl.BlockSpec((block_o,), lambda i, j: (j,)),
                    ],
                    out_specs=pl.BlockSpec((block_b, block_o), lambda i, j: (i, j)),
                ),
            )(x, w, b)

        x = kernel1(x, self.linear1_weight, self.linear1_bias)
        x = kernel2(x, self.linear2_weight, self.linear2_bias)
        x = jax.nn.logsumexp(x, axis=1)
        return x


batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, input_size))]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
