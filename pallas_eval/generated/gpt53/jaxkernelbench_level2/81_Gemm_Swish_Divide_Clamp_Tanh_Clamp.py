import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = jnp.zeros((in_features, out_features))
        if bias:
            self.bias = jnp.zeros((out_features,))
        else:
            self.bias = None

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        M, K = x.shape
        N = self.out_features

        bm = 128
        bn = 128
        bk = 128

        def kernel(x_ref, w_ref, b_ref, o_ref):
            acc = jnp.zeros((bm, bn), dtype=jnp.float32)

            def body(k, acc):
                k_start = k * bk
                x_block = x_ref[:, k_start:k_start + bk]
                w_block = w_ref[k_start:k_start + bk, :]
                acc = acc + jnp.dot(x_block, w_block, preferred_element_type=jnp.float32)
                return acc

            num_k = K // bk
            acc = jax.lax.fori_loop(0, num_k, body, acc)

            if b_ref is not None:
                acc = acc + b_ref[None, :]

            # Swish
            acc = acc * jax.nn.sigmoid(acc)
            acc = acc / 2.0
            acc = jnp.clip(acc, -1.0, 1.0)
            acc = jnp.tanh(acc)
            acc = jnp.clip(acc, -1.0, 1.0)

            o_ref[...] = acc.astype(o_ref.dtype)

        grid = (M // bm, N // bn)

        in_specs = [
            pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
            pl.BlockSpec((K, bn), lambda i, j: (0, j)),
        ]

        args = [x, self.weight]

        if self.bias is not None:
            in_specs.append(pl.BlockSpec((bn,), lambda i, j: (j,)))
            args.append(self.bias)
        else:
            args.append(None)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(*args)


batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features]
