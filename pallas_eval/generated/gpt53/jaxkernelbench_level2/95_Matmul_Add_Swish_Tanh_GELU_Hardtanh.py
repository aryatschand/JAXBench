import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def fused_kernel(x_ref, w_ref, b_ref, a_ref, o_ref):
    x = x_ref[:, :]            # (bm, K)
    w = w_ref[:, :]            # (K, bn)
    b = b_ref[:]               # (bn,)
    a = a_ref[:]               # (bn,)

    out = jnp.dot(x, w)        # (bm, bn)

    b_broadcast = pltpu.repeat(b[None, :], out.shape[0], axis=0)
    a_broadcast = pltpu.repeat(a[None, :], out.shape[0], axis=0)

    out = out + b_broadcast
    out = out + a_broadcast

    out = out * jax.nn.sigmoid(out)   # swish
    out = jnp.tanh(out)
    out = jax.nn.gelu(out)
    out = jnp.clip(out, -1.0, 1.0)

    o_ref[:, :] = out


class Model:
    def __init__(self, in_features, out_features, add_value_shape):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        self.add_value = jnp.zeros(add_value_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        bm = 128
        bn = 128

        grid = (x.shape[0] // bm, self.weight.shape[1] // bn)

        return pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((x.shape[0], self.weight.shape[1]), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, x.shape[1]), lambda i, j: (i, 0)),   # x
                    pl.BlockSpec((x.shape[1], bn), lambda i, j: (0, j)),   # w
                    pl.BlockSpec((bn,), lambda i, j: (j,)),                # bias
                    pl.BlockSpec((bn,), lambda i, j: (j,)),                # add_value
                ],
                out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            ),
        )(x, self.weight, self.bias, self.add_value)


batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]
