import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Model:
    def __init__(self, in_features, out_features, dropout_p):
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

        self.matmul_weight = jnp.zeros((out_features, in_features))
        self.matmul_bias = jnp.zeros(out_features)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        def kernel(x_ref, w_ref, b_ref, o_ref):
            x_block = x_ref[:, :]
            w_block = w_ref[:, :]
            b_block = b_ref[:]

            # Matmul: x @ w.T
            out = jnp.dot(x_block, w_block.T) + b_block

            # Stable softmax along axis=1
            max_val = jnp.max(out, axis=1, keepdims=True)
            exp_out = jnp.exp(out - max_val)
            sum_exp = jnp.sum(exp_out, axis=1, keepdims=True)
            softmax_out = exp_out / sum_exp

            o_ref[:, :] = softmax_out

        batch, in_feat = x.shape
        out_feat = self.out_features

        block = (batch, out_feat)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((batch, out_feat), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(1,),
                in_specs=[
                    pl.BlockSpec((batch, in_feat), lambda i: (0, 0)),
                    pl.BlockSpec((out_feat, in_feat), lambda i: (0, 0)),
                    pl.BlockSpec((out_feat,), lambda i: (0,)),
                ],
                out_specs=pl.BlockSpec((batch, out_feat), lambda i: (0, 0)),
            ),
        )(x, self.matmul_weight, self.matmul_bias)


batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2


def get_inputs():
    key = random.PRNGKey(0)
    return [random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, dropout_p]
