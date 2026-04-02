import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, w_ref, b_ref, gn_w_ref, gn_b_ref, o_ref):
    x = x_ref[0, :]  # (K,)
    w = w_ref[:, :]  # (K, N)
    b = b_ref[0, :]  # (N,)
    gn_w = gn_w_ref[0, :]
    gn_b = gn_b_ref[0, :]

    # GEMM (row * full weight)
    out = jnp.dot(x, w)  # (N,)

    # Bias
    out = out + b

    # Hardtanh
    out = jnp.clip(out, -1.0, 1.0)

    # Mish
    out = out * jnp.tanh(jax.nn.softplus(out))

    # GroupNorm
    N = out.shape[0]
    num_groups = 256
    group_size = N // num_groups

    xg = out.reshape(num_groups, group_size)
    mean = jnp.mean(xg, axis=1, keepdims=True)
    var = jnp.var(xg, axis=1, keepdims=True)
    xg = (xg - mean) / jnp.sqrt(var + 1e-5)
    out = xg.reshape(N)

    # Affine
    out = out * gn_w + gn_b

    o_ref[0, :] = out


class Model:
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((1, out_features))
        self.groupnorm_weight = jnp.ones((1, out_features))
        self.groupnorm_bias = jnp.zeros((1, out_features))
        self.num_groups = num_groups
        self.num_channels = out_features

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            arr = jnp.array(value)
            if name == "bias":
                arr = arr.reshape(1, -1)
            if name == "groupnorm.weight":
                arr = arr.reshape(1, -1)
            if name == "groupnorm.bias":
                arr = arr.reshape(1, -1)
            setattr(self, name.replace('.', '_'), arr)

    def forward(self, x):
        B, K = x.shape
        N = self.weight.shape[1]

        def run(x):
            return pl.pallas_call(
                fused_kernel,
                out_shape=jax.ShapeDtypeStruct((1, N), x.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(1,),
                    in_specs=[
                        pl.BlockSpec((1, K), lambda i: (i, 0)),
                        pl.BlockSpec((K, N), lambda i: (0, 0)),
                        pl.BlockSpec((1, N), lambda i: (0, 0)),
                        pl.BlockSpec((1, N), lambda i: (0, 0)),
                        pl.BlockSpec((1, N), lambda i: (0, 0)),
                    ],
                    out_specs=pl.BlockSpec((1, N), lambda i: (i, 0)),
                ),
            )(x, self.weight, self.bias, self.groupnorm_weight, self.groupnorm_bias)

        outputs = jax.vmap(run)(x)
        return outputs.reshape(B, N)


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]
