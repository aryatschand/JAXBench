import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, orig_ref, o_ref):
    x = x_ref[0, :]            # (F,)
    orig = orig_ref[0, :]      # (F,)

    # mean over features
    mean_val = jnp.mean(x)

    # logsumexp over single value = itself
    lse = mean_val

    # GELU
    gelu = 0.5 * lse * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (lse + 0.044715 * lse**3)))

    # residual add (broadcast scalar)
    out = orig + gelu

    o_ref[0, :] = out


class Model:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = jnp.zeros((in_features, out_features))
        if bias:
            self.bias = jnp.zeros(out_features)
        else:
            self.bias = None
        self.subtract = jnp.zeros(out_features)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        original_x = x

        # GEMM (kept in JAX, heavy op)
        x = jnp.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias

        # Subtract
        x = x - self.subtract

        B, F = x.shape

        def run_kernel(x, original_x):
            return pl.pallas_call(
                fused_kernel,
                out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(B,),
                    in_specs=[
                        pl.BlockSpec((1, F), lambda i: (i, 0)),
                        pl.BlockSpec((1, F), lambda i: (i, 0)),
                    ],
                    out_specs=pl.BlockSpec((1, F), lambda i: (i, 0)),
                ),
            )(x, original_x)

        x = run_kernel(x, original_x)
        return x


batch_size = 2048
in_features = 8192
out_features = 8192


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features]
