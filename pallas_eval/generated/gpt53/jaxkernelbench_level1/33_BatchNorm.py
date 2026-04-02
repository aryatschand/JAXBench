import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.running_mean = jnp.zeros(num_features)
        self.running_var = jnp.ones(num_features)
        self.eps = 1e-5

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        N, C, H, W = x.shape

        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        x2d = jnp.reshape(x, (N * H * W, C))

        # reshape params to 2D
        weight = self.weight.reshape(1, C)
        bias = self.bias.reshape(1, C)
        mean = self.running_mean.reshape(1, C)
        var = self.running_var.reshape(1, C)

        def kernel(x_ref, w_ref, b_ref, m_ref, v_ref, o_ref):
            x_block = x_ref[...]
            w = w_ref[...]
            b = b_ref[...]
            m = m_ref[...]
            v = v_ref[...]

            w = pltpu.repeat(w, x_block.shape[0], axis=0)
            b = pltpu.repeat(b, x_block.shape[0], axis=0)
            m = pltpu.repeat(m, x_block.shape[0], axis=0)
            v = pltpu.repeat(v, x_block.shape[0], axis=0)

            y = (x_block - m) / jnp.sqrt(v + self.eps)
            y = y * w + b
            o_ref[...] = y

        B = 256
        Ct = C  # 64

        grid = (x2d.shape[0] // B, 1)

        out2d = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x2d.shape, x2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((B, Ct), lambda i, j: (i, j)),
                    pl.BlockSpec((1, Ct), lambda i, j: (0, j)),
                    pl.BlockSpec((1, Ct), lambda i, j: (0, j)),
                    pl.BlockSpec((1, Ct), lambda i, j: (0, j)),
                    pl.BlockSpec((1, Ct), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((B, Ct), lambda i, j: (i, j)),
            ),
        )(x2d, weight, bias, mean, var)

        out = jnp.reshape(out2d, (N, H, W, C))
        out = jnp.transpose(out, (0, 3, 1, 2))
        return out

batch_size = 64
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (batch_size, features, dim1, dim2))
    return [x]

def get_init_inputs():
    return [features]
