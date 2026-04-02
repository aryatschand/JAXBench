import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def fused_kernel(x_ref, w_ref, lin_bias_ref, bn_scale_ref, bn_bias_ref,
                 bn_mean_ref, bn_var_ref, bias_ref, div_ref, o_ref):
    x = x_ref[:, :]                      # (B, K)
    w = w_ref[:, :]                      # (K, N)
    lin_bias = lin_bias_ref[0, :]        # (N,)
    bn_scale = bn_scale_ref[0, :]
    bn_bias = bn_bias_ref[0, :]
    bn_mean = bn_mean_ref[0, :]
    bn_var = bn_var_ref[0, :]
    bias = bias_ref[0, 0]
    divide_value = div_ref[0, 0]

    y = jnp.matmul(x, w) + lin_bias

    y = (y - bn_mean) / jnp.sqrt(bn_var + 1e-5)
    y = bn_scale * y + bn_bias

    y = y + bias
    y = y / divide_value

    y = y * jax.nn.sigmoid(y)

    o_ref[:, :] = y


class Model:
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        self.weight = jnp.zeros((in_features, out_features))
        self.linear_bias = jnp.zeros(out_features)

        self.bn_scale = jnp.ones(out_features)
        self.bn_bias = jnp.zeros(out_features)
        self.bn_mean = jnp.zeros(out_features)
        self.bn_var = jnp.ones(out_features)
        self.bn_eps = bn_eps

        self.bias = jnp.zeros(bias_shape)
        self.divide_value = divide_value

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'matmul.weight':
                setattr(self, 'weight', jnp.array(value.T))
            elif name == 'matmul.bias':
                setattr(self, 'linear_bias', jnp.array(value))
            elif name == 'bn.weight':
                setattr(self, 'bn_scale', jnp.array(value))
            elif name == 'bn.bias':
                setattr(self, 'bn_bias', jnp.array(value))
            elif name == 'bn.running_mean':
                setattr(self, 'bn_mean', jnp.array(value))
            elif name == 'bn.running_var':
                setattr(self, 'bn_var', jnp.array(value))
            elif name == 'bias':
                setattr(self, 'bias', jnp.array(value))

    def forward(self, x):
        B, K = x.shape
        N = self.weight.shape[1]

        block_m = 128
        block_n = 128

        grid = (B // block_m, N // block_n)

        x_2d = x
        w_2d = self.weight

        lin_bias = self.linear_bias.reshape(1, -1)
        bn_scale = self.bn_scale.reshape(1, -1)
        bn_bias = self.bn_bias.reshape(1, -1)
        bn_mean = self.bn_mean.reshape(1, -1)
        bn_var = self.bn_var.reshape(1, -1)
        bias = self.bias.reshape(1, 1)
        div = jnp.array(self.divide_value, dtype=x.dtype).reshape(1, 1)

        return pl.pallas_call(
            fused_kernel,
            out_shape=jax.ShapeDtypeStruct((B, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((block_m, K), lambda i, j: (i, 0)),
                    pl.BlockSpec((K, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_n), lambda i, j: (0, j)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                    pl.BlockSpec((1, 1), lambda i, j: (0, 0)),
                ],
                out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
            ),
        )(x_2d, w_2d, lin_bias, bn_scale, bn_bias, bn_mean, bn_var, bias, div)


batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0


def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, (batch_size, in_features))]


def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]
