import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        
        self.bn_scale = jnp.ones((out_features,))
        self.bn_bias = jnp.zeros((out_features,))
        self.bn_mean = jnp.zeros((out_features,))
        self.bn_var = jnp.ones((out_features,))
        self.bn_eps = bn_eps
        
        self.scale = jnp.ones(scale_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'gemm.weight':
                setattr(self, 'weight', jnp.array(value.T))
            elif name == 'gemm.bias':
                setattr(self, 'bias', jnp.array(value))
            elif name == 'bn.weight':
                setattr(self, 'bn_scale', jnp.array(value))
            elif name == 'bn.bias':
                setattr(self, 'bn_bias', jnp.array(value))
            elif name == 'bn.running_mean':
                setattr(self, 'bn_mean', jnp.array(value))
            elif name == 'bn.running_var':
                setattr(self, 'bn_var', jnp.array(value))
            elif name == 'scale':
                setattr(self, 'scale', jnp.array(value))

    def forward(self, x):
        M, K = x.shape
        N = self.weight.shape[1]

        bm = 128  # tile rows

        def kernel(x_ref, w_ref, b_ref, bn_s_ref, bn_b_ref, bn_m_ref, bn_v_ref, scale_ref, o_ref):
            x_block = x_ref[:, :]                 # (bm, K)
            w = w_ref[:, :]                      # (K, N)
            b = b_ref[0, :]                      # (N,)
            bn_s = bn_s_ref[0, :]
            bn_b = bn_b_ref[0, :]
            bn_m = bn_m_ref[0, :]
            bn_v = bn_v_ref[0, :]
            scale = scale_ref[0, 0]

            y = jnp.matmul(x_block, w) + b

            y = (y - bn_m) / jnp.sqrt(bn_v + self.bn_eps)
            y = bn_s * y + bn_b

            y = scale * y

            y = jax.nn.softmax(y, axis=1)

            o_ref[:, :] = y

        grid = (M // bm,)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=[
                    pl.BlockSpec((bm, K), lambda i: (i, 0)),     # x
                    pl.BlockSpec((K, N), lambda i: (0, 0)),      # weight
                    pl.BlockSpec((1, N), lambda i: (0, 0)),      # bias
                    pl.BlockSpec((1, N), lambda i: (0, 0)),      # bn_scale
                    pl.BlockSpec((1, N), lambda i: (0, 0)),      # bn_bias
                    pl.BlockSpec((1, N), lambda i: (0, 0)),      # bn_mean
                    pl.BlockSpec((1, N), lambda i: (0, 0)),      # bn_var
                    pl.BlockSpec((1, 1), lambda i: (0, 0)),      # scale
                ],
                out_specs=pl.BlockSpec((bm, N), lambda i: (i, 0)),
            ),
        )(
            x,
            self.weight,
            self.bias.reshape(1, -1),
            self.bn_scale.reshape(1, -1),
            self.bn_bias.reshape(1, -1),
            self.bn_mean.reshape(1, -1),
            self.bn_var.reshape(1, -1),
            self.scale.reshape(1, -1)[:, :1],
        )

batch_size = 1024
in_features = 8192  
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]
