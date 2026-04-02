import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_fused_kernel(hardtanh_min, hardtanh_max):
    def kernel(x_ref, bias_ref, gn_w_ref, gn_b_ref, out_ref):
        x = x_ref[...]
        bias = bias_ref[...]
        gn_w = gn_w_ref[...]
        gn_b = gn_b_ref[...]
        
        # Add linear bias
        x = x + bias
        
        # Group Norm
        mean = jnp.sum(x, axis=2, keepdims=True) / x.shape[2]
        var = jnp.sum(jnp.square(x - mean), axis=2, keepdims=True) / x.shape[2]
        x = (x - mean) * jax.lax.rsqrt(var + 1e-5)
        
        # Scale and shift
        x = x * gn_w + gn_b
        
        # HardTanh
        x = jnp.clip(x, hardtanh_min, hardtanh_max)
        
        out_ref[...] = x
    return kernel

class Model:
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Initialize weights and bias for linear layer
        self.weight = jnp.zeros((out_features, in_features))
        self.bias = jnp.zeros(out_features)
        
        # Initialize group norm parameters
        self.group_norm_weight = jnp.ones(out_features)
        self.group_norm_bias = jnp.zeros(out_features)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer (matmul only, bias is fused into the Pallas kernel)
        x = jnp.matmul(x, self.weight.T)

        N, C = x.shape
        G = self.num_groups
        D = C // G
        
        # Reshape for group norm and kernel processing
        x_reshaped = x.reshape(N, G, D)
        bias_reshaped = self.bias.reshape(G, D)
        gn_w_reshaped = self.group_norm_weight.reshape(G, D)
        gn_b_reshaped = self.group_norm_bias.reshape(G, D)
        
        # Determine block size for N dimension
        block_n = 16
        while N % block_n != 0 and block_n > 1:
            block_n //= 2
            
        grid_shape = (N // block_n,)
        
        kernel = get_fused_kernel(self.hardtanh_min, self.hardtanh_max)
        
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_reshaped.shape, x_reshaped.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((block_n, G, D), lambda i: (i, 0, 0)),
                    pl.BlockSpec((G, D), lambda i: (0, 0)),
                    pl.BlockSpec((G, D), lambda i: (0, 0)),
                    pl.BlockSpec((G, D), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((block_n, G, D), lambda i: (i, 0, 0)),
            ),
        )(x_reshaped, bias_reshaped, gn_w_reshaped, gn_b_reshaped)
        
        # Reshape back to (N, C)
        return out.reshape(N, C)

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]
