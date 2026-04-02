import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def get_fused_kernel(group_size, bs_block, block_out):
    groups_per_block = block_out // group_size
    
    def fused_kernel(x_ref, gemm_b_ref, gn_w_ref, gn_b_ref, mul_w_ref, o_ref):
        # Load blocks from HBM to VMEM
        x_val = x_ref[...]
        gemm_b = gemm_b_ref[...]
        gn_w = gn_w_ref[...]
        gn_b = gn_b_ref[...]
        mul_w = mul_w_ref[...]
        
        # 1. Add GEMM bias
        x_val = x_val + gemm_b
        
        # 2. GroupNorm
        # Reshape to compute mean and variance over the group_size
        x_reshaped = x_val.reshape((bs_block, groups_per_block, group_size))
        mean = jnp.mean(x_reshaped, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x_reshaped - mean), axis=-1, keepdims=True)
        
        x_norm = (x_reshaped - mean) / jnp.sqrt(var + 1e-5)
        x_norm = x_norm.reshape((bs_block, block_out))
        
        # Apply affine transformation
        x_val = x_norm * gn_w + gn_b
        
        # 3. Swish
        x_val = x_val * jax.nn.sigmoid(x_val)
        
        # 4. Multiply
        x_val = x_val * mul_w
        
        # 5. Swish
        x_val = x_val * jax.nn.sigmoid(x_val)
        
        # Store result back to HBM
        o_ref[...] = x_val
        
    return fused_kernel

class Model:
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        # Initialize weights with same shapes as PyTorch
        self.gemm_weight = jnp.zeros((out_features, in_features))
        self.gemm_bias = jnp.zeros((out_features,))
        
        # GroupNorm parameters
        self.group_norm_weight = jnp.ones((out_features,))
        self.group_norm_bias = jnp.zeros((out_features,))
        self.num_groups = num_groups
        self.out_features = out_features
        
        # Multiply weight
        self.multiply_weight = jnp.zeros(multiply_weight_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer without bias (bias addition is fused into the Pallas kernel)
        x = jnp.matmul(x, self.gemm_weight.T)

        batch_size = x.shape[0]
        group_size = self.out_features // self.num_groups
        
        # Determine optimal block sizes for the grid
        bs_block = 1
        for b in [128, 64, 32, 16, 8, 4, 2, 1]:
            if batch_size % b == 0:
                bs_block = b
                break
                
        block_out = group_size
        for b in [256, 128, 64, 32]:
            if self.out_features % b == 0 and b % group_size == 0:
                block_out = b
                break
                
        grid_shape = (batch_size // bs_block, self.out_features // block_out)
        
        # Reshape 1D parameters to 2D as required by TPU Pallas
        gemm_b = self.gemm_bias.reshape(1, self.out_features)
        gn_w = self.group_norm_weight.reshape(1, self.out_features)
        gn_b = self.group_norm_bias.reshape(1, self.out_features)
        mul_w = self.multiply_weight.reshape(1, self.out_features)
        
        kernel = get_fused_kernel(group_size, bs_block, block_out)
        
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((bs_block, block_out), lambda i, j: (i, j)),
                    pl.BlockSpec((1, block_out), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_out), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_out), lambda i, j: (0, j)),
                    pl.BlockSpec((1, block_out), lambda i, j: (0, j)),
                ],
                out_specs=pl.BlockSpec((bs_block, block_out), lambda i, j: (i, j)),
            ),
        )(x, gemm_b, gn_w, gn_b, mul_w)
        
        return out

batch_size = 1024
in_features = 8192  
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
