import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def pallas_epilogue(x, bias, gn_w, gn_b, num_groups):
    B, F = x.shape
    B_M = 256
    
    # Pad batch dimension if not divisible by B_M
    pad_B = (B_M - (B % B_M)) % B_M
    if pad_B > 0:
        x = jnp.pad(x, ((0, pad_B), (0, 0)))
    
    grid_B = x.shape[0] // B_M
    group_size = F // num_groups
    
    def epilogue_kernel(x_ref, bias_ref, gn_w_ref, gn_b_ref, o_ref):
        x_val = x_ref[...]
        bias_val = bias_ref[...]
        gn_w_val = gn_w_ref[...]
        gn_b_val = gn_b_ref[...]
        
        # Swish activation
        x_val = jax.nn.sigmoid(x_val) * x_val
        
        # Add bias
        x_val = x_val + bias_val
        
        # Group Norm
        x_reshaped = x_val.reshape((B_M, num_groups, group_size))
        
        mean = jnp.sum(x_reshaped, axis=-1, keepdims=True) / float(group_size)
        var = jnp.sum((x_reshaped - mean) ** 2, axis=-1, keepdims=True) / float(group_size)
        
        x_norm = (x_reshaped - mean) / jnp.sqrt(var + 1e-5)
        
        x_norm = x_norm.reshape((B_M, F))
        
        # Group Norm weight and bias
        out = x_norm * gn_w_val + gn_b_val
        
        o_ref[...] = out

    # Reshape 1D parameters to 2D for Pallas block specs
    bias_2d = bias.reshape(1, F)
    gn_w_2d = gn_w.reshape(1, F)
    gn_b_2d = gn_b.reshape(1, F)
    
    out = pl.pallas_call(
        epilogue_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(grid_B,),
            in_specs=[
                pl.BlockSpec((B_M, F), lambda i: (i, 0)),
                pl.BlockSpec((1, F), lambda i: (0, 0)),
                pl.BlockSpec((1, F), lambda i: (0, 0)),
                pl.BlockSpec((1, F), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((B_M, F), lambda i: (i, 0)),
        ),
    )(x, bias_2d, gn_w_2d, gn_b_2d)
    
    if pad_B > 0:
        out = out[:B]
    return out

class Model:
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros(bias_shape)
        self.group_norm_weight = jnp.ones(out_features)
        self.group_norm_bias = jnp.zeros(out_features)
        self.num_groups = num_groups
        self.out_features = out_features

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer (Matmul)
        x = jnp.matmul(x, self.weight)
        
        # Fused Swish, Bias, and GroupNorm in Pallas
        x = pallas_epilogue(
            x, 
            self.bias, 
            self.group_norm_weight, 
            self.group_norm_bias, 
            self.num_groups
        )
        
        return x

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    key = jax.random.PRNGKey(0)
    return [jax.random.uniform(key, shape=(batch_size, in_features))]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
