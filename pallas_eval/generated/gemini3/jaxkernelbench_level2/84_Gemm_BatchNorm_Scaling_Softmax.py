import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

def epilogue_kernel(x_ref, w_eff_ref, b_eff_ref, out_ref):
    # Read blocks from HBM to VMEM
    x = x_ref[...]
    w = w_eff_ref[...]
    b = b_eff_ref[...]
    
    # Apply fused Bias + BatchNorm + Scale
    x_scaled = x * w + b
    
    # Apply Softmax over axis 1
    x_max = jnp.max(x_scaled, axis=1, keepdims=True)
    x_exp = jnp.exp(x_scaled - x_max)
    x_sum = jnp.sum(x_exp, axis=1, keepdims=True)
    out = x_exp / x_sum
    
    # Write result back to HBM
    out_ref[...] = out

class Model:
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        # Linear layer weights
        self.weight = jnp.zeros((in_features, out_features))
        self.bias = jnp.zeros((out_features,))
        
        # BatchNorm parameters
        self.bn_scale = jnp.ones((out_features,))
        self.bn_bias = jnp.zeros((out_features,))
        self.bn_mean = jnp.zeros((out_features,))
        self.bn_var = jnp.ones((out_features,))
        self.bn_eps = bn_eps
        
        # Scale parameter
        self.scale = jnp.ones(scale_shape)

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            if name == 'gemm.weight':
                # Transpose weight for JAX convention
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
        # 1. GEMM (without bias)
        # Using preferred_element_type=jnp.float32 to ensure f32 accumulation
        x_gemm = jnp.matmul(x, self.weight, preferred_element_type=jnp.float32)
        
        # 2. Precompute effective weights and biases for Bias + BN + Scale
        # This collapses the affine transformations into a single w * x + b operation
        inv_std = 1.0 / jnp.sqrt(self.bn_var + self.bn_eps)
        W_eff = self.scale * self.bn_scale * inv_std
        B_eff = W_eff * (self.bias - self.bn_mean) + self.scale * self.bn_bias
        
        # Reshape to 2D as required by Pallas
        W_eff = W_eff.reshape(1, -1)
        B_eff = B_eff.reshape(1, -1)
        
        B_batch, N = x_gemm.shape
        
        # Choose block size for the batch dimension
        block_B = 16
        if B_batch % block_B != 0:
            for i in range(16, 0, -1):
                if B_batch % i == 0:
                    block_B = i
                    break
                    
        grid_shape = (B_batch // block_B,)
        
        # 3. Pallas kernel for Epilogue (Bias + BN + Scale + Softmax)
        out = pl.pallas_call(
            epilogue_kernel,
            out_shape=jax.ShapeDtypeStruct(x_gemm.shape, x_gemm.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid_shape,
                in_specs=[
                    pl.BlockSpec((block_B, N), lambda i: (i, 0)),
                    pl.BlockSpec((1, N), lambda i: (0, 0)),
                    pl.BlockSpec((1, N), lambda i: (0, 0)),
                ],
                out_specs=pl.BlockSpec((block_B, N), lambda i: (i, 0)),
            ),
        )(x_gemm, W_eff, B_eff)
        
        return out

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
