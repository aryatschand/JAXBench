import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        # Initialize weights with same shapes as PyTorch
        self.fc_weight = jnp.zeros((hidden_size, input_size))
        self.fc_bias = jnp.zeros(hidden_size)
        
        # Group norm parameters
        self.num_groups = num_groups
        self.num_channels = hidden_size
        self.eps = eps
        self.gn_weight = jnp.ones(hidden_size)
        self.gn_bias = jnp.zeros(hidden_size)
        
        self.negative_slope = negative_slope

    def set_weights(self, weights_dict):
        for name, value in weights_dict.items():
            setattr(self, name.replace('.', '_'), jnp.array(value))

    def forward(self, x):
        # Linear layer matmul (bias is fused into the Pallas kernel)
        x_matmul = jnp.matmul(x, self.fc_weight.T)
        
        N, C = x_matmul.shape
        group_size = C // self.num_groups
        
        # Reshape 1D parameters to 2D to satisfy Pallas constraints
        fc_bias_2d = self.fc_bias.reshape(1, C)
        gn_weight_2d = self.gn_weight.reshape(1, C)
        gn_bias_2d = self.gn_bias.reshape(1, C)
        
        # Block sizes for TPU (multiples of 8 and 128 for f32)
        N_block = min(N, 8)
        C_block = min(C, 512)
        
        grid_shape = (N // N_block, C // C_block)
        
        eps = self.eps
        negative_slope = self.negative_slope
        
        def epilogue_kernel(x_ref, fc_b_ref, gn_w_ref, gn_b_ref, o_ref):
            x_val = x_ref[...]
            fc_b_val = fc_b_ref[...]
            gn_w_val = gn_w_ref[...]
            gn_b_val = gn_b_ref[...]
            
            # Broadcast 1D parameters to 2D block shape
            fc_b_val = pltpu.repeat(fc_b_val, N_block, axis=0)
            gn_w_val = pltpu.repeat(gn_w_val
