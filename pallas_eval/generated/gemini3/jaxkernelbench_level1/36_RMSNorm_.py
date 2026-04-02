```python
import math
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

class Model:
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x: Input array of shape (batch_size, num_features, *).

        Returns:
            Output array with RMS Normalization applied, same shape as input.
        """
        original_shape = x.shape
        B = original_shape[0]
        F = original_shape[1]
        
        # Flatten all spatial dimensions into a single dimension S
        S = 1
        for d in original_shape[2:]:
            S *= d
            
        x_reshaped = x.reshape((B, F, S))
        
        # Pad F to the next power of 2 (minimum 8) to satisfy Pallas block shape constraints
        if F <= 8:
            F_padded = 8
        else:
            F_padded = 2 ** math.ceil(math.log2(F))
        pad_F = F_padded - F
        
        # Determine optimal block size for S dimension
        max_elements = 1048576  # 4 MB limit per block to comfortably fit in VMEM
        max_block_S = max(128, max_elements // F_padded)
        max_block_S = 2 ** int(math.floor(math.log2(max_block_S)))
        
        block_S = 128
        for size in [4096, 2048, 1024, 512, 256, 128]:
            if S >= size and size <= max_block_S:
                block_S = size
                break
                
        pad_S = (block_S - (S % block_S)) % block_S
        
        # Apply padding if necessary
        if pad_S > 0 or pad_F > 0:
            x_padded = jnp.pad(x_reshaped, ((0, 0), (0, pad_F), (0, pad_S)))
