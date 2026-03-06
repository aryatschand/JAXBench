"""Model configurations for end-to-end inference benchmarking.

Each config specifies the full model architecture and which kernels it uses.
`eval_layers` is a reduced layer count that fits on a single TPU v6e-1 (16 GB HBM)
while keeping kernel time proportions representative of the full model.
"""

LLAMA3_8B = {
    'name': 'llama3_8b',
    'display_name': 'Llama-3.1-8B',
    'model_type': 'llama3',
    'vocab_size': 128256,
    'd_model': 4096,
    'num_layers': 32,
    'eval_layers': 8,       # ~4 GB weights in bf16, fits v6e-1
    'num_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'ffn_dim': 14336,
    'rope_theta': 500_000.0,
    'rms_norm_eps': 1e-6,
    # Kernels used per transformer block (with call counts per block)
    'block_kernels': {
        'rmsnorm': 2,           # pre-attention + pre-MLP
        'rope': 1,              # applied to Q and K
        'gqa_attention': 1,
        'swiglu_mlp': 1,
    },
    # Additional per-forward kernels (not per-block)
    'forward_kernels': {
        'token_embed': 1,
        'rmsnorm': 1,           # final norm
    },
}

GLA_1_3B = {
    'name': 'gla_1_3b',
    'display_name': 'GLA-1.3B',
    'model_type': 'gla',
    'vocab_size': 50304,
    'd_model': 2048,
    'num_layers': 24,
    'eval_layers': 24,      # ~2.6 GB weights, fits v6e-1
    'num_heads': 16,
    'head_dim': 128,
    'gate_dim': 16,
    'ffn_dim': 5632,
    'rms_norm_eps': 1e-6,
    'block_kernels': {
        'rmsnorm': 2,
        'gated_linear_attention': 1,
        'swiglu_mlp': 1,
    },
    'forward_kernels': {
        'token_embed': 1,
        'rmsnorm': 1,
    },
}

MAMBA2_2_7B = {
    'name': 'mamba2_2_7b',
    'display_name': 'Mamba-2-2.7B',
    'model_type': 'mamba2',
    'vocab_size': 50304,
    'd_model': 2560,
    'num_layers': 64,
    'eval_layers': 32,      # ~2.7 GB weights, fits v6e-1
    'num_heads': 64,
    'head_dim': 80,         # d_inner / H = (2560*2) / 64 = 80
    'd_state': 128,
    'd_conv': 4,
    'expand': 2,            # d_inner = d_model * expand = 5120
    'rms_norm_eps': 1e-6,
    'block_kernels': {
        'rmsnorm': 1,
        'ssd_attention': 1,
    },
    'forward_kernels': {
        'token_embed': 1,
        'rmsnorm': 1,
    },
}

ALL_MODELS = {
    'llama3_8b': LLAMA3_8B,
    'gla_1_3b': GLA_1_3B,
    'mamba2_2_7b': MAMBA2_2_7B,
}
