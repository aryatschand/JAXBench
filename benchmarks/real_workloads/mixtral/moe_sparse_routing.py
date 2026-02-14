"""
Sparse Mixture of Experts (MoE) for Mixtral

Extracted from MaxText. Mixtral uses sparse top-k routing where
each token is processed by only k experts out of n total.

Mixtral configs:
- 8x7B: 8 experts, top-2 routing, mlp_dim=14336
- 8x22B: 8 experts, top-2 routing, mlp_dim=16384

Key components:
1. Router: Linear layer that produces expert scores
2. Top-k selection: Choose k experts per token
3. Expert computation: Run token through selected experts
4. Combine: Weighted sum of expert outputs
"""

import jax
import jax.numpy as jnp
from functools import partial

# Problem sizes from real Mixtral models
MIXTRAL_8x7B = {
    'name': 'Mixtral-8x7B',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 4096,
    'mlp_dim': 14336,
    'num_experts': 8,
    'num_experts_per_tok': 2,
}

MIXTRAL_8x22B = {
    'name': 'Mixtral-8x22B',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 6144,
    'mlp_dim': 16384,
    'num_experts': 8,
    'num_experts_per_tok': 2,
}

# Batch inference
MIXTRAL_8x7B_BATCH8 = {
    'name': 'Mixtral-8x7B-B8',
    'batch': 8,
    'seq_len': 512,
    'emb_dim': 4096,
    'mlp_dim': 14336,
    'num_experts': 8,
    'num_experts_per_tok': 2,
}


def router_forward(
    x: jnp.ndarray,
    router_weights: jnp.ndarray,
    num_experts_per_tok: int,
) -> tuple:
    """
    MoE router that selects top-k experts per token.

    Args:
        x: [batch, seq_len, emb_dim] input hidden states
        router_weights: [emb_dim, num_experts] routing projection
        num_experts_per_tok: Number of experts to route each token to

    Returns:
        (router_probs, selected_experts)
        - router_probs: [batch, seq_len, num_experts_per_tok] softmax weights
        - selected_experts: [batch, seq_len, num_experts_per_tok] expert indices
    """
    # Compute router logits
    router_logits = jnp.dot(x, router_weights)  # [batch, seq, num_experts]

    # Get top-k experts
    top_k_logits, top_k_indices = jax.lax.top_k(router_logits, num_experts_per_tok)

    # Softmax over selected experts only
    router_probs = jax.nn.softmax(top_k_logits, axis=-1)

    return router_probs, top_k_indices


def expert_mlp(
    x: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    up_kernel: jnp.ndarray,
    down_kernel: jnp.ndarray,
) -> jnp.ndarray:
    """
    Single expert MLP (same as SwiGLU).

    Args:
        x: [tokens, emb_dim]
        gate_kernel: [emb_dim, mlp_dim]
        up_kernel: [emb_dim, mlp_dim]
        down_kernel: [mlp_dim, emb_dim]

    Returns:
        output: [tokens, emb_dim]
    """
    gate = jax.nn.silu(jnp.dot(x, gate_kernel))
    up = jnp.dot(x, up_kernel)
    hidden = gate * up
    output = jnp.dot(hidden, down_kernel)
    return output


def sparse_moe_forward(
    x: jnp.ndarray,
    router_weights: jnp.ndarray,
    expert_gate_kernels: jnp.ndarray,
    expert_up_kernels: jnp.ndarray,
    expert_down_kernels: jnp.ndarray,
    num_experts_per_tok: int,
) -> tuple:
    """
    Sparse MoE forward pass as used in Mixtral.

    Each token is routed to top-k experts, computed through those experts,
    and the outputs are combined weighted by routing probabilities.

    Args:
        x: [batch, seq_len, emb_dim]
        router_weights: [emb_dim, num_experts]
        expert_gate_kernels: [num_experts, emb_dim, mlp_dim]
        expert_up_kernels: [num_experts, emb_dim, mlp_dim]
        expert_down_kernels: [num_experts, mlp_dim, emb_dim]
        num_experts_per_tok: Number of experts per token

    Returns:
        (output, router_probs) where output is [batch, seq_len, emb_dim]
    """
    batch, seq_len, emb_dim = x.shape
    num_experts = router_weights.shape[-1]

    # Routing
    router_probs, selected_experts = router_forward(x, router_weights, num_experts_per_tok)

    # Flatten batch and seq for easier processing
    x_flat = x.reshape(-1, emb_dim)  # [batch*seq, emb_dim]
    num_tokens = x_flat.shape[0]

    # For each token, compute expert outputs and combine
    # This is the naive implementation - production uses more efficient kernels
    output = jnp.zeros_like(x_flat)

    # Process each expert
    for expert_idx in range(num_experts):
        # Create mask for tokens routed to this expert
        # selected_experts: [batch, seq, k] -> [batch*seq, k]
        selected_flat = selected_experts.reshape(num_tokens, num_experts_per_tok)

        # For each position in top-k
        for k in range(num_experts_per_tok):
            mask = (selected_flat[:, k] == expert_idx)  # [batch*seq]
            routing_weight = router_probs.reshape(num_tokens, num_experts_per_tok)[:, k]

            if jnp.any(mask):
                # Get expert weights
                gate_k = expert_gate_kernels[expert_idx]
                up_k = expert_up_kernels[expert_idx]
                down_k = expert_down_kernels[expert_idx]

                # Compute expert output for all tokens (masked after)
                expert_out = expert_mlp(x_flat, gate_k, up_k, down_k)

                # Add weighted contribution
                contribution = expert_out * routing_weight[:, None] * mask[:, None]
                output = output + contribution

    output = output.reshape(batch, seq_len, emb_dim)
    return output, router_probs


def sparse_moe_einsum(
    x: jnp.ndarray,
    router_weights: jnp.ndarray,
    expert_gate_kernels: jnp.ndarray,
    expert_up_kernels: jnp.ndarray,
    expert_down_kernels: jnp.ndarray,
    num_experts_per_tok: int,
) -> tuple:
    """
    Efficient sparse MoE using einsum (batched expert computation).

    This version computes all experts in parallel and masks the output,
    which is more efficient on accelerators.

    Args:
        x: [batch, seq_len, emb_dim]
        router_weights: [emb_dim, num_experts]
        expert_gate_kernels: [num_experts, emb_dim, mlp_dim]
        expert_up_kernels: [num_experts, emb_dim, mlp_dim]
        expert_down_kernels: [num_experts, mlp_dim, emb_dim]
        num_experts_per_tok: Number of experts per token

    Returns:
        (output, aux_loss) where output is [batch, seq_len, emb_dim]
    """
    batch, seq_len, emb_dim = x.shape
    num_experts = router_weights.shape[-1]

    # Routing
    router_probs, selected_experts = router_forward(x, router_weights, num_experts_per_tok)

    # Compute all experts in parallel using einsum
    # x: [batch, seq, emb] -> need [batch, seq, experts, mlp] intermediate

    # Gate projection for all experts
    # x: [b, s, e], expert_gate: [experts, e, m] -> [b, s, experts, m]
    gate_out = jnp.einsum('bse,nem->bsnm', x, expert_gate_kernels)
    gate_out = jax.nn.silu(gate_out)

    # Up projection
    up_out = jnp.einsum('bse,nem->bsnm', x, expert_up_kernels)

    # Gating
    hidden = gate_out * up_out

    # Down projection
    expert_outputs = jnp.einsum('bsnm,nme->bsne', hidden, expert_down_kernels)
    # expert_outputs: [batch, seq, num_experts, emb]

    # Create one-hot for selected experts and weight by routing probs
    # selected_experts: [batch, seq, k]
    one_hot = jax.nn.one_hot(selected_experts, num_experts)  # [b, s, k, n]

    # Weight by routing probabilities
    # router_probs: [b, s, k]
    weighted_one_hot = one_hot * router_probs[..., None]  # [b, s, k, n]

    # Sum over k to get final weights per expert
    expert_weights = weighted_one_hot.sum(axis=2)  # [b, s, n]

    # Combine expert outputs
    output = jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights)

    # Compute auxiliary load balancing loss
    # Fraction of tokens routed to each expert
    tokens_per_expert = one_hot.sum(axis=(0, 1, 2)) / (batch * seq_len * num_experts_per_tok)
    # Routing probability mass per expert (sum over all positions and k slots)
    # router_probs: [b, s, k], weighted_one_hot: [b, s, k, n]
    prob_per_expert = expert_weights.sum(axis=(0, 1)) / (batch * seq_len)
    # This would be used for load balancing loss
    aux_loss = num_experts * (tokens_per_expert * prob_per_expert).sum()

    return output, aux_loss


def create_inputs(config: dict, dtype=jnp.bfloat16):
    """Create random inputs matching config."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)

    batch = config['batch']
    seq_len = config['seq_len']
    emb_dim = config['emb_dim']
    mlp_dim = config['mlp_dim']
    num_experts = config['num_experts']

    x = jax.random.normal(keys[0], (batch, seq_len, emb_dim), dtype=dtype)
    router_weights = jax.random.normal(keys[1], (emb_dim, num_experts), dtype=dtype) * 0.02
    expert_gate_kernels = jax.random.normal(keys[2], (num_experts, emb_dim, mlp_dim), dtype=dtype) * 0.02
    expert_up_kernels = jax.random.normal(keys[3], (num_experts, emb_dim, mlp_dim), dtype=dtype) * 0.02
    expert_down_kernels = jax.random.normal(keys[4], (num_experts, mlp_dim, emb_dim), dtype=dtype) * 0.02

    return x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels


def benchmark_sparse_moe(config: dict, num_warmup: int = 5, num_iters: int = 50):
    """Benchmark sparse MoE."""
    import time

    x, router_weights, gate_k, up_k, down_k = create_inputs(config)

    moe_fn = jax.jit(partial(
        sparse_moe_einsum,
        num_experts_per_tok=config['num_experts_per_tok'],
    ))

    # Warmup
    for _ in range(num_warmup):
        output, aux_loss = moe_fn(x, router_weights, gate_k, up_k, down_k)
        output.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output, aux_loss = moe_fn(x, router_weights, gate_k, up_k, down_k)
        output.block_until_ready()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iters * 1000

    # Calculate FLOPS
    # For sparse MoE, only num_experts_per_tok experts are activated per token
    batch = config['batch']
    seq_len = config['seq_len']
    emb_dim = config['emb_dim']
    mlp_dim = config['mlp_dim']
    k = config['num_experts_per_tok']

    # Routing: emb * num_experts
    routing_flops = batch * seq_len * emb_dim * config['num_experts'] * 2
    # Per-token expert computation (k experts): 3 matmuls * k
    expert_flops = batch * seq_len * k * (emb_dim * mlp_dim * 2 * 3)
    total_flops = routing_flops + expert_flops
    tflops = total_flops / (avg_time_ms / 1000) / 1e12

    return {
        'config': config['name'],
        'time_ms': avg_time_ms,
        'tflops': tflops,
        'shape': list(output.shape),
        'aux_loss': float(aux_loss),
    }


if __name__ == '__main__':
    print("=" * 80)
    print("MIXTRAL SPARSE MoE BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    configs = [MIXTRAL_8x7B, MIXTRAL_8x22B, MIXTRAL_8x7B_BATCH8]

    print(f"{'Config':<25} | {'Time (ms)':>10} | {'TFLOPS':>8} | {'Aux Loss':>10} | Output Shape")
    print("-" * 90)

    for config in configs:
        result = benchmark_sparse_moe(config)
        print(f"{result['config']:<25} | {result['time_ms']:>10.2f} | {result['tflops']:>8.1f} | {result['aux_loss']:>10.4f} | {result['shape']}")
