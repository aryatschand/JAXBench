"""
Mixture of Experts with Shared Experts for DeepSeek V3

Extracted from MaxText. DeepSeek V3 uses a novel MoE architecture with:
1. Shared experts: Always-active experts that process every token
2. Routed experts: Sparse top-k routing like Mixtral
3. Bias-based routing: Uses learnable bias instead of load balancing loss

Key innovations:
- routed_scaling_factor: Scales routed expert outputs (2.5 in V3)
- routed_score_func: sigmoid (instead of softmax)
- routed_bias: Learnable bias for load balancing

DeepSeek V3 config (671B):
- num_experts: 256
- num_experts_per_tok: 8
- shared_experts: 1
- routed_scaling_factor: 2.5
"""

import jax
import jax.numpy as jnp
from functools import partial

# Problem sizes from DeepSeek V3
DEEPSEEK_V3 = {
    'name': 'DeepSeek-V3-671B',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 7168,
    'mlp_dim': 2048,  # Per-expert MLP dim (smaller due to many experts)
    'num_experts': 256,
    'num_experts_per_tok': 8,
    'shared_experts': 1,
    'shared_mlp_dim': 18432,  # Full MLP dim for shared expert
    'routed_scaling_factor': 2.5,
}

# Smaller test config
DEEPSEEK_V3_SMALL = {
    'name': 'DeepSeek-V3-Small',
    'batch': 1,
    'seq_len': 2048,
    'emb_dim': 4096,
    'mlp_dim': 1024,
    'num_experts': 64,
    'num_experts_per_tok': 4,
    'shared_experts': 1,
    'shared_mlp_dim': 11008,
    'routed_scaling_factor': 2.0,
}


def sigmoid_routing(
    x: jnp.ndarray,
    router_weights: jnp.ndarray,
    router_bias: jnp.ndarray,
    num_experts_per_tok: int,
) -> tuple:
    """
    Sigmoid-based routing with bias as used in DeepSeek V3.

    Unlike softmax routing, sigmoid allows independent scoring of experts.

    Args:
        x: [batch, seq_len, emb_dim]
        router_weights: [emb_dim, num_experts]
        router_bias: [num_experts] learnable bias for load balancing
        num_experts_per_tok: Number of experts per token

    Returns:
        (router_probs, selected_experts)
    """
    # Compute router logits
    router_logits = jnp.dot(x, router_weights)  # [batch, seq, num_experts]

    # Apply sigmoid scoring (not softmax!)
    router_scores = jax.nn.sigmoid(router_logits)

    # Add bias for load balancing
    router_scores = router_scores + router_bias

    # Get top-k experts
    top_k_scores, top_k_indices = jax.lax.top_k(router_scores, num_experts_per_tok)

    # Normalize the selected scores to sum to 1
    router_probs = top_k_scores / (top_k_scores.sum(axis=-1, keepdims=True) + 1e-6)

    return router_probs, top_k_indices


def shared_expert_mlp(
    x: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    up_kernel: jnp.ndarray,
    down_kernel: jnp.ndarray,
) -> jnp.ndarray:
    """
    Shared expert MLP that processes all tokens.

    This is a standard SwiGLU but with larger dimensions than routed experts.
    """
    gate = jax.nn.silu(jnp.dot(x, gate_kernel))
    up = jnp.dot(x, up_kernel)
    hidden = gate * up
    output = jnp.dot(hidden, down_kernel)
    return output


def routed_experts_forward(
    x: jnp.ndarray,
    router_probs: jnp.ndarray,
    selected_experts: jnp.ndarray,
    expert_gate_kernels: jnp.ndarray,
    expert_up_kernels: jnp.ndarray,
    expert_down_kernels: jnp.ndarray,
    scaling_factor: float,
) -> jnp.ndarray:
    """
    Forward pass through routed experts.

    Uses einsum for efficient batched computation on accelerators.

    Args:
        x: [batch, seq_len, emb_dim]
        router_probs: [batch, seq_len, num_experts_per_tok]
        selected_experts: [batch, seq_len, num_experts_per_tok]
        expert_*_kernels: [num_experts, ...]
        scaling_factor: Multiplier for routed output (2.5 in V3)

    Returns:
        output: [batch, seq_len, emb_dim]
    """
    batch, seq_len, emb_dim = x.shape
    num_experts = expert_gate_kernels.shape[0]

    # Compute all experts in parallel
    gate_out = jnp.einsum('bse,nem->bsnm', x, expert_gate_kernels)
    gate_out = jax.nn.silu(gate_out)

    up_out = jnp.einsum('bse,nem->bsnm', x, expert_up_kernels)

    hidden = gate_out * up_out

    expert_outputs = jnp.einsum('bsnm,nme->bsne', hidden, expert_down_kernels)
    # expert_outputs: [batch, seq, num_experts, emb]

    # Create one-hot and weight by routing probs
    one_hot = jax.nn.one_hot(selected_experts, num_experts)  # [b, s, k, n]
    weighted_one_hot = one_hot * router_probs[..., None]     # [b, s, k, n]
    expert_weights = weighted_one_hot.sum(axis=2)            # [b, s, n]

    # Combine expert outputs
    output = jnp.einsum('bsne,bsn->bse', expert_outputs, expert_weights)

    # Apply scaling factor
    output = output * scaling_factor

    return output


def moe_with_shared_forward(
    x: jnp.ndarray,
    # Router
    router_weights: jnp.ndarray,
    router_bias: jnp.ndarray,
    # Shared expert
    shared_gate: jnp.ndarray,
    shared_up: jnp.ndarray,
    shared_down: jnp.ndarray,
    # Routed experts
    expert_gate_kernels: jnp.ndarray,
    expert_up_kernels: jnp.ndarray,
    expert_down_kernels: jnp.ndarray,
    # Config
    num_experts_per_tok: int,
    scaling_factor: float,
) -> tuple:
    """
    DeepSeek V3 MoE with shared experts.

    Architecture:
    1. Shared expert processes ALL tokens
    2. Router selects top-k routed experts per token
    3. Routed expert outputs are scaled
    4. Final output = shared_output + scaled_routed_output

    Args:
        x: [batch, seq_len, emb_dim]
        Various weight tensors
        Config parameters

    Returns:
        (output, aux_info) where output is [batch, seq_len, emb_dim]
    """
    # 1. Shared expert (always active)
    shared_output = shared_expert_mlp(x, shared_gate, shared_up, shared_down)

    # 2. Routing
    router_probs, selected_experts = sigmoid_routing(
        x, router_weights, router_bias, num_experts_per_tok
    )

    # 3. Routed experts
    routed_output = routed_experts_forward(
        x,
        router_probs,
        selected_experts,
        expert_gate_kernels,
        expert_up_kernels,
        expert_down_kernels,
        scaling_factor,
    )

    # 4. Combine
    output = shared_output + routed_output

    # Compute bias update for load balancing (used during training)
    num_experts = router_weights.shape[-1]
    tokens_per_expert = jax.nn.one_hot(selected_experts, num_experts).sum(axis=(0, 1, 2))
    total_tokens = x.shape[0] * x.shape[1] * num_experts_per_tok
    avg_load = total_tokens / num_experts
    bias_update = jnp.sign(avg_load - tokens_per_expert) * 0.001  # Small update rate

    aux_info = {
        'bias_update': bias_update,
        'tokens_per_expert': tokens_per_expert,
    }

    return output, aux_info


def create_inputs(config: dict, dtype=jnp.bfloat16):
    """Create random inputs matching config."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)

    batch = config['batch']
    seq_len = config['seq_len']
    emb_dim = config['emb_dim']
    mlp_dim = config['mlp_dim']
    shared_mlp_dim = config['shared_mlp_dim']
    num_experts = config['num_experts']

    x = jax.random.normal(keys[0], (batch, seq_len, emb_dim), dtype=dtype)

    # Router
    router_weights = jax.random.normal(keys[1], (emb_dim, num_experts), dtype=dtype) * 0.02
    router_bias = jnp.zeros((num_experts,), dtype=dtype)

    # Shared expert (larger MLP)
    shared_gate = jax.random.normal(keys[2], (emb_dim, shared_mlp_dim), dtype=dtype) * 0.02
    shared_up = jax.random.normal(keys[3], (emb_dim, shared_mlp_dim), dtype=dtype) * 0.02
    shared_down = jax.random.normal(keys[4], (shared_mlp_dim, emb_dim), dtype=dtype) * 0.02

    # Routed experts (smaller per-expert MLP)
    expert_gate = jax.random.normal(keys[5], (num_experts, emb_dim, mlp_dim), dtype=dtype) * 0.02
    expert_up = jax.random.normal(keys[6], (num_experts, emb_dim, mlp_dim), dtype=dtype) * 0.02
    expert_down = jax.random.normal(keys[7], (num_experts, mlp_dim, emb_dim), dtype=dtype) * 0.02

    return (x, router_weights, router_bias,
            shared_gate, shared_up, shared_down,
            expert_gate, expert_up, expert_down)


def benchmark_moe_shared(config: dict, num_warmup: int = 5, num_iters: int = 50):
    """Benchmark MoE with shared experts."""
    import time

    inputs = create_inputs(config)
    x = inputs[0]

    moe_fn = jax.jit(partial(
        moe_with_shared_forward,
        num_experts_per_tok=config['num_experts_per_tok'],
        scaling_factor=config['routed_scaling_factor'],
    ))

    # Warmup
    for _ in range(num_warmup):
        output, aux = moe_fn(*inputs)
        output.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        output, aux = moe_fn(*inputs)
        output.block_until_ready()
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_iters * 1000

    # Calculate FLOPS
    batch = config['batch']
    seq_len = config['seq_len']
    emb_dim = config['emb_dim']
    mlp_dim = config['mlp_dim']
    shared_mlp_dim = config['shared_mlp_dim']
    k = config['num_experts_per_tok']

    # Shared expert: 3 matmuls with full shared_mlp_dim
    shared_flops = batch * seq_len * emb_dim * shared_mlp_dim * 2 * 3

    # Routed experts: k experts per token
    routed_flops = batch * seq_len * k * (emb_dim * mlp_dim * 2 * 3)

    # Routing overhead
    routing_flops = batch * seq_len * emb_dim * config['num_experts'] * 2

    total_flops = shared_flops + routed_flops + routing_flops
    tflops = total_flops / (avg_time_ms / 1000) / 1e12

    return {
        'config': config['name'],
        'time_ms': avg_time_ms,
        'tflops': tflops,
        'shape': list(output.shape),
    }


if __name__ == '__main__':
    print("=" * 80)
    print("DEEPSEEK V3 MoE WITH SHARED EXPERTS BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    configs = [DEEPSEEK_V3_SMALL, DEEPSEEK_V3]

    print("Architecture Details:")
    print("-" * 60)
    for config in configs:
        print(f"  {config['name']}:")
        print(f"    - Routed experts: {config['num_experts']} (top-{config['num_experts_per_tok']})")
        print(f"    - Shared experts: {config['shared_experts']}")
        print(f"    - Scaling factor: {config['routed_scaling_factor']}")
    print()

    print(f"{'Config':<25} | {'Time (ms)':>10} | {'TFLOPS':>8} | Output Shape")
    print("-" * 80)

    for config in configs:
        try:
            result = benchmark_moe_shared(config)
            print(f"{result['config']:<25} | {result['time_ms']:>10.2f} | {result['tflops']:>8.1f} | {result['shape']}")
        except Exception as e:
            print(f"{config['name']:<25} | ERROR: {e}")
