"""Block-Sparse Attention — BigBird (google/bigbird-roberta-base).

Combines local sliding window, global tokens, and random block attention
for O(n) complexity on long sequences. Minimum seq_len = 7 * block_size.

Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/big_bird/modeling_flax_big_bird.py
Paper: "Big Bird: Transformers for Longer Sequences" (Zaheer et al., 2020)
"""
import jax
import jax.numpy as jnp
from functools import partial

CONFIG = {
    'name': 'bigbird_block_sparse_attention',
    'model': 'BigBird-RoBERTa-Base',
    'operator': 'block_sparse_attention',
    'batch': 1,
    'seq_len': 4096,
    'num_heads': 12,
    'head_dim': 64,
    'block_size': 64,
    'num_random_blocks': 3,
    'window_blocks': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    """Returns (query, key, value, rand_attn)."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B, S = CONFIG['batch'], CONFIG['seq_len']
    H, D = CONFIG['num_heads'], CONFIG['head_dim']
    BS = CONFIG['block_size']
    n_blocks = S // BS
    n_rand = CONFIG['num_random_blocks']
    query = jax.random.normal(k1, (B, S, H, D), dtype=dtype)
    key_t = jax.random.normal(k2, (B, S, H, D), dtype=dtype)
    value = jax.random.normal(k3, (B, S, H, D), dtype=dtype)
    # Random block indices for middle blocks: (H, n_blocks-2, n_rand)
    # Each middle query block randomly attends to n_rand key blocks
    rand_attn = jax.random.randint(
        k4, (H, n_blocks - 2, n_rand), minval=1, maxval=n_blocks - 1
    )
    return query, key_t, value, rand_attn


def workload(query, key, value, rand_attn):
    """BigBird block-sparse attention.

    5 regions:
    1. First block: attends to all (global)
    2. Second block: sliding + random + global endpoints
    3. Middle blocks: 3-block sliding window + random + global endpoints
    4. Second-to-last: sliding + random + global endpoints
    5. Last block: attends to all (global)
    """
    B, S, H, D = query.shape
    BS = CONFIG['block_size']
    n_rand = CONFIG['num_random_blocks']
    n_blocks = S // BS
    scale = D ** -0.5
    mask_val = -10000.0
    # Reshape to blocks: (B, H, n_blocks, BS, D)
    q = query.transpose(0, 2, 1, 3).reshape(B, H, n_blocks, BS, D)
    k = key.transpose(0, 2, 1, 3).reshape(B, H, n_blocks, BS, D)
    v = value.transpose(0, 2, 1, 3).reshape(B, H, n_blocks, BS, D)
    # Flatten key/value for global attention: (B, H, S, D)
    k_flat = k.reshape(B, H, S, D)
    v_flat = v.reshape(B, H, S, D)

    # --- Part 1: First block (global) ---
    attn1 = jnp.einsum('bhqd,bhkd->bhqk', q[:, :, 0], k_flat) * scale  # (B,H,BS,S)
    attn1 = jax.nn.softmax(attn1, axis=-1)
    out1 = jnp.einsum('bhqk,bhkd->bhqd', attn1, v_flat)  # (B,H,BS,D)

    # --- Part 5: Last block (global) ---
    attn5 = jnp.einsum('bhqd,bhkd->bhqk', q[:, :, -1], k_flat) * scale
    attn5 = jax.nn.softmax(attn5, axis=-1)
    out5 = jnp.einsum('bhqk,bhkd->bhqd', attn5, v_flat)

    # --- Part 3: Middle blocks (sliding + random + global endpoints) ---
    # Sliding window keys: concatenate left, self, right blocks
    k_left = k[:, :, 1:-3]   # (B, H, n_blocks-4, BS, D)
    k_self = k[:, :, 2:-2]
    k_right = k[:, :, 3:-1]
    v_left = v[:, :, 1:-3]
    v_self = v[:, :, 2:-2]
    v_right = v[:, :, 3:-1]
    k_window = jnp.concatenate([k_left, k_self, k_right], axis=3)  # (B,H,n-4,3*BS,D)
    v_window = jnp.concatenate([v_left, v_self, v_right], axis=3)
    q_mid = q[:, :, 2:-2]  # (B, H, n_blocks-4, BS, D)
    # Sliding window scores
    attn_window = jnp.einsum('bhlqd,bhlkd->bhlqk', q_mid, k_window) * scale
    # Global first block scores
    attn_first = jnp.einsum('bhlqd,bhkd->bhlqk', q_mid, k[:, :, 0]) * scale
    # Global last block scores
    attn_last = jnp.einsum('bhlqd,bhkd->bhlqk', q_mid, k[:, :, -1]) * scale
    # Random block scores: gather random key blocks
    # rand_attn: (H, n_blocks-2, n_rand) -> use middle portion (H, n_blocks-4, n_rand)
    rand_mid = rand_attn[:, 1:-1, :]  # (H, n_blocks-4, n_rand)

    def gather_random(tensor, indices):
        """Gather random blocks: tensor (B,H,n,BS,D), indices (H,n-4,n_rand) -> (B,H,n-4,n_rand*BS,D)."""
        # For each head and middle block position, gather n_rand key blocks
        def _gather_one(t, idx):
            # t: (n_blocks, BS, D), idx: (n-4, n_rand)
            gathered = t[idx]  # (n-4, n_rand, BS, D)
            return gathered.reshape(idx.shape[0], -1, D)  # (n-4, n_rand*BS, D)
        # vmap over batch then heads
        return jax.vmap(jax.vmap(_gather_one))(tensor, jnp.broadcast_to(indices[None], (B,) + indices.shape))

    k_rand = gather_random(k, rand_mid)  # (B, H, n-4, n_rand*BS, D)
    v_rand = gather_random(v, rand_mid)
    attn_rand = jnp.einsum('bhlqd,bhlkd->bhlqk', q_mid, k_rand) * scale
    # Concatenate all score components: first + window + random + last
    attn_mid = jnp.concatenate([attn_first, attn_window, attn_rand, attn_last], axis=-1)
    attn_mid = jax.nn.softmax(attn_mid, axis=-1)
    # Split weights and aggregate values from each source
    n_keys = attn_mid.shape[-1]
    w_first = attn_mid[..., :BS]
    w_window = attn_mid[..., BS:BS + 3 * BS]
    w_rand = attn_mid[..., BS + 3 * BS:n_keys - BS]
    w_last = attn_mid[..., n_keys - BS:]
    out_mid = (
        jnp.einsum('bhlqk,bhkd->bhlqd', w_first, v[:, :, 0]) +
        jnp.einsum('bhlqk,bhlkd->bhlqd', w_window, v_window) +
        jnp.einsum('bhlqk,bhlkd->bhlqd', w_rand, v_rand) +
        jnp.einsum('bhlqk,bhkd->bhlqd', w_last, v[:, :, -1])
    )  # (B, H, n-4, BS, D)

    # --- Part 2: Second block (simplified: sliding + global endpoints) ---
    k2_cat = jnp.concatenate([
        k[:, :, 0], k[:, :, 1], k[:, :, 2], k[:, :, -1]
    ], axis=2)  # (B, H, 4*BS, D)
    v2_cat = jnp.concatenate([
        v[:, :, 0], v[:, :, 1], v[:, :, 2], v[:, :, -1]
    ], axis=2)
    attn2 = jnp.einsum('bhqd,bhkd->bhqk', q[:, :, 1], k2_cat) * scale
    attn2 = jax.nn.softmax(attn2, axis=-1)
    out2 = jnp.einsum('bhqk,bhkd->bhqd', attn2, v2_cat)

    # --- Part 4: Second-to-last block ---
    k4_cat = jnp.concatenate([
        k[:, :, 0], k[:, :, -3], k[:, :, -2], k[:, :, -1]
    ], axis=2)
    v4_cat = jnp.concatenate([
        v[:, :, 0], v[:, :, -3], v[:, :, -2], v[:, :, -1]
    ], axis=2)
    attn4 = jnp.einsum('bhqd,bhkd->bhqk', q[:, :, -2], k4_cat) * scale
    attn4 = jax.nn.softmax(attn4, axis=-1)
    out4 = jnp.einsum('bhqk,bhkd->bhqd', attn4, v4_cat)

    # --- Assemble ---
    # out1, out2: (B,H,BS,D) -> (B,H,1,BS,D)
    # out_mid: (B,H,n-4,BS,D)
    # out4, out5: (B,H,BS,D) -> (B,H,1,BS,D)
    out = jnp.concatenate([
        out1[:, :, None],
        out2[:, :, None],
        out_mid,
        out4[:, :, None],
        out5[:, :, None],
    ], axis=2)  # (B, H, n_blocks, BS, D)
    out = out.reshape(B, H, S, D)
    return out.transpose(0, 2, 1, 3)  # (B, S, H, D)


def benchmark(num_warmup=5, num_iters=100):
    """Benchmark and return results dict."""
    import time
    inputs = create_inputs()
    fn = jax.jit(workload)
    for _ in range(num_warmup):
        out = fn(*inputs)
        out.block_until_ready()
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    import numpy as np
    times = np.array(times) * 1000
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    BS = CONFIG['block_size']
    n_blocks = S // BS
    n_rand = CONFIG['num_random_blocks']
    # Global (first+last): 2 * B*H*BS*S*D
    # Middle sliding: (n_blocks-4) * B*H*BS*(3*BS)*D
    # Middle random: (n_blocks-4) * B*H*BS*(n_rand*BS)*D
    # Middle global endpoints: (n_blocks-4) * B*H*BS*2*BS*D
    # Edge blocks: 2 * B*H*BS*4*BS*D
    flops = (
        2 * B * H * BS * S * D * 4 +
        (n_blocks - 4) * B * H * BS * (3 + n_rand + 2) * BS * D * 4 +
        2 * B * H * BS * 4 * BS * D * 4
    )
    avg = float(np.mean(times))
    return {
        'name': CONFIG['name'],
        'model': CONFIG['model'],
        'operator': CONFIG['operator'],
        'config': {k: v for k, v in CONFIG.items() if k not in ('name', 'model', 'operator')},
        'time_ms': round(avg, 4),
        'std_ms': round(float(np.std(times)), 4),
        'tflops': round(flops / (avg / 1000) / 1e12, 2),
        'output_shape': list(out.shape),
        'status': 'success',
    }


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark()))
