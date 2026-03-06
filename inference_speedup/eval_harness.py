"""End-to-end inference evaluation harness.

Measures tokens/s for prefill and decode, profiles kernel time breakdown,
predicts impact of kernel speedups (Amdahl's law), and validates kernel
swaps with empirical measurement.

Usage:
    from inference_speedup.eval_harness import evaluate_model
    results = evaluate_model('llama3_8b', profile=True, optimized_kernels=['rmsnorm'])
"""

import time
import json
import jax
import jax.numpy as jnp
import numpy as np

from inference_speedup.config import ALL_MODELS
from inference_speedup.kernels import get_kernel, swap_kernel, reset_kernels, list_kernels
from inference_speedup.pallas_kernels import AVAILABLE_PALLAS_KERNELS
from inference_speedup.models import llama3, gla, mamba2


# Model dispatch table
MODEL_MODULES = {
    'llama3': llama3,
    'gla': gla,
    'mamba2': mamba2,
}


def _get_module(config):
    return MODEL_MODULES[config['model_type']]


# ---------------------------------------------------------------------------
# Kernel profiling — time each kernel in isolation
# ---------------------------------------------------------------------------

def profile_kernels(config, batch_size=1, seq_len=2048, num_warmup=5, num_iters=50):
    """Time each kernel used by the model in isolation.

    Returns dict of kernel_name -> {time_ms, calls_per_forward, total_ms_per_forward}
    """
    D = config['d_model']
    n_layers = config['eval_layers']
    dtype = jnp.bfloat16

    # Build input shapes for each kernel type
    kernel_benchmarks = {}

    # RMSNorm
    x_norm = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, D), dtype=dtype)
    w_norm = jnp.ones(D, dtype=dtype)
    fn = jax.jit(lambda x, w: get_kernel('rmsnorm')(x, w, eps=config['rms_norm_eps']))
    kernel_benchmarks['rmsnorm'] = (fn, (x_norm, w_norm))

    # Token embed
    tokens = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, config['vocab_size'])
    embed_table = jax.random.normal(jax.random.PRNGKey(2), (config['vocab_size'], D), dtype=dtype) * 0.02
    fn = jax.jit(get_kernel('token_embed'))
    kernel_benchmarks['token_embed'] = (fn, (tokens, embed_table))

    if config['model_type'] == 'llama3':
        Hq, Hkv, Dh = config['num_heads'], config['num_kv_heads'], config['head_dim']
        FFN = config['ffn_dim']

        # RoPE
        q = jax.random.normal(jax.random.PRNGKey(3), (batch_size, seq_len, Hq, Dh), dtype=dtype)
        k = jax.random.normal(jax.random.PRNGKey(4), (batch_size, seq_len, Hkv, Dh), dtype=dtype)
        positions = jnp.arange(seq_len)
        fn = jax.jit(lambda q, k: get_kernel('rope')(q, k, positions, theta=config['rope_theta']))
        kernel_benchmarks['rope'] = (fn, (q, k))

        # GQA attention
        v = jax.random.normal(jax.random.PRNGKey(5), (batch_size, seq_len, Hkv, Dh), dtype=dtype)
        fn = jax.jit(get_kernel('gqa_attention'))
        kernel_benchmarks['gqa_attention'] = (fn, (q, k, v))

        # SwiGLU
        x_mlp = jax.random.normal(jax.random.PRNGKey(6), (batch_size, seq_len, D), dtype=dtype)
        wg = jax.random.normal(jax.random.PRNGKey(7), (D, FFN), dtype=dtype) * 0.02
        wu = jax.random.normal(jax.random.PRNGKey(8), (D, FFN), dtype=dtype) * 0.02
        wd = jax.random.normal(jax.random.PRNGKey(9), (FFN, D), dtype=dtype) * 0.02
        fn = jax.jit(get_kernel('swiglu_mlp'))
        kernel_benchmarks['swiglu_mlp'] = (fn, (x_mlp, wg, wu, wd))

    elif config['model_type'] == 'gla':
        H, Dh = config['num_heads'], config['head_dim']
        FFN = config['ffn_dim']

        # GLA attention
        q = jax.random.normal(jax.random.PRNGKey(3), (batch_size, H, seq_len, Dh), dtype=dtype)
        k = jax.random.normal(jax.random.PRNGKey(4), (batch_size, H, seq_len, Dh), dtype=dtype)
        v = jax.random.normal(jax.random.PRNGKey(5), (batch_size, H, seq_len, Dh), dtype=dtype)
        gate = jax.random.normal(jax.random.PRNGKey(6), (batch_size, H, seq_len), dtype=jnp.float32)
        fn = jax.jit(get_kernel('gated_linear_attention'))
        kernel_benchmarks['gated_linear_attention'] = (fn, (q, k, v, gate))

        # SwiGLU
        x_mlp = jax.random.normal(jax.random.PRNGKey(7), (batch_size, seq_len, D), dtype=dtype)
        wg = jax.random.normal(jax.random.PRNGKey(8), (D, FFN), dtype=dtype) * 0.02
        wu = jax.random.normal(jax.random.PRNGKey(9), (D, FFN), dtype=dtype) * 0.02
        wd = jax.random.normal(jax.random.PRNGKey(10), (FFN, D), dtype=dtype) * 0.02
        fn = jax.jit(get_kernel('swiglu_mlp'))
        kernel_benchmarks['swiglu_mlp'] = (fn, (x_mlp, wg, wu, wd))

    elif config['model_type'] == 'mamba2':
        H, Dh = config['num_heads'], config['head_dim']

        # SSD attention
        q = jax.random.normal(jax.random.PRNGKey(3), (batch_size, H, seq_len, Dh), dtype=dtype)
        k = jax.random.normal(jax.random.PRNGKey(4), (batch_size, H, seq_len, Dh), dtype=dtype)
        v = jax.random.normal(jax.random.PRNGKey(5), (batch_size, H, seq_len, Dh), dtype=dtype)
        A_log = jax.random.normal(jax.random.PRNGKey(6), (batch_size, H, seq_len), dtype=jnp.float32) * 0.5 - 4.0
        fn = jax.jit(get_kernel('ssd_attention'))
        kernel_benchmarks['ssd_attention'] = (fn, (q, k, v, A_log))

    # Time each kernel
    results = {}
    for name, (fn, inputs) in kernel_benchmarks.items():
        # Warmup
        for _ in range(num_warmup):
            out = fn(*inputs)
            if isinstance(out, tuple):
                out[0].block_until_ready()
            else:
                out.block_until_ready()

        # Benchmark
        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            out = fn(*inputs)
            if isinstance(out, tuple):
                out[0].block_until_ready()
            else:
                out.block_until_ready()
            times.append(time.perf_counter() - t0)

        times_ms = np.array(times) * 1000
        avg_ms = float(np.mean(times_ms))

        # Count calls per forward pass
        block_count = config['block_kernels'].get(name, 0)
        forward_count = config['forward_kernels'].get(name, 0)
        total_calls = block_count * n_layers + forward_count

        results[name] = {
            'time_ms': round(avg_ms, 4),
            'std_ms': round(float(np.std(times_ms)), 4),
            'calls_per_forward': total_calls,
            'total_ms_per_forward': round(avg_ms * total_calls, 4),
        }

    return results


# ---------------------------------------------------------------------------
# End-to-end inference benchmark
# ---------------------------------------------------------------------------

def benchmark_prefill(config, weights, batch_size=1, seq_len=2048,
                      num_warmup=3, num_iters=20):
    """Benchmark prefill tokens/s."""
    mod = _get_module(config)
    tokens = jax.random.randint(jax.random.PRNGKey(42), (batch_size, seq_len), 0, config['vocab_size'])

    fn = jax.jit(lambda t: mod.prefill(weights, t, config))

    # Warmup
    for _ in range(num_warmup):
        out = fn(tokens)
        jax.tree.map(lambda x: x.block_until_ready(), out)

    # Benchmark
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn(tokens)
        jax.tree.map(lambda x: x.block_until_ready(), out)
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    avg_ms = float(np.mean(times_ms))
    tokens_per_sec = (batch_size * seq_len) / (avg_ms / 1000)

    return {
        'mode': 'prefill',
        'batch_size': batch_size,
        'seq_len': seq_len,
        'time_ms': round(avg_ms, 4),
        'std_ms': round(float(np.std(times_ms)), 4),
        'tokens_per_sec': round(tokens_per_sec, 1),
    }


def benchmark_decode(config, weights, prompt_len=128, gen_tokens=64,
                     batch_size=1, num_warmup=2, num_iters=10):
    """Benchmark decode tokens/s (autoregressive generation)."""
    mod = _get_module(config)
    tokens = jax.random.randint(jax.random.PRNGKey(42), (batch_size, prompt_len), 0, config['vocab_size'])

    # Prefill first
    prefill_fn = jax.jit(lambda t: mod.prefill(weights, t, config))
    prefill_out = prefill_fn(tokens)
    jax.tree.map(lambda x: x.block_until_ready(), prefill_out)

    if config['model_type'] == 'llama3':
        logits, kv_cache_prefill = prefill_out
        # Extend KV cache to max length for decode
        max_seq = prompt_len + gen_tokens
        kv_cache = []
        for ck, cv in kv_cache_prefill:
            # Pad cache to max_seq
            pad_len = max_seq - ck.shape[2]
            ck = jnp.pad(ck, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
            cv = jnp.pad(cv, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
            kv_cache.append((ck, cv))

        decode_fn = jax.jit(lambda tok, cache, pos: mod.decode_step(weights, tok, cache, pos, config))

        # Warmup decode
        for _ in range(num_warmup):
            tok = jnp.zeros((batch_size, 1), dtype=jnp.int32)
            _, _ = decode_fn(tok, kv_cache, prompt_len)

        # Benchmark decode loop
        times = []
        for _ in range(num_iters):
            cache = kv_cache  # reset cache each iter
            t0 = time.perf_counter()
            for step in range(gen_tokens):
                tok = jnp.zeros((batch_size, 1), dtype=jnp.int32)
                _, cache = decode_fn(tok, cache, prompt_len + step)
            # Wait for last output
            jax.tree.map(lambda x: x.block_until_ready(), cache)
            times.append(time.perf_counter() - t0)

    elif config['model_type'] == 'gla':
        logits, states = prefill_out
        decode_fn = jax.jit(lambda tok, st: mod.decode_step(weights, tok, st, config))

        for _ in range(num_warmup):
            tok = jnp.zeros((batch_size, 1), dtype=jnp.int32)
            _, _ = decode_fn(tok, states)

        times = []
        for _ in range(num_iters):
            st = states
            t0 = time.perf_counter()
            for step in range(gen_tokens):
                tok = jnp.zeros((batch_size, 1), dtype=jnp.int32)
                _, st = decode_fn(tok, st)
            jax.tree.map(lambda x: x.block_until_ready(), st)
            times.append(time.perf_counter() - t0)

    elif config['model_type'] == 'mamba2':
        logits, ssm_states, conv_states = prefill_out
        decode_fn = jax.jit(lambda tok, ss, cs: mod.decode_step(weights, tok, ss, cs, config))

        for _ in range(num_warmup):
            tok = jnp.zeros((batch_size, 1), dtype=jnp.int32)
            _, _, _ = decode_fn(tok, ssm_states, conv_states)

        times = []
        for _ in range(num_iters):
            ss, cs = ssm_states, conv_states
            t0 = time.perf_counter()
            for step in range(gen_tokens):
                tok = jnp.zeros((batch_size, 1), dtype=jnp.int32)
                _, ss, cs = decode_fn(tok, ss, cs)
            jax.tree.map(lambda x: x.block_until_ready(), ss)
            times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    avg_ms = float(np.mean(times_ms))
    per_token_ms = avg_ms / gen_tokens
    tokens_per_sec = gen_tokens / (avg_ms / 1000)

    return {
        'mode': 'decode',
        'batch_size': batch_size,
        'prompt_len': prompt_len,
        'gen_tokens': gen_tokens,
        'total_time_ms': round(avg_ms, 4),
        'per_token_ms': round(per_token_ms, 4),
        'tokens_per_sec': round(tokens_per_sec, 1),
    }


# ---------------------------------------------------------------------------
# Speedup analysis — Amdahl's law predictions
# ---------------------------------------------------------------------------

def compute_speedup_predictions(kernel_profile, total_forward_ms):
    """Predict model-level speedup from kernel-level improvements.

    For each kernel, computes the impact of 1.5x, 2x, 3x, 5x speedup
    on overall tokens/s using Amdahl's law:
        new_total = total - kernel_time + kernel_time / speedup_factor
        model_speedup = total / new_total
    """
    predictions = {}
    for name, kdata in kernel_profile.items():
        kernel_total_ms = kdata['total_ms_per_forward']
        fraction = kernel_total_ms / total_forward_ms if total_forward_ms > 0 else 0

        speedups = {}
        for factor in [1.5, 2.0, 3.0, 5.0]:
            new_total = total_forward_ms - kernel_total_ms + kernel_total_ms / factor
            model_speedup = total_forward_ms / new_total if new_total > 0 else 1.0
            tokens_pct_increase = (model_speedup - 1.0) * 100
            speedups[f'{factor}x'] = {
                'new_forward_ms': round(new_total, 4),
                'model_speedup': round(model_speedup, 4),
                'tokens_pct_increase': round(tokens_pct_increase, 2),
            }

        predictions[name] = {
            'fraction_of_forward': round(fraction, 4),
            'total_ms_per_forward': kernel_total_ms,
            'speedup_impact': speedups,
        }

    return predictions


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_model(model_name, seq_len=2048, batch_size=1, num_layers=None,
                   profile=True, optimized_kernels=None, decode=True,
                   decode_prompt_len=128, decode_gen_tokens=64):
    """Run full evaluation for a model.

    Args:
        model_name: 'llama3_8b', 'gla_1_3b', or 'mamba2_2_7b'
        seq_len: sequence length for prefill
        batch_size: batch size
        num_layers: override eval layer count
        profile: whether to profile individual kernels
        optimized_kernels: list of kernel names to swap with Pallas versions
        decode: whether to benchmark decode as well
        decode_prompt_len: prompt length for decode benchmark
        decode_gen_tokens: number of tokens to generate in decode
    Returns:
        results dict
    """
    config = ALL_MODELS[model_name].copy()
    if num_layers is not None:
        config['eval_layers'] = num_layers
    mod = _get_module(config)

    print(f"\n{'=' * 70}")
    print(f"  {config['display_name']} Inference Benchmark")
    print(f"  Layers: {config['eval_layers']} (of {config['num_layers']}), "
          f"Seq: {seq_len}, Batch: {batch_size}")
    print(f"{'=' * 70}")

    # Reset to vanilla kernels
    reset_kernels()

    # --- Baseline measurements ---
    print("\n[1/5] Initializing model weights...")
    rng = jax.random.PRNGKey(0)
    weights = mod.init_weights(config, rng, num_layers=config['eval_layers'])
    # Wait for weight initialization to complete on device
    jax.tree.map(lambda x: x.block_until_ready(), weights)
    print(f"  Weights allocated on {jax.devices()[0]}")

    results = {
        'model': model_name,
        'config': {
            'eval_layers': config['eval_layers'],
            'full_layers': config['num_layers'],
            'd_model': config['d_model'],
            'seq_len': seq_len,
            'batch_size': batch_size,
        },
        'swapped_kernels': list(optimized_kernels or []),
    }

    # --- Kernel profiling ---
    kernel_profile = None
    if profile:
        print("\n[2/5] Profiling individual kernels...")
        kernel_profile = profile_kernels(config, batch_size=batch_size, seq_len=seq_len)
        results['kernel_profile'] = kernel_profile

        theoretical_total = sum(k['total_ms_per_forward'] for k in kernel_profile.values())
        print(f"\n  {'Kernel':<30} {'Time/call':>10} {'Calls':>6} {'Total':>10} {'Fraction':>9}")
        print(f"  {'-' * 65}")
        for name, kdata in sorted(kernel_profile.items(),
                                  key=lambda x: x[1]['total_ms_per_forward'], reverse=True):
            frac = kdata['total_ms_per_forward'] / theoretical_total * 100 if theoretical_total > 0 else 0
            print(f"  {name:<30} {kdata['time_ms']:>8.4f}ms {kdata['calls_per_forward']:>6} "
                  f"{kdata['total_ms_per_forward']:>8.4f}ms {frac:>7.1f}%")
        print(f"  {'Theoretical total':<30} {'':>10} {'':>6} {theoretical_total:>8.4f}ms")
    else:
        print("\n[2/5] Skipping kernel profiling (--no-profile)")

    # --- Baseline prefill ---
    print("\n[3/5] Benchmarking prefill (vanilla kernels)...")
    baseline_prefill = benchmark_prefill(config, weights, batch_size=batch_size, seq_len=seq_len)
    results['baseline_prefill'] = baseline_prefill
    print(f"  Prefill: {baseline_prefill['tokens_per_sec']:,.0f} tokens/s "
          f"({baseline_prefill['time_ms']:.2f}ms for {seq_len} tokens)")

    # --- Baseline decode ---
    baseline_decode = None
    if decode:
        print("\n[4/5] Benchmarking decode (vanilla kernels)...")
        baseline_decode = benchmark_decode(
            config, weights, prompt_len=decode_prompt_len,
            gen_tokens=decode_gen_tokens, batch_size=batch_size,
        )
        results['baseline_decode'] = baseline_decode
        print(f"  Decode: {baseline_decode['tokens_per_sec']:,.0f} tokens/s "
              f"({baseline_decode['per_token_ms']:.2f}ms/token)")
    else:
        print("\n[4/5] Skipping decode benchmark")

    # --- Speedup predictions ---
    if kernel_profile:
        print("\n[5/5] Kernel speedup impact analysis...")
        predictions = compute_speedup_predictions(kernel_profile, baseline_prefill['time_ms'])
        results['speedup_predictions'] = predictions

        print(f"\n  {'Kernel':<30} {'% of fwd':>8}  "
              f"{'2x→tok/s':>10}  {'3x→tok/s':>10}  {'5x→tok/s':>10}")
        print(f"  {'-' * 72}")
        for name, pred in sorted(predictions.items(),
                                 key=lambda x: x[1]['fraction_of_forward'], reverse=True):
            frac = pred['fraction_of_forward'] * 100
            s2 = pred['speedup_impact']['2.0x']['tokens_pct_increase']
            s3 = pred['speedup_impact']['3.0x']['tokens_pct_increase']
            s5 = pred['speedup_impact']['5.0x']['tokens_pct_increase']
            print(f"  {name:<30} {frac:>6.1f}%  "
                  f"{'+' if s2 >= 0 else ''}{s2:>8.1f}%  "
                  f"{'+' if s3 >= 0 else ''}{s3:>8.1f}%  "
                  f"{'+' if s5 >= 0 else ''}{s5:>8.1f}%")
    else:
        print("\n[5/5] Skipping speedup predictions (no profile data)")

    # --- Optimized kernel comparison ---
    if optimized_kernels:
        print(f"\n{'=' * 70}")
        print(f"  Optimized Kernel Comparison")
        print(f"  Swapping: {', '.join(optimized_kernels)}")
        print(f"{'=' * 70}")

        for kname in optimized_kernels:
            if kname in AVAILABLE_PALLAS_KERNELS:
                swap_kernel(kname, AVAILABLE_PALLAS_KERNELS[kname])
                print(f"  Swapped {kname} → Pallas optimized")
            else:
                print(f"  WARNING: No Pallas kernel available for '{kname}', skipping")

        # Re-profile with optimized kernels
        if profile:
            print("\n  Profiling optimized kernels...")
            opt_profile = profile_kernels(config, batch_size=batch_size, seq_len=seq_len)
            results['optimized_kernel_profile'] = opt_profile

            for name in optimized_kernels:
                if name in opt_profile and name in kernel_profile:
                    old = kernel_profile[name]['time_ms']
                    new = opt_profile[name]['time_ms']
                    speedup = old / new if new > 0 else float('inf')
                    print(f"  {name}: {old:.4f}ms → {new:.4f}ms ({speedup:.2f}x faster)")

        # Re-benchmark prefill
        print("\n  Benchmarking prefill (optimized)...")
        # Need to re-init weights and re-JIT with new kernel registry
        opt_prefill = benchmark_prefill(config, weights, batch_size=batch_size, seq_len=seq_len)
        results['optimized_prefill'] = opt_prefill

        prefill_speedup = baseline_prefill['tokens_per_sec'] / opt_prefill['tokens_per_sec'] \
            if opt_prefill['tokens_per_sec'] > 0 else 0
        prefill_improvement = (opt_prefill['tokens_per_sec'] / baseline_prefill['tokens_per_sec'] - 1) * 100 \
            if baseline_prefill['tokens_per_sec'] > 0 else 0

        print(f"  Prefill: {baseline_prefill['tokens_per_sec']:,.0f} → "
              f"{opt_prefill['tokens_per_sec']:,.0f} tokens/s "
              f"({'+' if prefill_improvement >= 0 else ''}{prefill_improvement:.1f}%)")

        # Re-benchmark decode
        if decode:
            print("  Benchmarking decode (optimized)...")
            opt_decode = benchmark_decode(
                config, weights, prompt_len=decode_prompt_len,
                gen_tokens=decode_gen_tokens, batch_size=batch_size,
            )
            results['optimized_decode'] = opt_decode

            decode_improvement = (opt_decode['tokens_per_sec'] / baseline_decode['tokens_per_sec'] - 1) * 100 \
                if baseline_decode and baseline_decode['tokens_per_sec'] > 0 else 0
            print(f"  Decode:  {baseline_decode['tokens_per_sec']:,.0f} → "
                  f"{opt_decode['tokens_per_sec']:,.0f} tokens/s "
                  f"({'+' if decode_improvement >= 0 else ''}{decode_improvement:.1f}%)")

        # Reset kernels
        reset_kernels()

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")

    return results
