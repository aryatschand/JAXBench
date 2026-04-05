#!/usr/bin/env python3
"""Run all 50 benchmark workloads on TPU, collecting timing and FLOP utilization.

Runs each workload's baseline.py (and optimized.py if present), measures runtime,
extracts XLA FLOP counts via cost_analysis, and computes TPU utilization %.

Usage (on TPU VM):
    PJRT_DEVICE=TPU python3 benchmark/run_all.py

Output: benchmark/results.json + summary table to stdout.
"""

import importlib
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# TPU v6e-1 peak TFLOPS (bf16, single chip, both MXU paths)
# Source: Google Cloud TPU v6e spec sheet
TPU_PEAK_TFLOPS_BF16 = 460.0

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BENCHMARK_DIR))  # project root


def get_xla_flops(fn, inputs):
    """Extract FLOP count from XLA cost analysis."""
    try:
        compiled = jax.jit(fn).lower(*inputs).compile()
        cost = compiled.cost_analysis()
        if isinstance(cost, list):
            cost = cost[0] if cost else {}
        flops = cost.get('flops', 0)
        return int(flops)
    except Exception:
        return 0


def benchmark_fn(fn, inputs, num_warmup=5, num_iters=50):
    """Time a JIT-compiled function. Returns (times_ms, output)."""
    jitted = jax.jit(fn)
    # Warmup
    for _ in range(num_warmup):
        out = jitted(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    # Measure
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = jitted(*inputs)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    return times, out


def run_one(workload_dir, variant='baseline'):
    """Run one workload variant. Returns result dict."""
    name = os.path.basename(workload_dir)
    module_path = os.path.join(workload_dir, f'{variant}.py')

    if not os.path.exists(module_path):
        return None

    result = {
        'name': name,
        'variant': variant,
        'status': 'error',
    }

    try:
        # Dynamic import
        spec = importlib.util.spec_from_file_location(
            f'{name}.{variant}', module_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Get CONFIG
        config = getattr(mod, 'CONFIG', {})
        result['config'] = {k: v for k, v in config.items()
                           if isinstance(v, (int, float, str, bool))}

        # Check for _skip_jit
        skip_jit = getattr(mod, '_skip_jit', False)

        # Create inputs
        create_fn = getattr(mod, 'create_inputs')
        if 'dtype' in create_fn.__code__.co_varnames:
            inputs = create_fn(dtype=jnp.bfloat16)
        else:
            inputs = create_fn()
        if not isinstance(inputs, (list, tuple)):
            inputs = (inputs,)

        # Input shapes
        result['input_shapes'] = [
            list(x.shape) if hasattr(x, 'shape') else str(type(x))
            for x in inputs
        ]

        # Get XLA FLOP count (skip for non-jittable)
        workload_fn = getattr(mod, 'workload')
        xla_flops = 0
        if not skip_jit:
            xla_flops = get_xla_flops(workload_fn, inputs)
        result['xla_flops'] = xla_flops

        # Benchmark
        if skip_jit:
            # Eager benchmarking
            for _ in range(2):
                out = workload_fn(*inputs)
                if hasattr(out, 'block_until_ready'):
                    out.block_until_ready()
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                out = workload_fn(*inputs)
                if hasattr(out, 'block_until_ready'):
                    out.block_until_ready()
                times.append((time.perf_counter() - t0) * 1000)
        else:
            times, out = benchmark_fn(workload_fn, inputs)

        # Compute stats
        times_arr = np.array(times)
        median_ms = float(np.median(times_arr))
        mean_ms = float(np.mean(times_arr))
        std_ms = float(np.std(times_arr))
        min_ms = float(np.min(times_arr))

        # Compute TFLOPS and utilization
        tflops = 0.0
        utilization_pct = 0.0
        if xla_flops > 0 and median_ms > 0:
            tflops = xla_flops / (median_ms / 1000) / 1e12
            utilization_pct = (tflops / TPU_PEAK_TFLOPS_BF16) * 100

        result.update({
            'status': 'success',
            'median_ms': round(median_ms, 4),
            'mean_ms': round(mean_ms, 4),
            'std_ms': round(std_ms, 4),
            'min_ms': round(min_ms, 4),
            'xla_flops': xla_flops,
            'tflops': round(tflops, 2),
            'utilization_pct': round(utilization_pct, 1),
            'output_shape': list(out.shape) if hasattr(out, 'shape') else [],
            'num_iters': len(times),
        })

    except Exception as e:
        import traceback
        result['error'] = str(e)[:300]
        result['traceback'] = traceback.format_exc()[-500:]

    return result


def main():
    print(f"JAX {jax.__version__} | Devices: {jax.devices()}")
    print(f"TPU peak: {TPU_PEAK_TFLOPS_BF16} TFLOPS (bf16)")
    print()

    # Discover workloads
    workload_dirs = sorted([
        os.path.join(BENCHMARK_DIR, d)
        for d in os.listdir(BENCHMARK_DIR)
        if os.path.isdir(os.path.join(BENCHMARK_DIR, d))
        and not d.startswith(('_', '.'))
        and os.path.exists(os.path.join(BENCHMARK_DIR, d, 'baseline.py'))
    ])

    print(f"Found {len(workload_dirs)} workloads")
    print()

    results = []
    header = f"{'#':>3} {'Workload':<42} {'Var':<10} {'Median(ms)':>10} {'TFLOPS':>8} {'Util%':>7} {'XLA FLOPS':>14}"
    print(header)
    print("-" * len(header))

    for i, wd in enumerate(workload_dirs, 1):
        name = os.path.basename(wd)

        for variant in ['baseline', 'optimized']:
            vpath = os.path.join(wd, f'{variant}.py')
            if not os.path.exists(vpath):
                continue

            # Clear TPU state between runs
            jax.clear_caches()

            r = run_one(wd, variant)
            if r is None:
                continue

            results.append(r)

            if r['status'] == 'success':
                flops_str = f"{r['xla_flops']:,}" if r['xla_flops'] > 0 else "—"
                tflops_str = f"{r['tflops']:.1f}" if r['tflops'] > 0 else "—"
                util_str = f"{r['utilization_pct']:.1f}%" if r['utilization_pct'] > 0 else "—"
                print(f"{i:>3} {name:<42} {variant:<10} {r['median_ms']:>10.3f} {tflops_str:>8} {util_str:>7} {flops_str:>14}")
            else:
                err = r.get('error', '?')[:40]
                print(f"{i:>3} {name:<42} {variant:<10} {'ERROR':>10} {'':>8} {'':>7} {err}")

    # Summary
    print()
    successes = [r for r in results if r['status'] == 'success']
    baselines = [r for r in successes if r['variant'] == 'baseline']
    optimized = [r for r in successes if r['variant'] == 'optimized']

    print(f"Total: {len(results)} runs ({len(successes)} success, {len(results)-len(successes)} error)")
    print(f"Baselines: {len(baselines)}/50 succeeded")
    if optimized:
        print(f"Optimized: {len(optimized)} variants")

    # Utilization stats
    utils = [r['utilization_pct'] for r in baselines if r['utilization_pct'] > 0]
    if utils:
        print(f"\nTPU Utilization (baselines with FLOP data):")
        print(f"  Median: {np.median(utils):.1f}%")
        print(f"  Mean:   {np.mean(utils):.1f}%")
        print(f"  Max:    {np.max(utils):.1f}%")
        print(f"  Min:    {np.min(utils):.1f}%")

    # Speedup stats for optimized variants
    if optimized:
        print(f"\nOptimized vs Baseline speedups:")
        for opt_r in optimized:
            base_r = next((r for r in baselines if r['name'] == opt_r['name']), None)
            if base_r:
                sp = base_r['median_ms'] / opt_r['median_ms'] if opt_r['median_ms'] > 0 else 0
                print(f"  {opt_r['name']}: {base_r['median_ms']:.3f}ms -> {opt_r['median_ms']:.3f}ms ({sp:.2f}x)")

    # Save results
    output = {
        'metadata': {
            'jax_version': jax.__version__,
            'devices': str(jax.devices()),
            'tpu_peak_tflops_bf16': TPU_PEAK_TFLOPS_BF16,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'num_workloads': len(workload_dirs),
            'num_succeeded': len(successes),
        },
        'results': results,
    }
    output_path = os.path.join(BENCHMARK_DIR, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
