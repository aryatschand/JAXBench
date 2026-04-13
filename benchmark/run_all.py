#!/usr/bin/env python3
"""Run all 50 benchmark workloads on TPU, collecting timing and FLOP utilization.

Uses jax.profiler.trace() for device-side timing: captures Perfetto traces and
extracts jit_*() event durations, which measure only the actual TPU kernel
execution time (excluding host dispatch, Python overhead, etc.).

Falls back to wall-clock timing only for _skip_jit eager workloads (2 of 50).

Usage (on TPU VM):
    PJRT_DEVICE=TPU python3 benchmark/run_all.py

Output: benchmark/results.json + benchmark/results.csv + summary table to stdout.
"""

import csv
import gzip
import importlib
import json
import os
import shutil
import sys
import time
import glob

import jax
import jax.numpy as jnp
import numpy as np

# TPU v6e peak TFLOPS (bf16, single chip, both MXUs)
# Source: Google Cloud TPU v6e (Trillium) spec — 918 TFLOPS bf16 per chip
# (each chip has 2x MXU with 256x256 systolic arrays)
TPU_PEAK_TFLOPS_BF16 = 918.0

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BENCHMARK_DIR))  # project root

NUM_WARMUP = 5
NUM_ITERS = 50


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


def extract_device_times_from_trace(trace_dir, num_iters, is_eager=False):
    """Parse Perfetto JSON trace and extract per-iteration device kernel times.

    For JIT workloads: matches jit_workload(...) events which wrap the complete
    device execution per iteration (including all sub-fusions/Pallas kernels).

    For eager (_skip_jit) workloads: collects all jit_*() device events and
    groups them into per-iteration batches by dividing total events by num_iters.

    All workloads use jax.named_scope('bench_kernel') to annotate the computation.

    Returns list of durations in milliseconds, or None if no events found.
    """
    perfetto_files = glob.glob(f"{trace_dir}/**/perfetto_trace.json.gz", recursive=True)
    if not perfetto_files:
        return None

    with gzip.open(perfetto_files[0], 'rt') as f:
        data = json.load(f)

    events = data.get('traceEvents', data) if isinstance(data, dict) else data

    if not is_eager:
        # JIT workloads: match jit_workload(...) or jit_<fn>(<hash>) wrapper events
        kernel_times_ms = []
        for e in events:
            if not isinstance(e, dict) or e.get('dur', 0) <= 0:
                continue
            name = e.get('name', '')
            if name.startswith('jit_') and '(' in name:
                kernel_times_ms.append(e['dur'] / 1000.0)
        return kernel_times_ms if kernel_times_ms else None
    else:
        # Eager workloads: collect all individual jit device ops and group per iteration
        all_jit_times = []
        for e in events:
            if not isinstance(e, dict) or e.get('dur', 0) <= 0:
                continue
            name = e.get('name', '')
            if name.startswith('jit_') and '(' in name:
                all_jit_times.append(e['dur'] / 1000.0)

        if not all_jit_times:
            return None

        # Group into per-iteration batches
        ops_per_iter = len(all_jit_times) // num_iters if num_iters > 0 else 0
        if ops_per_iter <= 0:
            return None

        iter_times = []
        for i in range(num_iters):
            batch = all_jit_times[i * ops_per_iter:(i + 1) * ops_per_iter]
            iter_times.append(sum(batch))
        return iter_times


def run_one(workload_dir, variant='baseline'):
    """Run one workload variant with profiler-based timing. Returns result dict."""
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
        # Dynamic import — register in sys.modules so dataclasses can resolve __module__
        spec = importlib.util.spec_from_file_location(
            f'{name}.{variant}', module_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
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

        # Get FLOP count: prefer module's get_flops(), fall back to XLA cost_analysis
        workload_fn = getattr(mod, 'workload')
        xla_flops = 0
        get_flops_fn = getattr(mod, 'get_flops', None)
        if get_flops_fn is not None:
            xla_flops = int(get_flops_fn())
        elif not skip_jit:
            xla_flops = get_xla_flops(workload_fn, inputs)
        result['xla_flops'] = xla_flops

        # Benchmark — all workloads use jax.profiler for device-side timing
        num_bench_iters = 10 if skip_jit else NUM_ITERS
        run_fn = workload_fn if skip_jit else jax.jit(workload_fn)

        # Warmup
        for _ in range(NUM_WARMUP):
            out = run_fn(*inputs)
            if hasattr(out, 'block_until_ready'):
                out.block_until_ready()

        # Wall-clock timing (for comparison)
        wall_times = []
        for _ in range(num_bench_iters):
            t0 = time.perf_counter()
            out = run_fn(*inputs)
            if hasattr(out, 'block_until_ready'):
                out.block_until_ready()
            wall_times.append((time.perf_counter() - t0) * 1000)

        # Device-side profiler timing via jax.profiler.trace()
        trace_dir = f'/tmp/jax_profile_{name}_{variant}'
        if os.path.exists(trace_dir):
            shutil.rmtree(trace_dir)
        os.makedirs(trace_dir, exist_ok=True)

        with jax.profiler.trace(trace_dir, create_perfetto_link=False, create_perfetto_trace=True):
            for _ in range(num_bench_iters):
                with jax.named_scope('bench_kernel'):
                    out = run_fn(*inputs)
                if hasattr(out, 'block_until_ready'):
                    out.block_until_ready()

        device_times = extract_device_times_from_trace(
            trace_dir, num_bench_iters, is_eager=skip_jit
        )
        shutil.rmtree(trace_dir, ignore_errors=True)

        if device_times and len(device_times) >= num_bench_iters:
            times = device_times[:num_bench_iters]
            timing_method = 'device_profiler'
        else:
            times = wall_times
            timing_method = 'wall_clock_fallback'

        result['wall_clock_median_ms'] = round(float(np.median(wall_times)), 4)

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
            'timing_method': timing_method,
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
    print(f"Timing: jax.profiler device-side (jit_*() events)")
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
    header = f"{'#':>3} {'Workload':<42} {'Var':<10} {'Median(ms)':>10} {'TFLOPS':>8} {'Util%':>7} {'Method':<18}"
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
                tflops_str = f"{r['tflops']:.1f}" if r['tflops'] > 0 else "—"
                util_str = f"{r['utilization_pct']:.1f}%" if r['utilization_pct'] > 0 else "—"
                method = r.get('timing_method', '?')
                print(f"{i:>3} {name:<42} {variant:<10} {r['median_ms']:>10.3f} {tflops_str:>8} {util_str:>7} {method:<18}")
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

    # Timing method summary
    profiler_count = sum(1 for r in successes if r.get('timing_method') == 'device_profiler')
    wall_count = sum(1 for r in successes if r.get('timing_method') in ('wall_clock', 'wall_clock_fallback'))
    print(f"\nTiming: {profiler_count} device_profiler, {wall_count} wall_clock")

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

    # Save results JSON
    output = {
        'metadata': {
            'jax_version': jax.__version__,
            'devices': str(jax.devices()),
            'tpu_peak_tflops_bf16': TPU_PEAK_TFLOPS_BF16,
            'timing_method': 'jax.profiler device-side (jit_*() events from Perfetto trace)',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'num_workloads': len(workload_dirs),
            'num_succeeded': len(successes),
            'num_warmup': NUM_WARMUP,
            'num_iters': NUM_ITERS,
        },
        'results': results,
    }
    output_path = os.path.join(BENCHMARK_DIR, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save results CSV (sorted by workload number)
    csv_path = os.path.join(BENCHMARK_DIR, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['workload', 'variant', 'status', 'timing_method',
                         'median_ms', 'mean_ms', 'std_ms', 'min_ms',
                         'xla_flops', 'tflops', 'utilization_pct',
                         'output_shape', 'error'])

        def sort_key(r):
            num = ''
            for c in r['name']:
                if c.isdigit():
                    num += c
                else:
                    break
            return int(num) if num else 999

        for r in sorted(results, key=sort_key):
            writer.writerow([
                r['name'], r['variant'], r['status'],
                r.get('timing_method', ''),
                r.get('median_ms', ''), r.get('mean_ms', ''),
                r.get('std_ms', ''), r.get('min_ms', ''),
                r.get('xla_flops', ''), r.get('tflops', ''),
                r.get('utilization_pct', ''),
                str(r.get('output_shape', '')), r.get('error', '')
            ])
    print(f"CSV saved to {csv_path}")


if __name__ == '__main__':
    main()
