"""Simple block-size autotuner for Pallas kernels.

Runs on the TPU VM. For each kernel, tries all valid block-size combinations
and reports the best. With --apply, writes the best TUNED_PARAMS dict directly
into the kernel files.

Usage:
    python autotune_block_sizes.py [--apply] [kernel_name ...]
    python autotune_block_sizes.py                           # tune all, report only
    python autotune_block_sizes.py --apply                   # tune all and update files
    python autotune_block_sizes.py --apply flash_attention   # tune and update one
"""
import importlib.util
import itertools
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp


NUM_WARMUP = 5
NUM_TRIALS = 50

KERNELS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _divisors(n, candidates):
    return [c for c in candidates if n % c == 0 and c <= n]


def _time_fn(fn, inputs, num_warmup=NUM_WARMUP, num_trials=NUM_TRIALS):
    for _ in range(num_warmup):
        out = fn(*inputs)
        jax.block_until_ready(out)
    times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        out = fn(*inputs)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _try_config(make_fn, inputs, label):
    try:
        fn = jax.jit(make_fn())
        latency = _time_fn(fn, inputs)
        print(f"  {label}: {latency:.3f} ms")
        return latency
    except Exception as e:
        err = str(e)[:80]
        print(f"  {label}: FAIL ({err})")
        return None


# ── Per-kernel tuning definitions ────────────────────────────────────────────

def tune_flash_attention(mod):
    inputs = mod.create_inputs()
    S = mod.CONFIG['seq_len']

    power2 = [128, 256, 512, 1024, 2048]
    block_q_opts = _divisors(S, power2)
    block_k_major_opts = _divisors(S, power2)
    block_k_opts = _divisors(S, [128, 256, 512])

    results = []
    for bq, bkm, bk in itertools.product(block_q_opts, block_k_major_opts, block_k_opts):
        if bk > bkm:
            continue
        if bkm % bk != 0:
            continue
        label = f"block_q={bq}, block_k_major={bkm}, block_k={bk}"
        bs = mod.BlockSizes(
            block_q=bq, block_k_major=bkm, block_k=bk, block_b=1,
            block_q_major_dkv=128, block_k_major_dkv=128, block_k_dkv=128,
            block_q_dkv=128, block_k_major_dq=128, block_k_dq=128, block_q_dq=128,
        )
        sm_scale = 1.0 / math.sqrt(mod.CONFIG['head_dim'])
        def make_fn(bs=bs, sm_scale=sm_scale):
            return lambda q, k, v: mod.flash_attention(
                q, k, v, causal=True, sm_scale=sm_scale, block_sizes=bs,
            )
        latency = _try_config(make_fn, inputs, label)
        if latency is not None:
            results.append((latency, label, {
                'block_q': bq, 'block_k_major': bkm, 'block_k': bk,
                'block_b': 1,
                'block_q_major_dkv': 128, 'block_k_major_dkv': 128,
                'block_k_dkv': 128, 'block_q_dkv': 128,
                'block_k_major_dq': 128, 'block_k_dq': 128, 'block_q_dq': 128,
            }))
    return results


def tune_splash_attention(mod):
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
    inputs = mod.create_inputs()
    S = mod.CONFIG['seq_len']
    H_q = mod.CONFIG['num_query_heads']

    power2 = [128, 256, 512, 1024, 2048]
    block_q_opts = _divisors(S, power2)
    block_kv_opts = _divisors(S, power2)
    block_kv_compute_ratios = [1, 2]  # block_kv_compute = block_kv // ratio
    layout_opts = [1, 2]  # HEAD_DIM_MINOR=1, SEQ_MINOR=2

    results = []
    for bq, bkv in itertools.product(block_q_opts, block_kv_opts):
        for bkvc_ratio in block_kv_compute_ratios:
            bkvc = bkv // bkvc_ratio
            if bkvc < 128:
                continue
            for q_layout, k_layout, v_layout in [(1,1,1), (2,2,2)]:
                label = f"block_q={bq}, block_kv={bkv}, block_kv_compute={bkvc}, layout={q_layout}"
                bs = mod.BlockSizes(
                    block_q=bq, block_kv=bkv, block_kv_compute=bkvc,
                    q_layout=mod.QKVLayout(q_layout),
                    k_layout=mod.QKVLayout(k_layout),
                    v_layout=mod.QKVLayout(v_layout),
                )
                mask = mask_lib.CausalMask(shape=(S, S))
                multi_head_mask = mask_lib.MultiHeadMask([mask] * H_q)
                def make_fn(bs=bs, mhm=multi_head_mask):
                    kernel = mod._make_splash_attention(
                        mhm, block_sizes=bs, is_mqa=False, head_shards=1, q_seq_shards=1,
                    )
                    @jax.vmap
                    def attend(q, k, v):
                        H_kv = v.shape[0]
                        hpg = q.shape[0] // H_kv
                        return kernel(q, jnp.repeat(k, hpg, axis=0), jnp.repeat(v, hpg, axis=0))
                    return attend
                latency = _try_config(make_fn, inputs, label)
                if latency is not None:
                    results.append((latency, label, {
                        'block_q': bq, 'block_kv': bkv, 'block_kv_compute': bkvc,
                        'q_layout': q_layout, 'k_layout': k_layout, 'v_layout': v_layout,
                        'head_shards': 1, 'q_seq_shards': 1,
                        'block_q_dkv': None, 'block_kv_dkv': None,
                        'block_kv_dkv_compute': None,
                        'block_q_dq': None, 'block_kv_dq': None,
                    }))
    return results


def tune_matmul(mod):
    inputs = mod.create_inputs()
    M, K, N = mod.CONFIG['M'], mod.CONFIG['K'], mod.CONFIG['N']

    bm_opts = _divisors(M, [128, 256, 512, 1024, 2048])
    bn_opts = _divisors(N, [128, 256, 512, 1024, 2048])
    bk_opts = _divisors(K, [128, 256, 512, 1024])

    results = []
    for bm, bn, bk in itertools.product(bm_opts, bn_opts, bk_opts):
        label = f"block_shape=({bm},{bn}), block_k={bk}"
        def make_fn(bm=bm, bn=bn, bk=bk):
            return lambda x, y: mod.matmul(x, y, block_shape=(bm, bn), block_k=bk)
        latency = _try_config(make_fn, inputs, label)
        if latency is not None:
            results.append((latency, label, {'block_shape': (bm, bn), 'block_k': bk}))
    return results


def tune_megablox_gmm(mod):
    inputs = mod.create_inputs()

    tile_opts = [128, 256, 512]
    results = []
    for tm, tn, tk in itertools.product(tile_opts, tile_opts, tile_opts):
        label = f"tiling=({tm},{tn},{tk})"
        def make_fn(t=(tm, tn, tk)):
            return lambda lhs, rhs, gs: mod.gmm(lhs, rhs, gs, tiling=t)
        latency = _try_config(make_fn, inputs, label)
        if latency is not None:
            results.append((latency, label, {'tiling': (tm, tn, tk)}))
    return results


def tune_paged_attention(mod):
    inputs = mod.create_inputs()

    opts = [4, 8, 16, 32, 64]
    results = []
    for ppb in opts:
        label = f"pages_per_compute_block={ppb}"
        def make_fn(ppb=ppb):
            return lambda q, kp, vp, lens, pi: mod.paged_attention(
                q, kp, vp, lens, pi, pages_per_compute_block=ppb,
            )
        latency = _try_config(make_fn, inputs, label)
        if latency is not None:
            results.append((latency, label, {'pages_per_compute_block': ppb}))
    return results


def tune_ragged_paged_attention(mod):
    inputs = mod.create_inputs()
    head_dim = mod.CONFIG['head_dim']

    kv_opts = [4, 8, 16, 32]
    q_opts = [32, 64, 128, 256]
    results = []
    for nkv, nq in itertools.product(kv_opts, q_opts):
        label = f"num_kv_pages_per_block={nkv}, num_queries_per_block={nq}"
        sm_scale = 1.0 / math.sqrt(head_dim)
        def make_fn(nkv=nkv, nq=nq, sm=sm_scale):
            return lambda q, kvp, kvl, pi, cql, ns: mod.ragged_paged_attention(
                q, kvp, kvl, pi, cql, ns, sm_scale=sm,
                num_kv_pages_per_block=nkv, num_queries_per_block=nq,
                vmem_limit_bytes=32 * 1024 * 1024,
            )
        latency = _try_config(make_fn, inputs, label)
        if latency is not None:
            results.append((latency, label, {
                'num_kv_pages_per_block': nkv, 'num_queries_per_block': nq,
                'vmem_limit_bytes': 32 * 1024 * 1024,
            }))
    return results


TUNERS = {
    'flash_attention': tune_flash_attention,
    'splash_attention': tune_splash_attention,
    'matmul': tune_matmul,
    'megablox_gmm': tune_megablox_gmm,
    'paged_attention': tune_paged_attention,
    'ragged_paged_attention': tune_ragged_paged_attention,
}


# ── Apply tuned params via regex ─────────────────────────────────────────────

import re


def _apply_tuned_params(file_path, params):
    """Regex-replace the TUNED_PARAMS dict in a kernel file."""
    with open(file_path, 'r') as f:
        content = f.read()
    # Convert tuples to lists for consistent repr
    serializable = {}
    for k, v in params.items():
        serializable[k] = list(v) if isinstance(v, tuple) else v
    new_dict = repr(serializable)
    pattern = r'TUNED_PARAMS\s*=\s*\{[^}]*\}'
    if not re.search(pattern, content):
        print(f"    WARNING: no TUNED_PARAMS found in {file_path}")
        return
    new_content = re.sub(pattern, f'TUNED_PARAMS = {new_dict}', content)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print(f"    Updated TUNED_PARAMS in {os.path.basename(file_path)}")


def main():
    apply = '--apply' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--apply']
    kernels = args if args else list(TUNERS.keys())

    all_results = {}
    for name in kernels:
        if name not in TUNERS:
            print(f"Unknown kernel: {name}. Available: {list(TUNERS.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"Tuning: {name}")
        print(f"{'='*60}")

        mod_path = os.path.join(KERNELS_DIR, f"{name}.py")
        mod = _load_module(mod_path, f"pallas_{name}")

        results = TUNERS[name](mod)
        if not results:
            print(f"  No valid configurations found!")
            continue

        results.sort(key=lambda x: x[0])
        best_latency, best_label, best_params = results[0]

        print(f"\n  Best: {best_label} -> {best_latency:.3f} ms")
        if len(results) > 1:
            worst_latency = results[-1][0]
            print(f"  Worst: {results[-1][1]} -> {worst_latency:.3f} ms")
            print(f"  Range: {best_latency:.3f} - {worst_latency:.3f} ms ({worst_latency/best_latency:.2f}x)")

        all_results[name] = {
            'best_params': best_params,
            'best_latency': round(best_latency, 3),
            'all_results': [(round(l, 3), lbl) for l, lbl, _ in results],
        }

        if apply:
            print(f"  Applying to {name}.py ...")
            _apply_tuned_params(mod_path, best_params)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"  {name}: {res['best_latency']:.3f} ms  {res['best_params']}")

    out_path = os.path.join(KERNELS_DIR, "autotune_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=list)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
