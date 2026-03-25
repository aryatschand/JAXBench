"""Cross-check JAX reference implementations against Pallas kernels.

Usage (on TPU):
    python check_references.py [kernel_name ...]

If no kernel names given, checks all. Example:
    python check_references.py matmul flash_attention

When a local Pallas kernel file fails to import or run (e.g. version mismatch),
falls back to importing the kernel from the installed JAX package.
"""

import importlib.util
import math
import os
import sys
import time
import types

import jax
import jax.numpy as jnp

KERNELS_DIR = os.path.dirname(os.path.abspath(__file__))
REFS_DIR = os.path.join(KERNELS_DIR, "jax_references")

ALL_KERNELS = [
    "matmul",
    "flash_attention",
    "megablox_gmm",
    "splash_attention",
    "paged_attention",
    "ragged_paged_attention",
]

ATOL = 3.125e-2
RTOL = 1e-2


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_installed_pallas_module(kernel_name, ref_mod):
    """Build a module-like object that calls the installed JAX Pallas kernel.

    Uses CONFIG and create_inputs from ref_mod to define the workload shape,
    then wraps the installed JAX kernel function with the same interface.
    """
    cfg = ref_mod.CONFIG
    mod = types.ModuleType(f"installed_pallas_{kernel_name}")

    if kernel_name == "flash_attention":
        from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
        sm_scale = 1.0 / math.sqrt(cfg['head_dim'])
        def workload(q, k, v):
            return flash_attention(q, k, v, causal=True, sm_scale=sm_scale)
        mod.workload = workload

    elif kernel_name == "splash_attention":
        from jax.experimental.pallas.ops.tpu.splash_attention import (
            splash_attention_kernel as sak,
            splash_attention_mask as mask_lib,
        )
        H_q = cfg['num_query_heads']
        S = cfg['seq_len']
        H_kv = cfg['num_kv_heads']
        heads_per_group = H_q // H_kv
        mask = mask_lib.CausalMask(shape=(S, S))
        multi_head_mask = mask_lib.MultiHeadMask(
            [mask] * H_q
        )
        splash_kernel = sak.make_splash_mha_single_device(
            multi_head_mask, head_shards=1, q_seq_shards=1,
        )
        @jax.vmap
        def _attend(q_batch, k_batch, v_batch):
            k_repeated = jnp.repeat(k_batch, heads_per_group, axis=0)
            v_repeated = jnp.repeat(v_batch, heads_per_group, axis=0)
            return splash_kernel(q_batch, k_repeated, v_repeated)
        def workload(q, k, v):
            return _attend(q, k, v)
        mod.workload = workload

    elif kernel_name == "paged_attention":
        from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
        def workload(q, k_pages, v_pages, lengths, page_indices):
            return paged_attention(
                q, k_pages, v_pages, lengths, page_indices,
                pages_per_compute_block=cfg['pages_per_seq'],
            )
        mod.workload = workload

    elif kernel_name == "ragged_paged_attention":
        from jax.experimental.pallas.ops.tpu.ragged_paged_attention import (
            ragged_paged_attention,
        )
        sm_scale = 1.0 / math.sqrt(cfg['head_dim'])
        def workload(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
            return ragged_paged_attention(
                q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
                sm_scale=sm_scale,
                num_kv_pages_per_block=8,
                num_queries_per_block=64,
                vmem_limit_bytes=32 * 1024 * 1024,
            )
        mod.workload = workload

    elif kernel_name == "megablox_gmm":
        from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm
        def workload(lhs, rhs, group_sizes):
            return gmm(lhs, rhs, group_sizes)
        mod.workload = workload

    elif kernel_name == "matmul":
        from jax.experimental.pallas.ops.tpu.matmul import matmul
        def workload(x, y):
            return matmul(x, y, block_shape=(512, 512))
        mod.workload = workload

    else:
        return None

    return mod


def _run_fn(fn, inputs, skip_jit=False):
    """Run fn on inputs, trying jit first, then falling back to eager."""
    if not skip_jit:
        try:
            jit_fn = jax.jit(fn)
            out = jit_fn(*inputs)
            jax.block_until_ready(out)
            return out, jit_fn
        except Exception:
            pass
    out = fn(*inputs)
    jax.block_until_ready(out)
    return out, fn


def check_one(kernel_name):
    ref_path = os.path.join(REFS_DIR, f"{kernel_name}.py")
    pallas_path = os.path.join(KERNELS_DIR, f"{kernel_name}.py")

    if not os.path.exists(ref_path):
        return {"status": "SKIP", "reason": f"{ref_path} not found"}

    try:
        ref_mod = _load_module(ref_path, f"ref_{kernel_name}")
    except Exception as e:
        return {"status": "ERROR", "reason": f"ref import failed: {e}"}

    # Try loading local Pallas kernel file, fall back to installed JAX
    pallas_mod = None
    pallas_source = "local"
    if os.path.exists(pallas_path):
        try:
            pallas_mod = _load_module(pallas_path, f"pallas_{kernel_name}")
        except Exception:
            pass

    if pallas_mod is None:
        pallas_mod = _make_installed_pallas_module(kernel_name, ref_mod)
        pallas_source = "installed"

    if pallas_mod is None:
        return {"status": "SKIP", "reason": "no Pallas kernel available"}

    # Run reference
    try:
        inputs = ref_mod.create_inputs()
        skip_jit = getattr(ref_mod, '_skip_jit', False)
        ref_out, ref_fn = _run_fn(ref_mod.workload, inputs, skip_jit=skip_jit)
    except Exception as e:
        return {"status": "ERROR", "reason": f"JAX reference failed: {e}"}

    # Run Pallas kernel (try local, fall back to installed)
    try:
        skip_jit = getattr(pallas_mod, '_skip_jit', False)
        pallas_out, pallas_fn = _run_fn(pallas_mod.workload, inputs, skip_jit=skip_jit)
    except Exception as e:
        if pallas_source == "local":
            print(f"  local kernel failed ({e}), trying installed JAX...")
            try:
                pallas_mod = _make_installed_pallas_module(kernel_name, ref_mod)
            except Exception as e_install:
                return {"status": "ERROR", "reason": f"installed kernel setup failed: {e_install}"}
            pallas_source = "installed"
            if pallas_mod is None:
                return {"status": "ERROR", "reason": f"Pallas kernel failed: {e}"}
            try:
                pallas_out, pallas_fn = _run_fn(pallas_mod.workload, inputs)
            except Exception as e2:
                return {"status": "ERROR", "reason": f"installed Pallas kernel also failed: {e2}"}
        else:
            return {"status": "ERROR", "reason": f"Pallas kernel failed: {e}"}

    if ref_out.shape != pallas_out.shape:
        return {
            "status": "FAIL",
            "reason": f"shape mismatch: ref={ref_out.shape} vs pallas={pallas_out.shape}",
        }

    ref_f32 = ref_out.astype(jnp.float32) if ref_out.dtype != jnp.float32 else ref_out
    pallas_f32 = pallas_out.astype(jnp.float32) if pallas_out.dtype != jnp.float32 else pallas_out

    max_diff = float(jnp.max(jnp.abs(ref_f32 - pallas_f32)))
    denom = jnp.maximum(jnp.max(jnp.abs(ref_f32)), 1e-6)
    max_rel_diff = float(jnp.max(jnp.abs(ref_f32 - pallas_f32) / denom))

    cfg = getattr(ref_mod, 'CONFIG', {})
    atol = cfg.get('atol', ATOL)
    rtol = cfg.get('rtol', RTOL)

    if not jnp.allclose(ref_f32, pallas_f32, atol=atol, rtol=rtol):
        return {
            "status": "FAIL",
            "reason": f"correctness check failed (max_diff={max_diff:.6f}, max_rel_diff={max_rel_diff:.6f})",
        }

    # Quick benchmark
    for _ in range(3):
        jax.block_until_ready(ref_fn(*inputs))
        jax.block_until_ready(pallas_fn(*inputs))

    t0 = time.perf_counter()
    for _ in range(10):
        jax.block_until_ready(ref_fn(*inputs))
    ref_ms = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(10):
        jax.block_until_ready(pallas_fn(*inputs))
    pallas_ms = (time.perf_counter() - t0) / 10 * 1000

    return {
        "status": "PASS",
        "source": pallas_source,
        "max_diff": round(max_diff, 6),
        "max_rel_diff": round(max_rel_diff, 6),
        "ref_ms": round(ref_ms, 3),
        "pallas_ms": round(pallas_ms, 3),
        "speedup": round(ref_ms / pallas_ms, 2) if pallas_ms > 0 else float("inf"),
    }


def main():
    kernels = sys.argv[1:] if len(sys.argv) > 1 else ALL_KERNELS
    results = {}
    for name in kernels:
        print(f"\n{'='*60}")
        print(f"Checking: {name}")
        print(f"{'='*60}")
        result = check_one(name)
        results[name] = result
        if result["status"] == "PASS":
            src = result.get("source", "local")
            print(f"  PASS ({src})  max_diff={result['max_diff']:.6f}  max_rel_diff={result['max_rel_diff']:.6f}")
            print(f"  JAX ref: {result['ref_ms']:.3f} ms  |  Pallas: {result['pallas_ms']:.3f} ms  |  speedup: {result['speedup']:.2f}x")
        else:
            print(f"  {result['status']}: {result.get('reason', '')}")

    print(f"\n{'='*60}")
    print("Summary:")
    for name, r in results.items():
        src = f" ({r.get('source', '')})" if r.get("source") else ""
        print(f"  {name:30s} {r['status']}{src}")


if __name__ == "__main__":
    main()
