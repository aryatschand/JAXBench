#!/usr/bin/env python3
"""
Run Exotic Quantization Benchmarks on TPU

Standalone script to benchmark all exotic quantization formats.
Run this directly on the TPU VM.

Usage:
    python run_exotic_tpu.py [matrix_size]

Example:
    python run_exotic_tpu.py 4096
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import time
import sys


# ============================================================================
# Common utilities
# ============================================================================

WARMUP_ITERS = 5
BENCHMARK_ITERS = 20


def cosine_similarity(a, b):
    """Compute cosine similarity between two arrays."""
    a_flat = a.flatten().astype(jnp.float32)
    b_flat = b.flatten().astype(jnp.float32)
    return jnp.dot(a_flat, b_flat) / (jnp.linalg.norm(a_flat) * jnp.linalg.norm(b_flat))


def benchmark_fn(fn, *args, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark a function with warmup."""
    for _ in range(warmup):
        result = fn(*args)
        result.block_until_ready()

    start = time.perf_counter()
    for _ in range(iters):
        result = fn(*args)
        result.block_until_ready()
    end = time.perf_counter()

    return (end - start) / iters * 1000, result


# ============================================================================
# INT3 - 3-bit Integer
# ============================================================================

def run_int3_benchmark(size):
    """INT3: 3-bit signed integers [-4, 3]"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)

    # Quantize to INT3 range
    x_scale = jnp.max(jnp.abs(x_fp32)) / 3.0
    w_scale = jnp.max(jnp.abs(w_fp32)) / 3.0

    x_int3 = jnp.clip(jnp.round(x_fp32 / x_scale), -4, 3).astype(jnp.int8)
    w_int3 = jnp.clip(jnp.round(w_fp32 / w_scale), -4, 3).astype(jnp.int8)

    def matmul(x, w):
        return jnp.dot(x.astype(jnp.int32), w.astype(jnp.int32))

    jax_fn = jax.jit(matmul)
    jax_time, jax_result = benchmark_fn(jax_fn, x_int3, w_int3)

    ref_result = jnp.dot(x_fp32, w_fp32)
    jax_dequant = jax_result.astype(jnp.float32) * x_scale * w_scale
    accuracy = float(cosine_similarity(jax_dequant, ref_result))

    # Compression: FP32 (32 bits) -> INT3 packed (3 bits) = 10.67x
    compression = 32 / 3

    return {
        'format': 'INT3',
        'bits': 3,
        'compression': compression,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': '3-bit signed [-4,3]'
    }


# ============================================================================
# INT5 - 5-bit Integer
# ============================================================================

def run_int5_benchmark(size):
    """INT5: 5-bit signed integers [-16, 15]"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)

    x_scale = jnp.max(jnp.abs(x_fp32)) / 15.0
    w_scale = jnp.max(jnp.abs(w_fp32)) / 15.0

    x_int5 = jnp.clip(jnp.round(x_fp32 / x_scale), -16, 15).astype(jnp.int8)
    w_int5 = jnp.clip(jnp.round(w_fp32 / w_scale), -16, 15).astype(jnp.int8)

    def matmul(x, w):
        return jnp.dot(x.astype(jnp.int32), w.astype(jnp.int32))

    jax_fn = jax.jit(matmul)
    jax_time, jax_result = benchmark_fn(jax_fn, x_int5, w_int5)

    ref_result = jnp.dot(x_fp32, w_fp32)
    jax_dequant = jax_result.astype(jnp.float32) * x_scale * w_scale
    accuracy = float(cosine_similarity(jax_dequant, ref_result))

    compression = 32 / 5

    return {
        'format': 'INT5',
        'bits': 5,
        'compression': compression,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': '5-bit signed [-16,15]'
    }


# ============================================================================
# INT6 - 6-bit Integer
# ============================================================================

def run_int6_benchmark(size):
    """INT6: 6-bit signed integers [-32, 31]"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)

    x_scale = jnp.max(jnp.abs(x_fp32)) / 31.0
    w_scale = jnp.max(jnp.abs(w_fp32)) / 31.0

    x_int6 = jnp.clip(jnp.round(x_fp32 / x_scale), -32, 31).astype(jnp.int8)
    w_int6 = jnp.clip(jnp.round(w_fp32 / w_scale), -32, 31).astype(jnp.int8)

    def matmul(x, w):
        return jnp.dot(x.astype(jnp.int32), w.astype(jnp.int32))

    jax_fn = jax.jit(matmul)
    jax_time, jax_result = benchmark_fn(jax_fn, x_int6, w_int6)

    ref_result = jnp.dot(x_fp32, w_fp32)
    jax_dequant = jax_result.astype(jnp.float32) * x_scale * w_scale
    accuracy = float(cosine_similarity(jax_dequant, ref_result))

    compression = 32 / 6

    return {
        'format': 'INT6',
        'bits': 6,
        'compression': compression,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': '6-bit signed [-32,31]'
    }


# ============================================================================
# FP E2M1 - 4-bit Custom Float
# ============================================================================

FP_E2M1_TABLE = jnp.array([
    0.0, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
    -0.0, -0.0, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0
], dtype=jnp.float32)


def run_fp_e2m1_benchmark(size):
    """FP E2M1: 4-bit float (1 sign, 2 exp, 1 mantissa)"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)

    # Scale and quantize
    x_scale = jnp.max(jnp.abs(x_fp32)) / 3.0
    w_scale = jnp.max(jnp.abs(w_fp32)) / 3.0

    def quantize(x, scale):
        x_scaled = x / scale
        distances = jnp.abs(x_scaled.flatten()[:, None] - FP_E2M1_TABLE[None, :])
        indices = jnp.argmin(distances, axis=1).astype(jnp.uint8)
        return indices.reshape(x.shape), scale

    x_enc, _ = quantize(x_fp32, x_scale)
    w_enc, _ = quantize(w_fp32, w_scale)

    def matmul(x_e, w_e, xs, ws):
        x_dec = FP_E2M1_TABLE[x_e] * xs
        w_dec = FP_E2M1_TABLE[w_e] * ws
        return jnp.dot(x_dec, w_dec)

    jax_fn = jax.jit(lambda xe, we: matmul(xe, we, x_scale, w_scale))
    jax_time, jax_result = benchmark_fn(jax_fn, x_enc, w_enc)

    ref_result = jnp.dot(x_fp32, w_fp32)
    accuracy = float(cosine_similarity(jax_result, ref_result))

    compression = 8.0  # 32 bits -> 4 bits

    return {
        'format': 'FP E2M1',
        'bits': 4,
        'compression': compression,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': '4-bit: 1s+2e+1m'
    }


# ============================================================================
# FP E1M2 - 4-bit Custom Float (High Precision)
# ============================================================================

FP_E1M2_TABLE = jnp.array([
    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
    -0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75
], dtype=jnp.float32)


def run_fp_e1m2_benchmark(size):
    """FP E1M2: 4-bit float (1 sign, 1 exp, 2 mantissa) - high precision"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)

    x_scale = jnp.max(jnp.abs(x_fp32)) / 1.75
    w_scale = jnp.max(jnp.abs(w_fp32)) / 1.75

    def quantize(x, scale):
        x_scaled = x / scale
        distances = jnp.abs(x_scaled.flatten()[:, None] - FP_E1M2_TABLE[None, :])
        indices = jnp.argmin(distances, axis=1).astype(jnp.uint8)
        return indices.reshape(x.shape), scale

    x_enc, _ = quantize(x_fp32, x_scale)
    w_enc, _ = quantize(w_fp32, w_scale)

    def matmul(x_e, w_e, xs, ws):
        x_dec = FP_E1M2_TABLE[x_e] * xs
        w_dec = FP_E1M2_TABLE[w_e] * ws
        return jnp.dot(x_dec, w_dec)

    jax_fn = jax.jit(lambda xe, we: matmul(xe, we, x_scale, w_scale))
    jax_time, jax_result = benchmark_fn(jax_fn, x_enc, w_enc)

    ref_result = jnp.dot(x_fp32, w_fp32)
    accuracy = float(cosine_similarity(jax_result, ref_result))

    return {
        'format': 'FP E1M2',
        'bits': 4,
        'compression': 8.0,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': '4-bit: 1s+1e+2m (precision)'
    }


# ============================================================================
# LNS - Logarithmic Number System
# ============================================================================

LOG_FRAC_BITS = 4
LOG_BIAS = 8


def run_lns_benchmark(size):
    """LNS: Logarithmic Number System (multiplication becomes addition)"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.uniform(key1, (M, K), dtype=jnp.float32, minval=0.1, maxval=10.0)
    w_fp32 = jax.random.uniform(key2, (K, N), dtype=jnp.float32, minval=0.1, maxval=10.0)

    # Add some negative values
    signs_x = jax.random.choice(key1, jnp.array([-1.0, 1.0]), (M, K))
    signs_w = jax.random.choice(key2, jnp.array([-1.0, 1.0]), (K, N))
    x_fp32 = x_fp32 * signs_x
    w_fp32 = w_fp32 * signs_w

    def encode_lns(x):
        sign = (x < 0).astype(jnp.uint8)
        abs_x = jnp.abs(x)
        log2_x = jnp.log2(abs_x + 1e-10)
        log_quantized = jnp.round((log2_x + LOG_BIAS) * (1 << LOG_FRAC_BITS))
        log_quantized = jnp.clip(log_quantized, 0, 127).astype(jnp.uint8)
        is_zero = abs_x < 1e-8
        log_quantized = jnp.where(is_zero, 0, log_quantized)
        return (sign << 7) | log_quantized

    def decode_lns(enc):
        sign = (enc >> 7) & 0x01
        log_q = enc & 0x7F
        log2_x = log_q.astype(jnp.float32) / (1 << LOG_FRAC_BITS) - LOG_BIAS
        abs_val = jnp.power(2.0, log2_x)
        sign_mult = jnp.where(sign == 1, -1.0, 1.0)
        return jnp.where(log_q == 0, 0.0, sign_mult * abs_val)

    x_enc = encode_lns(x_fp32)
    w_enc = encode_lns(w_fp32)

    def matmul(x_e, w_e):
        x_dec = decode_lns(x_e)
        w_dec = decode_lns(w_e)
        return jnp.dot(x_dec, w_dec)

    jax_fn = jax.jit(matmul)
    jax_time, jax_result = benchmark_fn(jax_fn, x_enc, w_enc)

    ref_result = jnp.dot(x_fp32, w_fp32)
    accuracy = float(cosine_similarity(jax_result, ref_result))

    return {
        'format': 'LNS',
        'bits': 8,
        'compression': 4.0,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': 'Log system (mul->add)'
    }


# ============================================================================
# BFP - Block Floating Point
# ============================================================================

BLOCK_SIZE = 16


def run_bfp_benchmark(size):
    """BFP: Block Floating Point (shared exponent per block)"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32) * 10.0
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32) * 10.0

    def encode_bfp(x):
        num_blocks_m = M // BLOCK_SIZE
        num_blocks_k = K // BLOCK_SIZE

        x_blocks = x.reshape(num_blocks_m, BLOCK_SIZE, num_blocks_k, BLOCK_SIZE)
        x_blocks = x_blocks.transpose(0, 2, 1, 3)

        block_max = jnp.max(jnp.abs(x_blocks), axis=(2, 3), keepdims=True)
        block_max = jnp.maximum(block_max, 1e-10)

        exponents = jnp.ceil(jnp.log2(block_max)).astype(jnp.int8)
        exponents = jnp.clip(exponents, -127, 127)

        scale = jnp.power(2.0, -exponents.astype(jnp.float32)) * 7.0
        mantissas = jnp.clip(jnp.round(x_blocks * scale), -8, 7).astype(jnp.int8)

        mantissas = mantissas.transpose(0, 2, 1, 3).reshape(M, K)
        exponents = exponents.squeeze((2, 3))

        return mantissas, exponents

    def decode_bfp(mantissas, exponents):
        num_blocks_m = M // BLOCK_SIZE
        num_blocks_k = K // BLOCK_SIZE

        m_blocks = mantissas.reshape(num_blocks_m, BLOCK_SIZE, num_blocks_k, BLOCK_SIZE)
        m_blocks = m_blocks.transpose(0, 2, 1, 3).astype(jnp.float32)

        exp_expanded = exponents[:, :, None, None]
        scale = jnp.power(2.0, exp_expanded.astype(jnp.float32)) / 7.0
        decoded = m_blocks * scale

        return decoded.transpose(0, 2, 1, 3).reshape(M, K)

    m_x, e_x = encode_bfp(x_fp32)
    m_w, e_w = encode_bfp(w_fp32)

    def matmul(mx, ex, mw, ew):
        x_dec = decode_bfp(mx, ex)
        w_dec = decode_bfp(mw, ew)
        return jnp.dot(x_dec, w_dec)

    jax_fn = jax.jit(matmul)
    jax_time, jax_result = benchmark_fn(jax_fn, m_x, e_x, m_w, e_w)

    ref_result = jnp.dot(x_fp32, w_fp32)
    accuracy = float(cosine_similarity(jax_result, ref_result))

    bits_per_element = 4 + 8 / (BLOCK_SIZE * BLOCK_SIZE)
    compression = 32 / bits_per_element

    return {
        'format': 'BFP',
        'bits': bits_per_element,
        'compression': compression,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': f'{BLOCK_SIZE}x{BLOCK_SIZE} block, shared exp'
    }


# ============================================================================
# POW2 - Power-of-2 Quantization
# ============================================================================

POW2_TABLE = jnp.concatenate([
    jnp.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]),
    -jnp.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
])


def run_pow2_benchmark(size):
    """POW2: Power-of-2 weights (multiply becomes shift)"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)

    w_scale = jnp.max(jnp.abs(w_fp32)) / 16.0
    w_scaled = w_fp32 / w_scale

    distances = jnp.abs(w_scaled.flatten()[:, None] - POW2_TABLE[None, :])
    w_enc = jnp.argmin(distances, axis=1).astype(jnp.uint8).reshape(K, N)

    def matmul(x, w_e, ws):
        w_dec = POW2_TABLE[w_e] * ws
        return jnp.dot(x, w_dec)

    jax_fn = jax.jit(lambda x, we: matmul(x, we, w_scale))
    jax_time, jax_result = benchmark_fn(jax_fn, x_fp32, w_enc)

    ref_result = jnp.dot(x_fp32, w_fp32)
    accuracy = float(cosine_similarity(jax_result, ref_result))

    return {
        'format': 'POW2',
        'bits': 4,
        'compression': 8.0,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': 'Weights: powers of 2'
    }


# ============================================================================
# Residual - Two-Level Quantization
# ============================================================================

def run_residual_benchmark(size):
    """Residual: W = W_coarse + W_fine (4-bit each)"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)

    def encode_residual(x):
        scale_c = jnp.max(jnp.abs(x)) / 7.0
        scale_c = jnp.maximum(scale_c, 1e-8)
        x_coarse = jnp.clip(jnp.round(x / scale_c), -8, 7).astype(jnp.int8)

        residual = x - x_coarse.astype(jnp.float32) * scale_c
        scale_f = jnp.max(jnp.abs(residual)) / 7.0
        scale_f = jnp.maximum(scale_f, 1e-10)
        x_fine = jnp.clip(jnp.round(residual / scale_f), -8, 7).astype(jnp.int8)

        return x_coarse, x_fine, scale_c, scale_f

    x_c, x_f, x_sc, x_sf = encode_residual(x_fp32)
    w_c, w_f, w_sc, w_sf = encode_residual(w_fp32)

    def matmul(xc, xf, wc, wf, x_sc, x_sf, w_sc, w_sf):
        x = xc.astype(jnp.float32) * x_sc + xf.astype(jnp.float32) * x_sf
        w = wc.astype(jnp.float32) * w_sc + wf.astype(jnp.float32) * w_sf
        return jnp.dot(x, w)

    jax_fn = jax.jit(lambda xc, xf, wc, wf: matmul(xc, xf, wc, wf, x_sc, x_sf, w_sc, w_sf))
    jax_time, jax_result = benchmark_fn(jax_fn, x_c, x_f, w_c, w_f)

    ref_result = jnp.dot(x_fp32, w_fp32)
    accuracy = float(cosine_similarity(jax_result, ref_result))

    return {
        'format': 'Residual',
        'bits': 8,
        'compression': 4.0,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': 'Two-level (4+4 bits)'
    }


# ============================================================================
# Posit8 - Tapered Precision
# ============================================================================

def run_posit_benchmark(size):
    """Posit8: Tapered precision (more near 1.0)"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)

    # Simplified posit-like encoding
    x_scale = jnp.max(jnp.abs(x_fp32))
    w_scale = jnp.max(jnp.abs(w_fp32))

    def encode_posit(x, scale):
        x_norm = x / scale
        sign = (x_norm < 0).astype(jnp.uint8)
        abs_x = jnp.clip(jnp.abs(x_norm), 0, 1)

        # Tapered: more precision near 0.5
        # Use sqrt for more levels near center
        encoded_mag = jnp.round(jnp.sqrt(abs_x) * 127).astype(jnp.uint8)
        encoded_mag = jnp.clip(encoded_mag, 0, 127)

        return (sign << 7) | encoded_mag

    def decode_posit(enc, scale):
        sign = (enc >> 7) & 0x01
        mag = enc & 0x7F

        # Reverse sqrt transform
        abs_val = (mag.astype(jnp.float32) / 127.0) ** 2
        sign_mult = jnp.where(sign == 1, -1.0, 1.0)

        return sign_mult * abs_val * scale

    x_enc = encode_posit(x_fp32, x_scale)
    w_enc = encode_posit(w_fp32, w_scale)

    def matmul(xe, we, xs, ws):
        x_dec = decode_posit(xe, xs)
        w_dec = decode_posit(we, ws)
        return jnp.dot(x_dec, w_dec)

    jax_fn = jax.jit(lambda xe, we: matmul(xe, we, x_scale, w_scale))
    jax_time, jax_result = benchmark_fn(jax_fn, x_enc, w_enc)

    ref_result = jnp.dot(x_fp32, w_fp32)
    accuracy = float(cosine_similarity(jax_result, ref_result))

    return {
        'format': 'Posit8',
        'bits': 8,
        'compression': 4.0,
        'time_ms': jax_time,
        'accuracy': accuracy,
        'notes': 'Tapered precision'
    }


# ============================================================================
# FP32 Baseline
# ============================================================================

def run_fp32_baseline(size):
    """FP32 baseline for comparison"""
    M, N = size
    K = M

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    x_fp32 = jax.random.normal(key1, (M, K), dtype=jnp.float32)
    w_fp32 = jax.random.normal(key2, (K, N), dtype=jnp.float32)

    def matmul(x, w):
        return jnp.dot(x, w)

    jax_fn = jax.jit(matmul)
    jax_time, jax_result = benchmark_fn(jax_fn, x_fp32, w_fp32)

    return {
        'format': 'FP32',
        'bits': 32,
        'compression': 1.0,
        'time_ms': jax_time,
        'accuracy': 1.0,
        'notes': 'Baseline'
    }


# ============================================================================
# Main
# ============================================================================

def run_all_benchmarks(size=(4096, 4096)):
    """Run all exotic format benchmarks."""

    print("=" * 80)
    print("EXOTIC QUANTIZATION FORMAT BENCHMARK")
    print("=" * 80)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Matrix size: {size[0]}x{size[1]}")
    print()

    benchmarks = [
        ("FP32 Baseline", run_fp32_baseline),
        ("INT3", run_int3_benchmark),
        ("INT5", run_int5_benchmark),
        ("INT6", run_int6_benchmark),
        ("FP E2M1", run_fp_e2m1_benchmark),
        ("FP E1M2", run_fp_e1m2_benchmark),
        ("LNS", run_lns_benchmark),
        ("BFP", run_bfp_benchmark),
        ("POW2", run_pow2_benchmark),
        ("Residual", run_residual_benchmark),
        ("Posit8", run_posit_benchmark),
    ]

    results = []
    fp32_time = None

    for name, benchmark_fn in benchmarks:
        print(f"\nRunning {name}...", end=" ", flush=True)
        try:
            result = benchmark_fn(size)
            results.append(result)
            if name == "FP32 Baseline":
                fp32_time = result['time_ms']
            print(f"Done ({result['time_ms']:.2f} ms, acc={result['accuracy']:.4f})")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'format': name,
                'bits': 'N/A',
                'compression': 0,
                'time_ms': float('inf'),
                'accuracy': 0,
                'notes': f'Error: {str(e)[:30]}'
            })

    # Print summary table
    print("\n")
    print("=" * 95)
    print("EXOTIC FORMAT BENCHMARK RESULTS")
    print("=" * 95)
    print()
    print(f"{'Format':<12} | {'Bits':>5} | {'Compress':>8} | {'Time (ms)':>10} | {'Speedup':>7} | {'Accuracy':>8} | {'Notes':<25}")
    print("-" * 95)

    for r in results:
        bits = r['bits']
        if isinstance(bits, float):
            bits_str = f"{bits:.2f}"
        else:
            bits_str = str(bits)

        compression = r['compression']
        comp_str = f"{compression:.1f}x" if isinstance(compression, (int, float)) and compression > 0 else "N/A"

        time_ms = r['time_ms']
        time_str = f"{time_ms:.2f}" if time_ms < float('inf') else "N/A"

        if fp32_time and time_ms < float('inf'):
            speedup = fp32_time / time_ms
            speed_str = f"{speedup:.2f}x"
        else:
            speed_str = "N/A"

        accuracy = r['accuracy']
        acc_str = f"{accuracy:.4f}" if accuracy > 0 else "N/A"

        notes = r.get('notes', '')[:25]

        print(f"{r['format']:<12} | {bits_str:>5} | {comp_str:>8} | {time_str:>10} | {speed_str:>7} | {acc_str:>8} | {notes:<25}")

    print("-" * 95)

    # Analysis
    print("\n")
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    successful = [r for r in results if r['accuracy'] > 0 and r['format'] != 'FP32']

    if successful:
        best_acc = max(successful, key=lambda x: x['accuracy'])
        print(f"\nBest Accuracy: {best_acc['format']} ({best_acc['accuracy']:.4f})")

        best_comp = max(successful, key=lambda x: x['compression'])
        print(f"Best Compression: {best_comp['format']} ({best_comp['compression']:.1f}x)")

        if fp32_time:
            best_speed = min(successful, key=lambda x: x['time_ms'])
            speedup = fp32_time / best_speed['time_ms']
            print(f"Fastest: {best_speed['format']} ({speedup:.2f}x vs FP32)")

    return results


if __name__ == "__main__":
    size = (4096, 4096)
    if len(sys.argv) > 1:
        try:
            dim = int(sys.argv[1])
            size = (dim, dim)
        except ValueError:
            pass

    results = run_all_benchmarks(size)
