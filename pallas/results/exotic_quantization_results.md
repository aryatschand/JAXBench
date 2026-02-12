# Exotic Quantization Format Benchmark Results

## Summary

Tested 11 non-standard quantization formats on TPU v5e to find which exotic formats provide speedups over FP32 baseline.

## Key Finding

**Simple integer quantization (INT3/5/6/8) dominates** - exotic formats with custom decode operations (table lookups, power operations) add overhead that negates the compression benefits.

## Results at 8192x8192 (Large Scale)

| Format | Bits | Compression | Time (ms) | Speedup | Accuracy | Notes |
|--------|------|-------------|-----------|---------|----------|-------|
| **INT5** | 5 | 6.4x | 0.92 | **3.40x** | 0.989 | Best overall |
| **INT6** | 6 | 5.3x | 0.93 | **3.40x** | 0.998 | High accuracy |
| **INT3** | 3 | 10.7x | 0.93 | **3.39x** | 0.785 | Max compression |
| Residual | 8 | 4.0x | 2.07 | **1.52x** | 0.9997 | Two-level quant |
| Posit8 | 8 | 4.0x | 3.05 | 1.03x | 0.9999 | Tapered precision |
| LNS | 8 | 4.0x | 3.19 | 0.99x | 0.796 | Log system |
| POW2 | 4 | 8.0x | 3.21 | 0.98x | 0.980 | Power-of-2 |
| FP E2M1 | 4 | 8.0x | 3.87 | 0.81x | 0.950 | Custom 4-bit float |
| FP E1M2 | 4 | 8.0x | 4.00 | 0.79x | 0.952 | Precision variant |
| BFP | 4 | 7.9x | 10.80 | 0.29x | 0.968 | Block floating point |
| FP32 | 32 | 1.0x | 3.14 | 1.00x | 1.000 | Baseline |

## Results at 4096x4096 (Medium Scale)

| Format | Speedup | Accuracy |
|--------|---------|----------|
| INT3/5/6 | 1.34-1.36x | 0.78-0.99 |
| Residual | 0.71x | 0.9997 |
| FP E2M1 | 0.48x | 0.950 |

## Scaling Behavior

| Matrix Size | INT8 Speedup | BF16 Speedup |
|-------------|--------------|--------------|
| 4096x4096 | 1.41x | 1.06x |
| 8192x8192 | **3.21x** | **1.85x** |

**Key insight**: Speedups increase dramatically with matrix size because memory bandwidth becomes the bottleneck.

## Why Simple INT Quantization Wins

1. **Native TPU support**: TPU has optimized INT8 matrix units
2. **No decode overhead**: Single scale multiply at the end
3. **Memory bandwidth**: 4x less data to transfer vs FP32
4. **Simple computation**: No table lookups or power operations

## Why Exotic Formats Are Slower

1. **Table lookup overhead**: FP E2M1, POW2 require per-element table lookups
2. **Reshape overhead**: BFP requires block reshape operations
3. **Transcendental functions**: LNS requires power operations in decode
4. **Two-pass**: Most exotic formats decode to FP32 then matmul, doubling memory traffic

## Recommendations

1. **For maximum speed**: INT5/INT6 (3.4x speedup, 0.99+ accuracy)
2. **For accuracy with speed**: Scaled INT8 (3.2x speedup, 0.9998 accuracy)
3. **For extreme compression**: INT3 (10.7x compression, 3.4x speedup)
4. **For near-lossless**: BF16 (1.85x speedup, 1.0 accuracy)

## Potential for Pallas Optimization

The exotic formats could benefit from Pallas fusion (decode + matmul in one kernel), but:
- Pallas kernels cannot capture JAX arrays as constants
- Need to pass lookup tables as inputs
- Current Pallas implementations show high overhead

Future work: Implement fused decode+matmul Pallas kernels that pass lookup tables properly.
