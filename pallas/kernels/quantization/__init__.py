"""Quantized matmul Pallas kernels.

Available formats:
- INT3/5/6/8: Simple integer quantization (uses JAX native path)
- FP E2M1: 4-bit custom float (1 sign, 2 exp, 1 mantissa)
- FP E1M2: 4-bit custom float (1 sign, 1 exp, 2 mantissa)
- LNS: Logarithmic number system
- BFP: Block floating point
- POW2: Power-of-2 weights
- Residual: Two-level quantization
- Posit8: Tapered precision

Key finding: Simple INT quantization via JAX native path is fastest.
Exotic formats with table lookups add decode overhead.
"""

SUPPORTED_FORMATS = [
    "INT3", "INT5", "INT6", "INT8",
    "FP_E2M1", "FP_E1M2",
    "LNS", "BFP", "POW2",
    "Residual", "Posit8",
]
