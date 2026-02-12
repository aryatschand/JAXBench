"""
Prompts for LLM-based Pallas kernel generation.
"""

KERNEL_GENERATION_PROMPT = """
Generate a Pallas TPU kernel for the following operation:

{operation_description}

Requirements:
1. The kernel must work on TPU (use jax.experimental.pallas.tpu)
2. Use tiled computation with configurable block sizes (default 128x128)
3. Include proper accumulator handling for reduction operations
4. Return a function that can be JIT-compiled

Input shapes:
{input_shapes}

Expected output shape:
{output_shape}

Additional constraints:
{constraints}

Please provide:
1. The kernel function
2. The wrapper function that calls pallas_call
3. A simple test to verify correctness

Remember:
- Do NOT capture JAX arrays as constants
- Block sizes must be divisible by 8 and 128 (last two dims)
- Use @pl.when for conditional execution
- Use preferred_element_type for mixed precision
"""

REFINEMENT_PROMPT = """
The Pallas kernel failed with the following error:

{error_message}

Original kernel code:
```python
{kernel_code}
```

Please fix the kernel to address this error. Common issues:
1. Constant capture: Pass values as inputs or use Python floats
2. Block size: Ensure divisibility by 8 and 128
3. Type mismatch: Check dtypes are compatible
4. Grid spec: Verify BlockSpec index functions are correct

Provide the corrected kernel code.
"""

OPTIMIZATION_PROMPT = """
The Pallas kernel runs but is slower than JAX native:

JAX time: {jax_time_ms:.2f} ms
Pallas time: {pallas_time_ms:.2f} ms
Slowdown: {slowdown:.2f}x

Current kernel:
```python
{kernel_code}
```

Analyze why this kernel is slow and suggest optimizations. Consider:
1. Memory access patterns
2. Block size tuning
3. Unnecessary operations in the kernel
4. Whether Pallas is even appropriate for this operation

If Pallas cannot beat JAX for this operation, explain why and recommend using JAX native.
"""

FUSION_PROMPT = """
Generate a fused Pallas kernel that combines these operations into a single kernel:

Operations to fuse:
{operations}

The goal is to reduce memory bandwidth by avoiding intermediate tensor materialization.

Input: {input_description}
Output: {output_description}

Requirements:
1. Single kernel that performs all operations
2. Intermediate results stored in registers/VMEM only
3. Only one pass over the input data
"""
