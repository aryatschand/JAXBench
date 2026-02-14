"""
Pallas TPU Kernel Prompt Templates - Optimized Version.

Contains system prompts and templates for translating JAX code to optimized Pallas TPU kernels.
Focuses on:
1. Avoiding unnecessary tiling overhead
2. Proper operation fusion patterns
3. TPU-specific optimizations
"""

# =============================================================================
# PALLAS SYSTEM PROMPT - Optimized for Performance
# =============================================================================

PALLAS_SYSTEM_PROMPT = """You are an expert at writing HIGH-PERFORMANCE Pallas TPU kernels for JAX.

## CRITICAL PERFORMANCE INSIGHT

The #1 mistake is OVER-TILING. Each pallas_call has overhead. For operations that fit in VMEM:
- **DO NOT TILE** - process the entire tensor in one kernel call
- Only tile when the tensor is too large for VMEM (~16MB on TPU v4/v5)
- A single kernel processing 1M elements is FASTER than 1000 kernels processing 1K each

## WHEN TO USE PALLAS vs STANDARD JAX

Pallas wins when:
1. **Fusing multiple operations** - avoid memory round-trips
2. **Custom memory access patterns** - flash attention, sliding windows
3. **Operations JAX/XLA doesn't optimize** - custom quantization, sparse ops

Pallas loses when:
- Single standard ops (matmul, softmax) - XLA is already optimal
- Small tensors - kernel launch overhead dominates

## IMPORTS (Always use these)

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools
```

## PATTERN 1: ELEMENTWISE WITHOUT TILING (Fastest for tensors < 16MB)

For ReLU, GELU, Sigmoid, Tanh on tensors that fit in VMEM:

```python
def relu_kernel(x_ref, o_ref):
    # Process entire tensor at once - no tiling overhead!
    o_ref[...] = jnp.maximum(x_ref[...], 0.0)

def relu_pallas(x):
    '''Fast Pallas ReLU - no tiling.'''
    return pl.pallas_call(
        relu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(x.shape, lambda: ())],
        out_specs=pl.BlockSpec(x.shape, lambda: ()),
        grid=(1,),  # Single kernel call!
    )(x)
```

## PATTERN 2: HYBRID MATMUL + PALLAS ACTIVATION (Key for Level 2)

**CRITICAL**: For large matrices, DON'T reimplement matmul in Pallas!
XLA's matmul is highly optimized. Instead, use a HYBRID approach:
1. Use JAX's jnp.matmul for the matmul (XLA optimized)
2. Use Pallas ONLY for fusing the POST-PROCESSING (activation, scale, etc.)

```python
def fused_activation_kernel(x_ref, bias_ref, o_ref, *, multiplier, negative_slope):
    '''Pallas kernel for ONLY the post-matmul operations.'''
    x = x_ref[...].astype(jnp.float32)  # Get matmul result
    x = x + bias_ref[...]  # Add bias
    x = x * multiplier     # Scale
    x = jnp.where(x >= 0, x, x * negative_slope)  # LeakyReLU
    o_ref[...] = x.astype(o_ref.dtype)

def forward_pallas(self, x):
    '''Hybrid: JAX matmul + Pallas fused activation.'''
    # Step 1: Use JAX's optimized matmul
    mm_result = jnp.matmul(x, self.weight)

    # Step 2: Fuse ALL post-processing in Pallas
    return pl.pallas_call(
        functools.partial(fused_activation_kernel,
                         multiplier=self.multiplier,
                         negative_slope=self.negative_slope),
        out_shape=jax.ShapeDtypeStruct(mm_result.shape, mm_result.dtype),
        in_specs=[
            pl.BlockSpec(mm_result.shape, lambda: ()),
            pl.BlockSpec(self.bias.shape, lambda: ()),
        ],
        out_specs=pl.BlockSpec(mm_result.shape, lambda: ()),
        grid=(1,),
    )(mm_result, self.bias)
```

This hybrid approach:
- Keeps XLA's fast matmul
- Fuses bias + scale + activation into ONE memory pass
- Avoids the overhead of implementing matmul in Pallas

## PATTERN 3: FUSED MATMUL + MULTIPLY + ACTIVATION

Common pattern in neural networks (Gemm + scale + activation):

```python
def matmul_scale_leakyrelu_kernel(x_ref, w_ref, scale_ref, o_ref, *, negative_slope):
    '''Fused: (x @ w) * scale with LeakyReLU.'''
    # Step 1: Matmul
    mm = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    # Step 2: Element-wise multiply with scale
    scaled = mm * scale_ref[...]
    # Step 3: LeakyReLU
    o_ref[...] = jnp.where(scaled > 0, scaled, negative_slope * scaled).astype(o_ref.dtype)

def matmul_scale_leakyrelu_pallas(x, w, scale, negative_slope=0.01):
    m, _ = x.shape
    _, n = w.shape
    return pl.pallas_call(
        functools.partial(matmul_scale_leakyrelu_kernel, negative_slope=negative_slope),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=[
            pl.BlockSpec(x.shape, lambda: ()),
            pl.BlockSpec(w.shape, lambda: ()),
            pl.BlockSpec(scale.shape, lambda: ()),
        ],
        out_specs=pl.BlockSpec((m, n), lambda: ()),
        grid=(1,),
    )(x, w, scale)
```

## PATTERN 4: FUSED MATMUL + REDUCTION (Sum/Mean)

```python
def matmul_sum_kernel(x_ref, w_ref, o_ref, *, axis):
    '''Fused matmul + sum reduction.'''
    mm = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = jnp.sum(mm, axis=axis).astype(o_ref.dtype)

def matmul_sum_pallas(x, w, axis=-1):
    m, k = x.shape
    _, n = w.shape
    if axis == -1 or axis == 1:
        out_shape = (m, 1)  # Sum over columns
    else:
        out_shape = (1, n)  # Sum over rows
    return pl.pallas_call(
        functools.partial(matmul_sum_kernel, axis=axis),
        out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
        in_specs=[
            pl.BlockSpec(x.shape, lambda: ()),
            pl.BlockSpec(w.shape, lambda: ()),
        ],
        out_specs=pl.BlockSpec(out_shape, lambda: ()),
        grid=(1,),
    )(x, w)
```

## PATTERN 5: FUSED MULTIPLE ELEMENTWISE OPS

Chain multiple elementwise ops to avoid memory round-trips:

```python
def gelu_dropout_kernel(x_ref, o_ref, *, dropout_key, dropout_rate):
    '''Fused GELU + Dropout.'''
    x = x_ref[...]
    # GELU activation
    gelu = x * 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))
    # Dropout (if training)
    if dropout_rate > 0:
        mask = jax.random.bernoulli(dropout_key, 1.0 - dropout_rate, shape=x.shape)
        gelu = jnp.where(mask, gelu / (1.0 - dropout_rate), 0.0)
    o_ref[...] = gelu.astype(o_ref.dtype)
```

## PATTERN 6: TILED MATMUL (Only for LARGE matrices)

Only use tiling when matrices are too large for VMEM (>16MB).
For a 4096x4096 float32 matrix = 64MB, we need tiling.
Use LARGE blocks (512x512 or 1024x512) to minimize overhead:

```python
def tiled_matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
    '''Tiled matmul with accumulator.'''
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)

def tiled_matmul_pallas(x, y):
    '''Tiled matmul for large matrices. Use 512x512 blocks for better perf.'''
    m, k = x.shape
    _, n = y.shape

    # Use LARGE blocks to minimize tiling overhead
    bm, bk, bn = 512, 512, 512

    # Adjust if dimensions don't divide evenly
    while m % bm != 0 and bm > 128:
        bm //= 2
    while k % bk != 0 and bk > 128:
        bk //= 2
    while n % bn != 0 and bn > 128:
        bn //= 2

    return pl.pallas_call(
        functools.partial(tiled_matmul_kernel, nsteps=k // bk),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(x, y)
```

## PATTERN 7: FUSED SOFTMAX (Numerically stable)

```python
def fused_softmax_kernel(x_ref, o_ref):
    '''Numerically stable softmax in single kernel.'''
    x = x_ref[...]
    # All in one kernel - no memory round-trips
    x_max = jnp.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max
    exp_x = jnp.exp(x_shifted)
    sum_exp = jnp.sum(exp_x, axis=-1, keepdims=True)
    o_ref[...] = (exp_x / sum_exp).astype(o_ref.dtype)

def softmax_pallas(x):
    return pl.pallas_call(
        fused_softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(x.shape, lambda: ())],
        out_specs=pl.BlockSpec(x.shape, lambda: ()),
        grid=(1,),
    )(x)
```

## PATTERN 8: FLASH ATTENTION (Level 3 - Key for speedups!)

Standard attention materializes O(T²) memory. Flash Attention tiles to O(T).
This is where Pallas can BEAT JAX - especially for custom attention variants.

```python
def flash_attention_kernel(q_ref, k_ref, v_ref, o_ref, *, scale):
    '''Flash attention for a single block - tiles over KV to avoid O(T²) memory.'''
    q = q_ref[...].astype(jnp.float32)  # (block_q, head_dim)

    # Initialize running max and sum for online softmax
    m_i = jnp.full((q.shape[0],), -jnp.inf)  # Running max
    l_i = jnp.zeros((q.shape[0],))           # Running sum of exp
    o_i = jnp.zeros_like(q)                   # Running output

    # Iterate over K,V blocks (this is the key - we don't materialize full attention!)
    k = k_ref[...]
    v = v_ref[...]

    # Compute attention scores for this block
    s = jnp.dot(q, k.T) * scale  # (block_q, block_kv)

    # Online softmax update
    m_new = jnp.maximum(m_i, jnp.max(s, axis=-1))
    p = jnp.exp(s - m_new[:, None])
    l_new = jnp.exp(m_i - m_new) * l_i + jnp.sum(p, axis=-1)

    # Update output
    o_new = (jnp.exp(m_i - m_new)[:, None] * o_i + jnp.dot(p, v)) / l_new[:, None]

    o_ref[...] = o_new.astype(o_ref.dtype)
```

For CAUSAL attention, add masking: `s = jnp.where(causal_mask, s, -jnp.inf)`

## PATTERN 9: FUSED ATTENTION PROJECTION

Fuse QKV projection + reshape in one kernel to avoid intermediate tensors:

```python
def fused_qkv_projection_kernel(x_ref, w_qkv_ref, o_q_ref, o_k_ref, o_v_ref, *, n_heads, head_dim):
    '''Fused QKV projection + reshape for attention.'''
    x = x_ref[...].astype(jnp.float32)  # (seq_len, embed_dim)
    w = w_qkv_ref[...]  # (embed_dim, 3 * embed_dim)

    # Single matmul for Q, K, V
    qkv = jnp.dot(x, w)  # (seq_len, 3 * embed_dim)

    # Split and reshape in one pass
    seq_len = x.shape[0]
    q, k, v = jnp.split(qkv, 3, axis=-1)

    # Reshape to (n_heads, seq_len, head_dim)
    o_q_ref[...] = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2).astype(o_q_ref.dtype)
    o_k_ref[...] = k.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2).astype(o_k_ref.dtype)
    o_v_ref[...] = v.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2).astype(o_v_ref.dtype)
```

## PATTERN 10: RMSNorm (Common in modern transformers)

```python
def rmsnorm_kernel(x_ref, weight_ref, o_ref, *, eps):
    '''Fused RMSNorm - faster than separate ops.'''
    x = x_ref[...].astype(jnp.float32)
    w = weight_ref[...]

    # RMSNorm: x / sqrt(mean(x²) + eps) * weight
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    o_ref[...] = ((x / rms) * w).astype(o_ref.dtype)
```

## TPU CONSTRAINTS (Must Follow)

1. **Block dimensions**: rows divisible by 8, cols divisible by 128
2. **VMEM limit**: ~16MB on TPU v4/v5, ~32MB on v6
3. **Scratch space**: Always use float32 for accumulators
4. **Grid spec**: Use PrefetchScalarGridSpec for best performance

## COMMON ERRORS AND FIXES

1. **VMEM exhausted**: Tensor too large for single kernel
   - FIX: Add tiling with BlockSpec, use smaller blocks

2. **Shape mismatch**: BlockSpec doesn't match tensor shape
   - FIX: Ensure block_shape divides tensor_shape evenly

3. **Numerical differences**: Float precision issues
   - FIX: Use float32 for intermediate computations, cast at end

4. **Mosaic compilation error**: Invalid Pallas construct
   - FIX: Simplify kernel, avoid dynamic shapes, check block alignment

## OUTPUT REQUIREMENTS

1. Return COMPLETE, runnable Python code
2. Include ALL imports at top
3. Keep Model class with `forward()` (original JAX) and `forward_pallas()` (optimized)
4. Match the EXACT output of the original JAX code
5. For fused operations: implement the FULL fusion in a single kernel
6. Do NOT use @jax.jit on methods - JIT is applied externally
"""


# =============================================================================
# PALLAS TRANSLATION TEMPLATE - Optimized
# =============================================================================

PALLAS_TRANSLATION_TEMPLATE = '''Convert this JAX code to an OPTIMIZED Pallas TPU kernel.

## Original JAX Code:
```python
{jax_code}
```

## Task Information:
- Task Name: {task_name}
- Operation Type: {operation_type}

## OPTIMIZATION STRATEGY:
{optimization_strategy}

## Requirements:
1. **FUSE ALL OPERATIONS** into a single Pallas kernel when possible
2. **AVOID TILING** unless tensors exceed ~16MB
3. Use `grid=(1,)` for tensors that fit in VMEM
4. Use float32 for intermediate computations
5. Keep Model class with both `forward()` and `forward_pallas()` methods

## Output:
Return ONLY the complete Python code starting with imports.
Do NOT include markdown code blocks or explanations.
'''


# =============================================================================
# PALLAS REFINEMENT TEMPLATE - Enhanced
# =============================================================================

PALLAS_REFINEMENT_TEMPLATE = '''The Pallas kernel has an error. Fix it.

## Original JAX Code:
```python
{jax_code}
```

## Current Pallas Code (with error):
```python
{pallas_code}
```

## Error Message:
{error}

## SPECIFIC FIX GUIDANCE:
{fix_guidance}

## General Fix Patterns:
1. **VMEM exhausted**: Add tiling - split into blocks
2. **Shape mismatch**: Make BlockSpec shapes divide tensor shapes evenly
3. **Numerical diff**: Use float32 intermediates, cast output to original dtype
4. **Mosaic error**: Simplify kernel, avoid dynamic indexing

Return ONLY the complete fixed Python code starting with imports.
'''


# =============================================================================
# OPERATION TYPE DETECTION - Enhanced
# =============================================================================

def get_hybrid_conv_strategy() -> str:
    """Get strategy for convolution + post-processing fusion."""
    return '''
HYBRID CONVOLUTION + FUSED POST-PROCESSING STRATEGY:

**CRITICAL**: Never reimplement convolution in Pallas!
Use JAX's lax.conv_general_dilated (XLA optimized), then fuse post-processing:

Example for Conv2D + ReLU + BiasAdd:
def forward_pallas(self, x):
    # Step 1: Use JAX's optimized convolution
    conv_result = jax.lax.conv_general_dilated(
        x, self.weight, window_strides=self.stride, padding=self.padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )

    # Step 2: Fuse bias + activation in Pallas (single memory pass)
    def fused_postconv_kernel(conv_ref, bias_ref, o_ref):
        x = conv_ref[...].astype(jnp.float32)
        x = x + bias_ref[...]  # Add bias (broadcast over spatial dims)
        x = jnp.maximum(x, 0.0)  # ReLU
        o_ref[...] = x.astype(o_ref.dtype)

    return pl.pallas_call(
        fused_postconv_kernel,
        out_shape=jax.ShapeDtypeStruct(conv_result.shape, conv_result.dtype),
        in_specs=[pl.BlockSpec(conv_result.shape, lambda: ()), pl.BlockSpec(self.bias.shape, lambda: ())],
        out_specs=pl.BlockSpec(conv_result.shape, lambda: ()),
        grid=(1,),
    )(conv_result, self.bias)

For Mish activation: x * jnp.tanh(jax.nn.softplus(x))
For Tanh: jnp.tanh(x)
For Sigmoid: jax.nn.sigmoid(x) or 1 / (1 + jnp.exp(-x))
'''


def detect_operation_type(jax_code: str) -> str:
    """Detect the primary operation type from JAX code."""
    code_lower = jax_code.lower()

    # Level 3 patterns (attention, transformers) - highest priority
    has_attention = 'attention' in code_lower or ('q' in code_lower and 'k' in code_lower and 'v' in code_lower)
    has_softmax = 'softmax' in code_lower
    has_qkv = 'c_attn' in code_lower or 'qkv' in code_lower or 'q, k, v' in code_lower
    has_transpose = 'transpose' in code_lower or 'swapaxes' in code_lower
    has_causal = 'causal' in code_lower or 'tril' in code_lower or 'mask' in code_lower

    # Attention patterns (Level 3 - where Pallas can really win!)
    if has_qkv and has_softmax:
        return 'attention_softmax'  # Standard attention - flash attention opportunity
    if has_qkv and 'maximum' in code_lower:
        return 'attention_relu'  # ReLU attention - definitely needs Pallas
    if has_attention or (has_qkv and has_transpose):
        return 'attention_general'

    # State space / Mamba patterns
    if 'segsum' in code_lower or 'cumsum' in code_lower and 'einsum' in code_lower:
        return 'state_space_model'

    # Check for fused patterns (Level 2)
    has_matmul = 'matmul' in code_lower or 'jnp.dot' in code_lower or '@ ' in code_lower
    has_conv = 'conv' in code_lower or 'lax.conv' in code_lower
    has_activation = any(x in code_lower for x in ['relu', 'gelu', 'sigmoid', 'tanh', 'leaky', 'mish', 'swish', 'softplus', 'hardswish'])
    has_reduction = any(x in code_lower for x in ['sum(', 'mean(', 'max(', 'jnp.sum', 'jnp.mean'])
    has_multiply = '*' in code_lower or 'multiply' in code_lower
    has_bias = 'bias' in code_lower or '+ self.' in code_lower

    # Normalization patterns
    if 'layernorm' in code_lower or 'layer_norm' in code_lower:
        return 'layernorm'
    if 'rmsnorm' in code_lower or 'rms_norm' in code_lower:
        return 'rmsnorm'
    if 'batchnorm' in code_lower or 'batch_norm' in code_lower:
        return 'batchnorm'

    # Fused conv patterns (Level 2 priority)
    if has_conv and has_activation:
        return 'fused_conv_activation'
    if has_conv and has_bias:
        return 'fused_conv_bias'

    # Fused matmul patterns
    if has_matmul and has_activation:
        return 'fused_matmul_activation'
    if has_matmul and has_reduction:
        return 'fused_matmul_reduction'
    if has_matmul and has_multiply:
        return 'fused_matmul_scale'

    # Single operations
    if has_matmul:
        return 'matmul'
    if has_conv:
        return 'convolution'
    if 'softmax' in code_lower:
        return 'softmax'
    if 'relu' in code_lower or ('maximum' in code_lower and '0' in code_lower):
        return 'elementwise_relu'
    if 'gelu' in code_lower:
        return 'elementwise_gelu'
    if 'sigmoid' in code_lower:
        return 'elementwise_sigmoid'
    if 'tanh' in code_lower:
        return 'elementwise_tanh'
    if 'mish' in code_lower:
        return 'elementwise_mish'
    if has_reduction:
        return 'reduction'

    return 'general'


def get_optimization_strategy(operation_type: str) -> str:
    """Get optimization strategy for an operation type."""
    strategies = {
        # Level 3 - Attention patterns (HIGHEST VALUE FOR PALLAS!)
        'attention_softmax': '''
FLASH ATTENTION STRATEGY (HIGH VALUE - Can beat JAX!):

Standard attention: att = softmax(Q @ K.T / sqrt(d)) @ V materializes O(T²) memory.
Flash Attention tiles this to use O(T) memory - MAJOR speedup for long sequences!

**CRITICAL**: The standard JAX softmax attention can be beaten with tiled Pallas.

Approach:
1. Use JAX for QKV projection (matmul - XLA is optimal)
2. Implement TILED attention in Pallas using online softmax:
   - Process Q in blocks
   - For each Q block, iterate over K,V blocks
   - Maintain running max and sum for numerically stable softmax
   - Never materialize the full T×T attention matrix!

Key insight: The softmax normalization can be computed incrementally.
m_new = max(m_old, max(scores))
l_new = exp(m_old - m_new) * l_old + sum(exp(scores - m_new))
output = (exp(m_old - m_new) * output_old + exp(scores - m_new) @ V) / l_new

For CAUSAL attention, add triangular mask before softmax.
''',
        'attention_relu': '''
RELU ATTENTION STRATEGY (BEST OPPORTUNITY - XLA has NO optimization!):

ReLU attention: att = ReLU(Q @ K.T / sqrt(d)) @ V

XLA does NOT have flash attention for ReLU attention - this is where Pallas WINS!

Approach:
1. Use JAX for QKV projection
2. Implement TILED ReLU attention in Pallas:
   - ReLU doesn't need online softmax (simpler than softmax attention!)
   - Just clamp negative values and accumulate
   - Still tile to avoid O(T²) memory

def relu_attention_kernel(q_ref, k_ref, v_ref, o_ref, *, scale):
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...]
    v = v_ref[...]
    scores = jnp.maximum(jnp.dot(q, k.T) * scale, 0.0)  # ReLU
    o_ref[...] = jnp.dot(scores, v).astype(o_ref.dtype)

This is simpler than flash attention and should provide good speedup!
''',
        'attention_general': '''
GENERAL ATTENTION OPTIMIZATION:

1. Fuse QKV projection if possible (single matmul + split)
2. For attention computation, consider tiling to avoid O(T²) memory
3. Fuse attention output projection with residual add if present

Key opportunities:
- QKV projection fusion
- Attention + output projection fusion
- Multi-head parallel processing
''',
        'state_space_model': '''
STATE SPACE MODEL (MAMBA) STRATEGY:

Mamba uses structured state spaces with:
- cumsum operations
- einsum for state updates
- segment sums with masking

Pallas can help by:
1. Fusing the segsum + exp computation
2. Fusing einsum chains
3. Custom scan kernels for state updates

Focus on fusing the inner loop operations.
''',
        'layernorm': '''
LAYERNORM FUSION STRATEGY:

LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta

Fuse all operations in one pass:
def layernorm_kernel(x_ref, gamma_ref, beta_ref, o_ref, *, eps):
    x = x_ref[...].astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean)**2, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    o_ref[...] = (x_norm * gamma_ref[...] + beta_ref[...]).astype(o_ref.dtype)
''',
        'rmsnorm': '''
RMSNORM FUSION STRATEGY (Common in LLMs):

RMSNorm: x / sqrt(mean(x²) + eps) * weight

Simpler than LayerNorm, no mean subtraction:
def rmsnorm_kernel(x_ref, weight_ref, o_ref, *, eps):
    x = x_ref[...].astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    o_ref[...] = ((x / rms) * weight_ref[...]).astype(o_ref.dtype)
''',
        'fused_conv_activation': get_hybrid_conv_strategy(),
        'fused_conv_bias': get_hybrid_conv_strategy(),
        'fused_matmul_activation': '''
HYBRID MATMUL + FUSED ACTIVATION STRATEGY:

**CRITICAL**: For matrices > 16MB, use HYBRID approach:
1. Use JAX's jnp.matmul (XLA optimized, don't reimplement!)
2. Use Pallas ONLY for fusing post-matmul operations (bias + activation)

Example:
def forward_pallas(self, x):
    # Step 1: JAX matmul (XLA optimized)
    mm_result = jnp.matmul(x, self.weight)

    # Step 2: Pallas fuses ALL post-processing
    def fused_postprocess_kernel(mm_ref, bias_ref, o_ref, *, scale, neg_slope):
        x = mm_ref[...].astype(jnp.float32)
        x = x + bias_ref[...]  # Add bias
        x = x * scale          # Scale/multiply
        x = jnp.where(x >= 0, x, x * neg_slope)  # LeakyReLU
        o_ref[...] = x.astype(o_ref.dtype)

    return pl.pallas_call(
        functools.partial(fused_postprocess_kernel, scale=self.multiplier, neg_slope=self.negative_slope),
        out_shape=jax.ShapeDtypeStruct(mm_result.shape, mm_result.dtype),
        in_specs=[pl.BlockSpec(mm_result.shape, lambda: ()), pl.BlockSpec(self.bias.shape, lambda: ())],
        out_specs=pl.BlockSpec(mm_result.shape, lambda: ()),
        grid=(1,),
    )(mm_result, self.bias)

This wins because:
- XLA matmul is already optimal (don't compete with it!)
- Pallas fuses bias+scale+activation into ONE memory pass
- Avoids intermediate memory writes between operations
''',
        'fused_matmul_reduction': '''
HYBRID MATMUL + FUSED REDUCTION STRATEGY:

For large matrices, use HYBRID approach:
1. Use JAX's jnp.matmul (XLA optimized)
2. Use Pallas ONLY for fusing post-matmul ops (divide + sum + scale)

Example:
def forward_pallas(self, x):
    # Step 1: JAX matmul
    mm_result = jnp.matmul(x, self.weight) + self.bias

    # Step 2: Pallas fuses divide + sum + scale
    def fused_reduction_kernel(x_ref, o_ref, *, divisor, scale):
        x = x_ref[...].astype(jnp.float32)
        x = x / divisor
        x = jnp.sum(x, axis=1, keepdims=True)
        x = x * scale
        o_ref[...] = x.astype(o_ref.dtype)

    out_shape = (mm_result.shape[0], 1)
    return pl.pallas_call(
        functools.partial(fused_reduction_kernel, divisor=self.divisor, scale=self.scale),
        out_shape=jax.ShapeDtypeStruct(out_shape, mm_result.dtype),
        in_specs=[pl.BlockSpec(mm_result.shape, lambda: ())],
        out_specs=pl.BlockSpec(out_shape, lambda: ()),
        grid=(1,),
    )(mm_result)

Wins by fusing divide + sum + scale in single memory pass.
''',
        'fused_matmul_scale': '''
FUSED MATMUL + SCALE STRATEGY:
- Combine matmul and element-wise multiply in ONE kernel
- Scale can be a scalar or broadcastable tensor
- Apply any subsequent activation in the same kernel

Example kernel structure:
def fused_kernel(x_ref, w_ref, scale_ref, o_ref):
    mm = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
    scaled = mm * scale_ref[...]  # or mm / scale_ref[...]
    o_ref[...] = scaled.astype(o_ref.dtype)
''',
        'elementwise_relu': '''
ELEMENTWISE RELU STRATEGY:
- Process ENTIRE tensor in one kernel (no tiling!)
- Use grid=(1,) for maximum speed
- Pattern: o_ref[...] = jnp.maximum(x_ref[...], 0.0)
- This can match or beat JAX baseline
''',
        'elementwise_gelu': '''
ELEMENTWISE GELU STRATEGY:
- Process ENTIRE tensor in one kernel
- **IMPORTANT**: TPU Pallas does NOT support erf! Use approximate formula:
- Approximate: x * sigmoid(1.702 * x) OR x * 0.5 * (1 + jnp.tanh(0.7978845608 * (x + 0.044715 * x**3)))
- No tiling needed for tensors < 16MB
''',
        'elementwise_mish': '''
ELEMENTWISE MISH STRATEGY:
- Mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
- Process ENTIRE tensor in one kernel
- Pattern: o_ref[...] = x_ref[...] * jnp.tanh(jax.nn.softplus(x_ref[...]))
- Or expanded: x * jnp.tanh(jnp.log(1 + jnp.exp(x)))
- No tiling needed for tensors < 16MB
''',
        'softmax': '''
SOFTMAX STRATEGY:
- Compute max, subtract, exp, sum, divide ALL in one kernel
- Numerically stable: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
- Process entire tensor if it fits in VMEM
- Use float32 for exp/sum, cast at end
''',
        'matmul': '''
MATMUL STRATEGY:
- For small matrices (<16MB): single kernel, no tiling
- For large matrices: use tiled pattern with LARGE blocks (512x512)
- Use preferred_element_type=jnp.float32 for precision
- NOTE: Hard to beat XLA's matmul - focus on fused ops instead
''',
        'reduction': '''
REDUCTION STRATEGY:
- Process entire tensor in one kernel if possible
- Use float32 accumulator for precision
- Apply keepdims=True if original code does
- Chain with other ops in same kernel when possible
''',
    }
    return strategies.get(operation_type, '''
GENERAL STRATEGY:
- Identify ALL operations and FUSE them into one kernel
- Avoid tiling unless tensors exceed 16MB
- Use grid=(1,) when possible
- Use float32 intermediates, cast output to match original dtype
''')


# =============================================================================
# QUANTIZATION PATTERNS - int8 and bfloat16
# =============================================================================

QUANTIZED_SYSTEM_PROMPT = """You are an expert at writing QUANTIZED Pallas TPU kernels for JAX.

## PRECISION-AWARE KERNEL DESIGN

When writing quantized kernels, the key is proper accumulator handling:
- **int8 matmul**: Accumulate in int32, then dequantize to float32/bfloat16
- **bfloat16 ops**: Accumulate in float32 for numerical stability

## IMPORTS (Always use these for quantized kernels)

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools
```

## PATTERN Q1: BFLOAT16 ELEMENTWISE (Simplest quantized pattern)

For bf16 elementwise ops, cast to float32 for computation, back to bf16 at end:

```python
def bf16_relu_kernel(x_ref, o_ref):
    '''BFloat16 ReLU with float32 intermediate.'''
    x = x_ref[...].astype(jnp.float32)  # Upcast for computation
    result = jnp.maximum(x, 0.0)
    o_ref[...] = result.astype(jnp.bfloat16)  # Back to bf16

def bf16_relu_pallas(x):
    return pl.pallas_call(
        bf16_relu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.bfloat16),
        in_specs=[pl.BlockSpec(x.shape, lambda: ())],
        out_specs=pl.BlockSpec(x.shape, lambda: ()),
        grid=(1,),
    )(x)
```

## PATTERN Q2: BFLOAT16 GELU

```python
def bf16_gelu_kernel(x_ref, o_ref):
    '''BFloat16 GELU with float32 intermediate.'''
    x = x_ref[...].astype(jnp.float32)
    # Approximate GELU (tanh approximation, avoids erf)
    gelu = x * 0.5 * (1.0 + jnp.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    o_ref[...] = gelu.astype(jnp.bfloat16)
```

## PATTERN Q3: BFLOAT16 SIGMOID

```python
def bf16_sigmoid_kernel(x_ref, o_ref):
    '''BFloat16 Sigmoid.'''
    x = x_ref[...].astype(jnp.float32)
    result = 1.0 / (1.0 + jnp.exp(-x))
    o_ref[...] = result.astype(jnp.bfloat16)
```

## PATTERN Q4: INT8 MATMUL WITH DEQUANTIZATION

For int8 matmul, we need:
1. int8 inputs
2. int32 accumulator
3. Scale factors for dequantization
4. Output in float32 or bfloat16

```python
def int8_matmul_kernel(x_ref, w_ref, scale_x_ref, scale_w_ref, o_ref):
    '''Int8 matmul with proper accumulator and dequantization.'''
    # Load int8 values
    x = x_ref[...].astype(jnp.int32)  # Upcast to int32 for accumulation
    w = w_ref[...].astype(jnp.int32)

    # Matmul in int32 (accumulates correctly)
    acc = jnp.dot(x, w)  # int32 accumulator

    # Dequantize: multiply by scale factors
    scale_x = scale_x_ref[...]
    scale_w = scale_w_ref[...]
    result = acc.astype(jnp.float32) * scale_x * scale_w

    o_ref[...] = result.astype(jnp.bfloat16)

def int8_matmul_pallas(x_int8, w_int8, scale_x, scale_w):
    '''Int8 matmul with dequantization to bf16.'''
    m, k = x_int8.shape
    _, n = w_int8.shape
    return pl.pallas_call(
        int8_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        in_specs=[
            pl.BlockSpec(x_int8.shape, lambda: ()),
            pl.BlockSpec(w_int8.shape, lambda: ()),
            pl.BlockSpec(scale_x.shape, lambda: ()),
            pl.BlockSpec(scale_w.shape, lambda: ()),
        ],
        out_specs=pl.BlockSpec((m, n), lambda: ()),
        grid=(1,),
    )(x_int8, w_int8, scale_x, scale_w)
```

## PATTERN Q5: INT8 FUSED MATMUL + ACTIVATION

```python
def int8_matmul_relu_kernel(x_ref, w_ref, scale_ref, o_ref):
    '''Fused int8 matmul + ReLU.'''
    x = x_ref[...].astype(jnp.int32)
    w = w_ref[...].astype(jnp.int32)

    # Matmul
    acc = jnp.dot(x, w)

    # Dequantize and ReLU
    result = acc.astype(jnp.float32) * scale_ref[...]
    result = jnp.maximum(result, 0.0)

    o_ref[...] = result.astype(jnp.bfloat16)
```

## PATTERN Q6: BFLOAT16 SOFTMAX (Numerically stable)

```python
def bf16_softmax_kernel(x_ref, o_ref):
    '''BFloat16 softmax with float32 intermediate.'''
    x = x_ref[...].astype(jnp.float32)
    x_max = jnp.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max
    exp_x = jnp.exp(x_shifted)
    sum_exp = jnp.sum(exp_x, axis=-1, keepdims=True)
    result = exp_x / sum_exp
    o_ref[...] = result.astype(jnp.bfloat16)
```

## PATTERN Q7: BFLOAT16 LAYERNORM

```python
def bf16_layernorm_kernel(x_ref, gamma_ref, beta_ref, o_ref, *, eps):
    '''BFloat16 LayerNorm with float32 intermediate.'''
    x = x_ref[...].astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean)**2, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    result = x_norm * gamma_ref[...].astype(jnp.float32) + beta_ref[...].astype(jnp.float32)
    o_ref[...] = result.astype(jnp.bfloat16)
```

## TPU QUANTIZATION CONSTRAINTS

1. **TPU v4+ has native bf16 support** - efficient bf16 compute
2. **int8 matmul**: Use int32 accumulator, TPU MXU can do this efficiently
3. **Block dimensions**: Same rules apply - rows div by 8, cols div by 128
4. **Memory**: Quantized tensors use less VMEM - can use larger blocks!

## TOLERANCE THRESHOLDS BY PRECISION

- **bfloat16**: rtol=1e-2, atol=1e-2 (vs fp32 reference)
- **int8**: rtol=5e-2, atol=5e-2 (quantization introduces more error)

## OUTPUT REQUIREMENTS

1. Return COMPLETE, runnable Python code
2. Include ALL imports at top
3. Keep Model class with:
   - `forward()` - original JAX implementation
   - `forward_pallas()` - quantized Pallas kernel
4. Include quantization helpers (e.g., `quantize_to_int8`, `dequantize`)
5. Use appropriate dtype throughout (bf16 or int8)
"""


QUANTIZED_TRANSLATION_TEMPLATE = '''Convert this JAX code to a QUANTIZED Pallas TPU kernel.

## Original JAX Code:
```python
{jax_code}
```

## Task Information:
- Task Name: {task_name}
- Target Precision: {precision}

## Quantization Strategy:
{quantization_strategy}

## Requirements:
1. **{precision} precision** - use {accumulator_type} accumulators
2. Process entire tensor in single kernel when possible (grid=(1,))
3. Cast to {accumulator_type} for computation, back to {precision} at end
4. Keep Model class with `forward()` (original) and `forward_pallas()` (quantized)

## Output:
Return ONLY the complete Python code starting with imports.
Do NOT include markdown code blocks or explanations.
'''


def get_quantization_strategy(precision: str, operation_type: str) -> str:
    """Get quantization strategy for a precision and operation type."""

    if precision == "bfloat16":
        accumulator = "float32"
        cast_up = "x.astype(jnp.float32)"
        cast_down = "result.astype(jnp.bfloat16)"

        strategies = {
            'elementwise_relu': f'''
BFLOAT16 RELU STRATEGY:
- Cast input to float32: {cast_up}
- Compute: result = jnp.maximum(x, 0.0)
- Cast back: {cast_down}
- Single kernel, grid=(1,)
''',
            'elementwise_gelu': f'''
BFLOAT16 GELU STRATEGY:
- Cast input to float32: {cast_up}
- Use tanh approximation (erf not supported on TPU):
  gelu = x * 0.5 * (1.0 + jnp.tanh(0.7978845608 * (x + 0.044715 * x**3)))
- Cast back: {cast_down}
''',
            'elementwise_sigmoid': f'''
BFLOAT16 SIGMOID STRATEGY:
- Cast input to float32: {cast_up}
- Compute: result = 1.0 / (1.0 + jnp.exp(-x))
- Cast back: {cast_down}
''',
            'elementwise_tanh': f'''
BFLOAT16 TANH STRATEGY:
- Cast input to float32: {cast_up}
- Compute: result = jnp.tanh(x)
- Cast back: {cast_down}
''',
            'softmax': f'''
BFLOAT16 SOFTMAX STRATEGY:
- Cast to float32 for numerical stability
- Compute stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
- Cast back to bfloat16
''',
            'matmul': f'''
BFLOAT16 MATMUL STRATEGY:
- Keep inputs in bfloat16 (TPU MXU handles this)
- Use preferred_element_type=jnp.float32 for accumulation
- Cast result to bfloat16
''',
        }

    elif precision == "int8":
        accumulator = "int32"

        strategies = {
            'matmul': '''
INT8 MATMUL STRATEGY:
- Quantize inputs: x_int8 = jnp.clip(jnp.round(x / scale), -128, 127).astype(jnp.int8)
- Cast to int32 for matmul: x.astype(jnp.int32)
- Matmul accumulates in int32
- Dequantize: result = acc.astype(jnp.float32) * scale_x * scale_w
- Output as bfloat16 or float32
''',
            'fused_matmul_activation': '''
INT8 FUSED MATMUL + ACTIVATION:
1. Int8 inputs, int32 accumulator
2. Dequantize after matmul
3. Apply activation in float32
4. Output in bfloat16
''',
        }

    else:
        strategies = {}

    base_strategy = strategies.get(operation_type, f'''
GENERAL {precision.upper()} STRATEGY:
- Cast to appropriate accumulator type for computation
- Perform operation
- Cast back to {precision}
- Use grid=(1,) for tensors that fit in VMEM
''')

    return base_strategy


def get_error_fix_guidance(error: str) -> str:
    """Get specific fix guidance based on error message."""
    error_lower = error.lower()

    if 'vmem' in error_lower or 'memory' in error_lower or 'exhausted' in error_lower:
        return '''
VMEM EXHAUSTED FIX:
The tensor is too large for a single kernel. Add tiling:
1. Use BlockSpec with smaller block sizes (e.g., 512x512)
2. Add grid dimensions to process blocks
3. Use PrefetchScalarGridSpec for tiled operations

Example fix:
Instead of: pl.BlockSpec(x.shape, lambda: ())
Use: pl.BlockSpec((512, 512), lambda i, j: (i, j))
And: grid=(M // 512, N // 512)
'''

    if 'shape' in error_lower or 'mismatch' in error_lower:
        return '''
SHAPE MISMATCH FIX:
1. Ensure BlockSpec block_shape divides tensor shape evenly
2. For reductions, output shape must match reduced dimensions
3. Check that all BlockSpec lambdas return correct indices

Example: For tensor (1024, 2048) with blocks (128, 256):
- grid = (1024 // 128, 2048 // 256) = (8, 8)
- lambda i, j: (i, j) returns block indices
'''

    if 'numerical' in error_lower or 'values differ' in error_lower or 'max_diff' in error_lower:
        return '''
NUMERICAL DIFFERENCE FIX:
1. Use float32 for ALL intermediate computations
2. Cast to output dtype only at the final step
3. Use preferred_element_type=jnp.float32 in jnp.dot
4. Ensure reduction order matches original (use same axis)

Example:
mm = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
result = some_operation(mm)  # Keep in float32
o_ref[...] = result.astype(o_ref.dtype)  # Cast at end
'''

    if 'mosaic' in error_lower or 'compile' in error_lower:
        return '''
MOSAIC COMPILATION FIX:
1. Avoid dynamic shapes - all dimensions must be static
2. Ensure block sizes are divisible by 8 (rows) and 128 (cols)
3. Don't use Python loops - use jax.lax.fori_loop if needed
4. Simplify the kernel - break complex ops into simpler ones

Try reducing kernel complexity and using standard patterns.
'''

    if 'swap' in error_lower or 'ref' in error_lower:
        return '''
REFERENCE/SWAP ERROR FIX:
1. Don't modify input refs - only write to output refs
2. Ensure ref[...] reads/writes match the BlockSpec shape exactly
3. For scratch space, use pltpu.VMEM in scratch_shapes

Example: If BlockSpec is (128, 128), ref[...] must be (128, 128)
'''

    return '''
GENERAL FIX:
1. Simplify the kernel to the minimum working version
2. Use grid=(1,) and full tensor BlockSpecs first
3. Add complexity incrementally
4. Check all shapes match between BlockSpec, grid, and tensor
'''


# =============================================================================
# SPARSE KERNEL PATTERNS - Exploiting Structured Sparsity
# =============================================================================

SPARSE_SYSTEM_PROMPT = """You are an expert at writing SPARSE Pallas TPU kernels for JAX.

## KEY INSIGHT: SPARSITY EXPLOITATION

The goal is to write Pallas kernels that SKIP computation on zero values/blocks.
Dense baselines (JAX matmul, attention) compute everything including zeros.
Your sparse kernel should ONLY compute on non-zero elements for speedup.

## IMPORTS (Always use these)

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools
```

## PATTERN S1: BLOCK-SPARSE MATMUL (Skip zero blocks)

For block-sparse matrices where 50%+ of 128x128 blocks are zero:

```python
def block_sparse_matmul_kernel(
    x_ref,           # Input: (M, K)
    w_blocks_ref,    # Non-zero weight blocks: (n_blocks, block_k, block_n)
    block_idx_ref,   # Block indices: (n_blocks, 2) - (k_idx, n_idx) per block
    o_ref,           # Output: (M, N)
    *,
    block_size,
    n_blocks,
):
    '''Block-sparse matmul - only processes non-zero blocks.'''
    # Initialize output to zero
    acc = jnp.zeros_like(o_ref[...])

    # Process only non-zero blocks
    def process_block(i, acc):
        k_idx = block_idx_ref[i, 0]
        n_idx = block_idx_ref[i, 1]

        # Extract input block (M, block_k)
        k_start = k_idx * block_size
        x_block = jax.lax.dynamic_slice(x_ref[...], (0, k_start), (x_ref.shape[0], block_size))

        # Get weight block (block_k, block_n)
        w_block = w_blocks_ref[i]

        # Compute partial product
        partial = jnp.dot(x_block, w_block, preferred_element_type=jnp.float32)

        # Add to correct output location
        n_start = n_idx * block_size
        acc = jax.lax.dynamic_update_slice(acc, acc[:, n_start:n_start+block_size] + partial, (0, n_start))
        return acc

    acc = jax.lax.fori_loop(0, n_blocks, process_block, acc)
    o_ref[...] = acc.astype(o_ref.dtype)
```

## PATTERN S2: DIAGONAL BLOCK MATMUL (Independent blocks - easiest!)

For block-diagonal matrices (only diagonal blocks non-zero):

```python
def diagonal_block_matmul_kernel(x_ref, blocks_ref, o_ref, *, n_blocks, block_size):
    '''Block-diagonal matmul - process each diagonal block independently.

    This is EQUIVALENT to processing independent groups.
    '''
    # x: (batch, n_blocks * block_size)
    # blocks: (n_blocks, block_size, block_size)
    # output: (batch, n_blocks * block_size)

    batch = x_ref.shape[0]

    # Process all blocks in parallel using vmap-like pattern
    def process_one_block(block_idx):
        start = block_idx * block_size
        x_block = x_ref[:, start:start+block_size]  # (batch, block_size)
        w_block = blocks_ref[block_idx]  # (block_size, block_size)
        return jnp.dot(x_block, w_block, preferred_element_type=jnp.float32)

    # Stack results
    outputs = jax.vmap(process_one_block)(jnp.arange(n_blocks))  # (n_blocks, batch, block_size)
    o_ref[...] = jnp.transpose(outputs, (1, 0, 2)).reshape(batch, -1).astype(o_ref.dtype)
```

Alternative simpler pattern using einsum:
```python
def diagonal_block_matmul_simple(x_ref, blocks_ref, o_ref, *, n_blocks, block_size):
    '''Diagonal block matmul using einsum.'''
    batch = x_ref.shape[0]
    # Reshape x to (batch, n_blocks, block_size)
    x_blocked = x_ref[...].reshape(batch, n_blocks, block_size)
    # Compute: output[b,n,m] = sum_h x[b,n,h] * blocks[n,h,m]
    y_blocked = jnp.einsum('bnh,nhm->bnm', x_blocked, blocks_ref[...], preferred_element_type=jnp.float32)
    o_ref[...] = y_blocked.reshape(batch, -1).astype(o_ref.dtype)
```

## PATTERN S3: LOCAL/WINDOW ATTENTION (Only compute nearby scores)

For attention where each query only attends to nearby keys (window attention):

```python
def local_attention_kernel(
    q_ref, k_ref, v_ref, o_ref,
    *,
    window_size,  # One-sided window
    scale,
):
    '''Local attention - only compute scores within window.

    Each query at position i attends to keys at positions [i-W, i+W].
    This computes O(T*W) scores instead of O(T^2).
    '''
    seq_len = q_ref.shape[1]
    batch, _, head_dim = q_ref.shape

    # Initialize output
    output = jnp.zeros_like(o_ref[...])

    # For each query position, only compute scores to nearby keys
    def compute_local_attention(query_pos, output):
        q = q_ref[:, query_pos, :]  # (batch, head_dim)

        # Window bounds
        k_start = jnp.maximum(0, query_pos - window_size)
        k_end = jnp.minimum(seq_len, query_pos + window_size + 1)
        window_len = k_end - k_start

        # Extract local keys and values
        k_local = jax.lax.dynamic_slice(k_ref[...], (0, k_start, 0), (batch, window_len, head_dim))
        v_local = jax.lax.dynamic_slice(v_ref[...], (0, k_start, 0), (batch, window_len, head_dim))

        # Compute attention scores (only for local window!)
        scores = jnp.einsum('bd,bwd->bw', q, k_local) * scale  # (batch, window_len)

        # Softmax (only over local window)
        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Weighted sum of local values
        out = jnp.einsum('bw,bwd->bd', attn_weights, v_local)  # (batch, head_dim)

        return jax.lax.dynamic_update_slice(output, out[:, None, :], (0, query_pos, 0))

    output = jax.lax.fori_loop(0, seq_len, compute_local_attention, output)
    o_ref[...] = output.astype(o_ref.dtype)
```

## PATTERN S4: 2:4 STRUCTURED SPARSITY

For NVIDIA-style 2:4 sparsity (every 4 elements has exactly 2 zeros):

```python
def sparse_2_4_matmul_kernel(
    x_ref,              # Input: (M, K)
    values_ref,         # Non-zero values: (K//4, N, 2)
    indices_ref,        # Indices within group of 4: (K//4, N, 2)
    o_ref,              # Output: (M, N)
):
    '''2:4 sparse matmul - only 2 multiplies per group of 4.'''
    M = x_ref.shape[0]
    K = x_ref.shape[1]
    N = values_ref.shape[1]
    n_groups = K // 4

    # Initialize output
    acc = jnp.zeros((M, N), dtype=jnp.float32)

    # For each group of 4 in K dimension
    def process_group(g, acc):
        # Get the 2 indices and values for this group
        idx0 = indices_ref[g, :, 0]  # (N,) - local index 0-3
        idx1 = indices_ref[g, :, 1]  # (N,) - local index 0-3
        val0 = values_ref[g, :, 0]   # (N,) - value at idx0
        val1 = values_ref[g, :, 1]   # (N,) - value at idx1

        # Gather input values
        base = g * 4
        x_group = x_ref[:, base:base+4]  # (M, 4)

        # For each output column, gather 2 input values and multiply
        for n in range(N):
            i0, i1 = idx0[n], idx1[n]
            v0, v1 = val0[n], val1[n]
            # Only 2 muls instead of 4
            acc = acc.at[:, n].add(x_group[:, i0] * v0 + x_group[:, i1] * v1)

        return acc

    acc = jax.lax.fori_loop(0, n_groups, process_group, acc)
    o_ref[...] = acc.astype(o_ref.dtype)
```

## PATTERN S5: BLOCK-SPARSE MLP (Two sparse matmuls + GELU)

```python
def sparse_mlp_kernel(
    x_ref,
    w1_blocks_ref, w1_idx_ref,  # Sparse W1
    w2_blocks_ref, w2_idx_ref,  # Sparse W2
    o_ref,
    *,
    block_size,
    n_blocks_w1, n_blocks_w2,
):
    '''Sparse MLP with block-sparse W1 and W2.'''
    # Layer 1: Sparse matmul
    hidden = block_sparse_matmul(x_ref[...], w1_blocks_ref, w1_idx_ref, block_size, n_blocks_w1)

    # GELU activation (fused)
    hidden = hidden * 0.5 * (1.0 + jnp.tanh(0.7978845608 * (hidden + 0.044715 * hidden**3)))

    # Layer 2: Sparse matmul
    output = block_sparse_matmul(hidden, w2_blocks_ref, w2_idx_ref, block_size, n_blocks_w2)

    o_ref[...] = output.astype(o_ref.dtype)
```

## KEY SPARSITY EXPLOITATION STRATEGIES

1. **Block-sparse**: Skip entire 128x128 blocks that are zero
2. **Diagonal**: Process independent diagonal blocks (trivially parallel)
3. **Local attention**: Only compute O(T*W) scores, not O(T^2)
4. **2:4 sparse**: Only 2 multiplies per group of 4 elements

## CORRECTNESS REQUIREMENTS

1. Sparse kernel output MUST MATCH dense baseline exactly
2. The sparse pattern is FIXED at initialization - don't assume runtime sparsity
3. Store non-zero blocks/indices in model init, pass to kernel

## PERFORMANCE EXPECTATIONS

- 50% sparsity -> ~2x speedup (theoretical)
- 75% sparsity -> ~4x speedup (theoretical)
- Actual speedup depends on memory bandwidth vs compute tradeoff

## OUTPUT REQUIREMENTS

1. Return COMPLETE Python code with imports
2. Keep Model class with:
   - `forward()` - dense baseline (computes all elements)
   - `forward_pallas()` - sparse kernel (skips zeros)
3. Include `get_sparsity_info()` method that returns:
   - sparsity_ratio
   - theoretical_speedup
   - pattern description
"""


SPARSE_TRANSLATION_TEMPLATE = '''Convert this sparse JAX workload to a Pallas kernel that EXPLOITS SPARSITY.

## Original JAX Code (Dense Baseline):
```python
{jax_code}
```

## Task Information:
- Task Name: {task_name}
- Sparsity Pattern: {sparsity_pattern}
- Sparsity Ratio: {sparsity_ratio}
- Theoretical Speedup: {theoretical_speedup}x

## KEY REQUIREMENT: SKIP ZERO COMPUTATION!

The dense baseline computes ALL elements including zeros.
Your Pallas kernel should ONLY compute non-zero elements.

## Sparsity Exploitation Strategy:
{sparsity_strategy}

## Requirements:
1. **SKIP ZEROS**: Only process non-zero blocks/elements
2. Use sparse storage (e.g., block indices, CSR, packed format)
3. Achieve actual speedup from reduced computation
4. Output MUST match dense baseline exactly

## Output:
Return ONLY the complete Python code starting with imports.
Include Model class with forward() (dense) and forward_pallas() (sparse).
'''


def get_sparsity_strategy(sparsity_pattern: str) -> str:
    """Get sparsity exploitation strategy for a pattern type."""

    strategies = {
        'block_sparse_matmul': '''
BLOCK-SPARSE MATMUL STRATEGY:

The weight matrix has block-sparse pattern (128x128 blocks with 50% zeros).

1. **Store non-zero blocks compactly**:
   - Extract non-zero blocks into (n_blocks, 128, 128) tensor
   - Store block indices (k_idx, n_idx) for each non-zero block

2. **Kernel structure**:
   - Loop over non-zero blocks only
   - For each block: compute partial matmul and accumulate
   - Use jax.lax.fori_loop for the block loop

3. **Memory access**:
   - Input: dynamic_slice to extract columns for each K block
   - Weight: sequential access through non-zero blocks
   - Output: dynamic_update_slice to accumulate partial results

Expected speedup: ~2x for 50% sparsity
''',
        'block_sparse_attention': '''
LOCAL ATTENTION STRATEGY:

Standard attention computes O(T^2) scores. Local attention only computes
O(T * W) where W is window size.

1. **Don't materialize full attention matrix**:
   - For each query position, only compute scores to nearby keys
   - Window: [max(0, i-W), min(T, i+W)]

2. **Kernel structure**:
   - Loop over query positions
   - For each query, extract local K,V using dynamic_slice
   - Compute softmax over local window only
   - Weighted sum of local values

3. **Softmax handling**:
   - Softmax is computed over local window
   - No masking needed since we only compute valid scores

Expected speedup: ~T/W for window_size << seq_len
''',
        'structured_2_4': '''
2:4 STRUCTURED SPARSITY STRATEGY:

Every group of 4 consecutive elements has exactly 2 non-zeros.

1. **Compact storage**:
   - values: (K//4, N, 2) - 2 non-zero values per group per output col
   - indices: (K//4, N, 2) - local indices (0-3) of non-zeros

2. **Kernel structure**:
   - Loop over groups of 4 in K dimension
   - For each group: only 2 multiplies instead of 4
   - Accumulate partial products

3. **Gather pattern**:
   - x_group = x[:, g*4:(g+1)*4]  # (M, 4)
   - partial = x_group[:, idx0] * val0 + x_group[:, idx1] * val1

Expected speedup: ~2x (fixed 50% sparsity)
''',
        'sparse_mlp': '''
SPARSE MLP STRATEGY:

Block-sparse FFN with 75% sparsity in W1 and W2.

1. **Two sparse matmuls**:
   - hidden = sparse_matmul(x, W1) using block-sparse pattern
   - output = sparse_matmul(GELU(hidden), W2) using block-sparse pattern

2. **Fuse GELU activation**:
   - Apply GELU between the two sparse matmuls
   - Use tanh approximation (no erf on TPU)

3. **Block structure**:
   - Store W1 and W2 non-zero blocks separately
   - Use same block-sparse kernel for both layers

Expected speedup: ~4x for 75% sparsity
''',
        'diagonal_block': '''
DIAGONAL BLOCK MATMUL STRATEGY (EASIEST!):

Only diagonal blocks are non-zero - blocks are completely independent.

1. **Simple structure**:
   - Reshape x to (batch, n_blocks, block_size)
   - Process each block independently: y[b,n,:] = x[b,n,:] @ W[n,:,:]
   - Use einsum: 'bnh,nhm->bnm'

2. **This is equivalent to**:
   - n_blocks independent small matmuls
   - Fully parallelizable across blocks

3. **Pallas kernel**:
   - Either use vmap pattern over blocks
   - Or single einsum (JAX may optimize this)

Expected speedup: ~n_blocks (only 1/n_blocks of computation)
''',
    }

    return strategies.get(sparsity_pattern, '''
GENERAL SPARSE STRATEGY:
1. Identify the sparsity pattern (block, diagonal, 2:4, window)
2. Store only non-zero elements compactly
3. Use dynamic_slice/dynamic_update_slice for scattered access
4. Use jax.lax.fori_loop for iteration over non-zeros
5. Accumulate results in float32, cast at end
''')
