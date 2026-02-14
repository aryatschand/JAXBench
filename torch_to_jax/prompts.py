"""
Translation Prompts

Prompts for PyTorch to JAX translation.
"""

SYSTEM_PROMPT = """You are an expert at translating PyTorch code to JAX.

Key translation rules:
1. PyTorch uses NCHW format, JAX typically uses NHWC - handle conversion
2. Replace torch.nn.Module with a JAX function or flax.linen.Module
3. Replace torch operations with jax.numpy equivalents
4. Use jax.random for random number generation (requires explicit keys)
5. Generated JAX models should include a set_weights() method for weight transfer

Common mappings:
- torch.tensor -> jnp.array
- torch.nn.Linear -> Dense layer or manual matmul
- torch.nn.Conv2d -> jax.lax.conv_general_dilated
- torch.relu -> jax.nn.relu
- torch.softmax -> jax.nn.softmax
- torch.matmul -> jnp.matmul or @
- .to(device) -> (not needed in JAX, uses default device)
- .cuda() -> (not needed in JAX)

Important:
- Always include type hints
- Make the code JIT-compatible (avoid Python control flow that depends on values)
- Handle batch dimensions correctly
"""

TRANSLATION_TEMPLATE = """Translate the following PyTorch code to JAX.

PyTorch code:
```python
{pytorch_code}
```

Requirements:
1. Create a JAX function or class that matches the PyTorch behavior
2. Include a set_weights() method to transfer weights from PyTorch
3. Make sure the code is JIT-compatible
4. Handle any format conversions (NCHW -> NHWC if needed)

Respond with only the JAX code, no explanations.
"""

REFINEMENT_TEMPLATE = """The previous JAX translation had an error:

Error:
{error}

Previous JAX code:
```python
{jax_code}
```

Original PyTorch code:
```python
{pytorch_code}
```

Fix the error and provide the corrected JAX code. Respond with only the code.
"""
