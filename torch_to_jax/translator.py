"""
PyTorch to JAX Translator using LLM.

Translates KernelBench PyTorch workloads to equivalent JAX code.
"""

import re
from typing import Optional, Tuple
from src.llm_client import LLMClient


TRANSLATION_SYSTEM_PROMPT = """You are an expert at translating PyTorch code to JAX.

Key translation rules:
1. Replace `torch` with `jax.numpy` (as `jnp`)
2. Replace `torch.nn` operations with `jax` equivalents
3. Replace `torch.rand`/`torch.randn` with `jax.random.uniform`/`jax.random.normal`
4. Use `@jax.jit` for compiled functions
5. JAX arrays are immutable - no in-place operations
6. Use `jax.nn` for activation functions (relu, softmax, etc.)
7. Replace `torch.matmul` with `jnp.dot` or `jnp.matmul`
8. Replace `F.conv2d` with `jax.lax.conv_general_dilated`
9. For pooling, use `jax.lax.reduce_window`
10. For normalization, compute manually or use flax.linen

Output ONLY valid Python code. No explanations."""


TRANSLATION_PROMPT_TEMPLATE = '''Translate this PyTorch workload to JAX.

## PyTorch Code:
```python
{pytorch_code}
```

## Requirements:
1. Keep the same class name `Model` with a `forward` method
2. Keep `get_inputs()` and `get_init_inputs()` functions
3. Use `jax.random.PRNGKey(0)` for random generation
4. Match input/output shapes exactly
5. Use `@jax.jit` on the forward method or computation
6. Handle any model initialization parameters

## Output Format:
Return ONLY the complete JAX Python code, starting with imports.
Do NOT include markdown code blocks or explanations.
'''


REFINEMENT_PROMPT_TEMPLATE = '''The JAX translation has an error. Fix it.

## Original PyTorch:
```python
{pytorch_code}
```

## Current JAX (with error):
```python
{jax_code}
```

## Error:
{error}

## Instructions:
Fix the JAX code to resolve this error. Return ONLY the complete fixed JAX code.
Do NOT include markdown code blocks or explanations.
'''


def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response, handling markdown blocks."""
    # Try to extract from markdown code block
    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If no code block, assume the whole response is code
    # Remove any leading/trailing non-code text
    lines = response.strip().split('\n')
    
    # Find first line that looks like Python code
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from ') or line.startswith('#'):
            start_idx = i
            break
    
    return '\n'.join(lines[start_idx:]).strip()


class PyTorchToJAXTranslator:
    """Translates PyTorch code to JAX using LLM."""
    
    def __init__(self, provider: str = "bedrock", model: Optional[str] = None):
        """
        Initialize translator.
        
        Args:
            provider: LLM provider ("bedrock" or "gemini")
            model: Model name (optional)
        """
        self.client = LLMClient(provider=provider, model=model)
        self.provider = provider
    
    def translate(self, pytorch_code: str, max_attempts: int = 3) -> Tuple[str, bool, str]:
        """
        Translate PyTorch code to JAX.
        
        Args:
            pytorch_code: Original PyTorch code
            max_attempts: Maximum translation attempts
            
        Returns:
            Tuple of (jax_code, success, error_message)
        """
        prompt = TRANSLATION_PROMPT_TEMPLATE.format(pytorch_code=pytorch_code)
        
        try:
            response = self.client.generate(
                prompt=prompt,
                system=TRANSLATION_SYSTEM_PROMPT,
                max_tokens=8192,
                temperature=0.2
            )
            jax_code = extract_code_from_response(response)
            return jax_code, True, ""
        except Exception as e:
            return "", False, f"Translation failed: {str(e)}"
    
    def refine(self, pytorch_code: str, jax_code: str, error: str) -> Tuple[str, bool, str]:
        """
        Refine JAX code based on error feedback.
        
        Args:
            pytorch_code: Original PyTorch code
            jax_code: Current JAX code with error
            error: Error message to fix
            
        Returns:
            Tuple of (refined_jax_code, success, error_message)
        """
        prompt = REFINEMENT_PROMPT_TEMPLATE.format(
            pytorch_code=pytorch_code,
            jax_code=jax_code,
            error=error
        )
        
        try:
            response = self.client.generate(
                prompt=prompt,
                system=TRANSLATION_SYSTEM_PROMPT,
                max_tokens=8192,
                temperature=0.2
            )
            refined_code = extract_code_from_response(response)
            return refined_code, True, ""
        except Exception as e:
            return jax_code, False, f"Refinement failed: {str(e)}"


def test_translator():
    """Test the translator with a simple example."""
    pytorch_code = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """Simple model that performs a single square matrix multiplication (C = A * B)"""
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

N = 2048

def get_inputs():
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    return [A, B]

def get_init_inputs():
    return []
'''
    
    print("Testing PyTorch to JAX translator...")
    print("\nOriginal PyTorch:")
    print(pytorch_code[:200] + "...")
    
    translator = PyTorchToJAXTranslator(provider="bedrock")
    jax_code, success, error = translator.translate(pytorch_code)
    
    if success:
        print("\n✅ Translation successful!")
        print("\nGenerated JAX code:")
        print(jax_code)
    else:
        print(f"\n❌ Translation failed: {error}")


if __name__ == "__main__":
    test_translator()

