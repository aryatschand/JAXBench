"""
Pallas Translator - JAX to Pallas TPU Kernel Translation.

Translates standard JAX code to optimized Pallas TPU kernels using LLM.
"""

import re
from typing import Optional, Tuple
from src.llm_client import LLMClient
from src.pallas_prompts import (
    PALLAS_SYSTEM_PROMPT,
    PALLAS_TRANSLATION_TEMPLATE,
    PALLAS_REFINEMENT_TEMPLATE,
    detect_operation_type,
    get_optimization_strategy,
    get_error_fix_guidance,
)


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


class PallasTranslator:
    """Translates JAX code to Pallas TPU kernels using LLM."""

    def __init__(self, provider: str = "bedrock", model: Optional[str] = None):
        """
        Initialize Pallas translator.

        Args:
            provider: LLM provider ("bedrock" or "gemini")
            model: Model name (defaults to "opus" for stronger reasoning)
        """
        # Default to Opus for Pallas generation - needs strong reasoning
        self.model = model or "opus"
        self.client = LLMClient(provider=provider, model=self.model)
        self.provider = provider

    def translate_to_pallas(
        self,
        jax_code: str,
        task_name: str = "unknown",
        max_attempts: int = 1
    ) -> Tuple[str, bool, str]:
        """
        Translate JAX code to Pallas TPU kernel.

        Args:
            jax_code: Original JAX code
            task_name: Name of the task for context
            max_attempts: Maximum translation attempts

        Returns:
            Tuple of (pallas_code, success, error_message)
        """
        # Detect operation type for targeted optimization strategy
        operation_type = detect_operation_type(jax_code)
        optimization_strategy = get_optimization_strategy(operation_type)

        # Build prompt with optimization strategy
        prompt = PALLAS_TRANSLATION_TEMPLATE.format(
            jax_code=jax_code,
            task_name=task_name,
            operation_type=operation_type,
            optimization_strategy=optimization_strategy,
        )

        try:
            response = self.client.generate(
                prompt=prompt,
                system=PALLAS_SYSTEM_PROMPT,
                max_tokens=12000,
                temperature=0.2
            )
            pallas_code = extract_code_from_response(response)

            if not pallas_code.strip():
                return "", False, "Empty response from LLM"

            return pallas_code, True, ""

        except Exception as e:
            return "", False, f"Translation failed: {str(e)}"

    def refine_pallas(
        self,
        jax_code: str,
        pallas_code: str,
        error: str
    ) -> Tuple[str, bool, str]:
        """
        Refine Pallas code based on error feedback.

        Args:
            jax_code: Original JAX code
            pallas_code: Current Pallas code with error
            error: Error message to fix

        Returns:
            Tuple of (refined_pallas_code, success, error_message)
        """
        # Get specific fix guidance based on error
        fix_guidance = get_error_fix_guidance(error)

        prompt = PALLAS_REFINEMENT_TEMPLATE.format(
            jax_code=jax_code,
            pallas_code=pallas_code,
            error=error,
            fix_guidance=fix_guidance,
        )

        try:
            response = self.client.generate(
                prompt=prompt,
                system=PALLAS_SYSTEM_PROMPT,
                max_tokens=12000,
                temperature=0.2
            )
            refined_code = extract_code_from_response(response)

            if not refined_code.strip():
                return pallas_code, False, "Empty response from LLM"

            return refined_code, True, ""

        except Exception as e:
            return pallas_code, False, f"Refinement failed: {str(e)}"


def test_pallas_translator():
    """Test the Pallas translator with a simple example."""
    jax_code = '''import jax
import jax.numpy as jnp

class Model:
    """Simple model that performs a single square matrix multiplication (C = A * B)"""
    def __init__(self):
        pass

    def forward(self, A, B):
        return jnp.matmul(A, B)

N = 4096

def get_inputs():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    A = jax.random.uniform(key1, shape=(N, N))
    B = jax.random.uniform(key2, shape=(N, N))
    return [A, B]

def get_init_inputs():
    return []
'''

    print("Testing Pallas Translator...")
    print("\nOriginal JAX:")
    print(jax_code[:300] + "...")

    translator = PallasTranslator(provider="bedrock", model="opus")
    pallas_code, success, error = translator.translate_to_pallas(
        jax_code,
        task_name="Square_matrix_multiplication"
    )

    if success:
        print("\n✅ Translation successful!")
        print("\nGenerated Pallas code:")
        print(pallas_code)
    else:
        print(f"\n❌ Translation failed: {error}")


if __name__ == "__main__":
    test_pallas_translator()
