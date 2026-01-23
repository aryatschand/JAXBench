#!/usr/bin/env python3
"""
Test LLM client connectivity.

This test verifies that:
1. AWS Bedrock (Claude) is accessible
2. Google Gemini API is accessible
3. LLM can generate valid JAX code

Usage:
    python tests/test_llm_client.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import LLMClient, BedrockClient, GeminiClient


def test_bedrock():
    """Test AWS Bedrock connectivity."""
    print("=" * 60)
    print("Test 1: AWS Bedrock (Claude)")
    print("=" * 60)
    
    try:
        client = BedrockClient(model="haiku")  # Use Haiku for faster/cheaper test
        
        response = client.generate(
            prompt="Say 'Bedrock test passed!' in exactly 4 words.",
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"  ✅ Bedrock connected successfully")
        print(f"     Response: {response[:100]}")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_gemini():
    """Test Google Gemini API connectivity."""
    print("\n" + "=" * 60)
    print("Test 2: Google Gemini")
    print("=" * 60)
    
    try:
        client = GeminiClient(model="gemini-2.0-flash")
        
        response = client.generate(
            prompt="Say 'Gemini test passed!' in exactly 4 words.",
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"  ✅ Gemini connected successfully")
        print(f"     Response: {response[:100]}")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def test_jax_generation():
    """Test that LLM can generate valid JAX code."""
    print("\n" + "=" * 60)
    print("Test 3: JAX Code Generation")
    print("=" * 60)
    
    try:
        client = LLMClient(provider="bedrock", model="haiku")
        
        prompt = """Write a simple JAX function that performs element-wise ReLU activation.
        
Output ONLY the Python code, no explanations:

```python
import jax.numpy as jnp

def relu(x):
    # Your implementation here
```"""
        
        response = client.generate(
            prompt=prompt,
            system="You are a JAX expert. Output only valid Python code.",
            max_tokens=200,
            temperature=0.1
        )
        
        # Check if response contains valid JAX code
        has_jax = "jax" in response.lower() or "jnp" in response.lower()
        has_relu = "relu" in response.lower()
        
        if has_jax and has_relu:
            print(f"  ✅ JAX code generation successful")
            print(f"     Generated code preview:")
            for line in response.split('\n')[:5]:
                print(f"       {line}")
            return True
        else:
            print(f"  ⚠️  Response may not be valid JAX code")
            print(f"     Response: {response[:200]}")
            return False
        
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False


def main():
    """Run all LLM client tests."""
    print("\n" + "=" * 60)
    print("JAXBench LLM Client Tests")
    print("=" * 60)
    
    results = {}
    
    results["bedrock"] = test_bedrock()
    results["gemini"] = test_gemini()
    results["jax_generation"] = test_jax_generation()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        if not passed:
            all_passed = False
        print(f"  {test_name}: {status}")
    
    print()
    if all_passed:
        print("✅ All LLM client tests passed!")
        return 0
    else:
        print("⚠️  Some LLM tests failed. Translation may still work with available providers.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

