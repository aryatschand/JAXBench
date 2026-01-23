"""
JAXBench - PyTorch to JAX Translation Library

This package provides tools for translating PyTorch code to JAX
and benchmarking on TPUs.
"""

from .llm_client import LLMClient, BedrockClient, GeminiClient
from .tpu_manager import TPUManager, get_credentials

__all__ = [
    "LLMClient",
    "BedrockClient", 
    "GeminiClient",
    "TPUManager",
    "get_credentials",
]
