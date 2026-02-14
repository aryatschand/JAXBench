"""
PyTorch to JAX Translation Pipeline

Translates PyTorch nn.Module implementations to JAX using LLMs.

Usage:
    python -m torch_to_jax.run --level 1 --tasks 10 --keep-tpu
"""

from .llm_client import LLMClient
from .translator import translate_pytorch_to_jax
from .pipeline import TranslationPipeline
