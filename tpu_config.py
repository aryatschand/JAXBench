"""
TPU Configuration for Modal.

This module provides configuration classes and utilities for TPU allocation.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class TPUConfig:
    """Configuration for TPU allocation on Modal."""
    
    # TPU type (e.g., "tpu-v4-8", "tpu-v5e-4")
    tpu_type: str = "tpu-v4-8"
    
    # Timeout in seconds
    timeout: int = 3600
    
    # Memory limit (optional, Modal handles this)
    memory: Optional[int] = None
    
    # Cloud provider (GCP for TPUs)
    cloud: str = "gcp"
    
    # Python version
    python_version: str = "3.10"
    
    # JAX version
    jax_version: str = "0.4.30"
    
    @property
    def accelerator(self) -> str:
        """Get Modal accelerator string."""
        return self.tpu_type
    
    def to_modal_kwargs(self) -> dict:
        """Convert to Modal function kwargs."""
        kwargs = {
            "cloud": self.cloud,
            "accelerator": self.accelerator,
            "timeout": self.timeout,
        }
        if self.memory:
            kwargs["memory"] = self.memory
        return kwargs


# Predefined TPU configurations
TPU_CONFIGS = {
    # TPU v4 (Puffin)
    "v4-8": TPUConfig(tpu_type="tpu-v4-8"),
    "v4-16": TPUConfig(tpu_type="tpu-v4-16"),
    "v4-32": TPUConfig(tpu_type="tpu-v4-32"),
    
    # TPU v5e (economical)
    "v5e-4": TPUConfig(tpu_type="tpu-v5e-4"),
    "v5e-8": TPUConfig(tpu_type="tpu-v5e-8"),
    "v5e-16": TPUConfig(tpu_type="tpu-v5e-16"),
    
    # TPU v5p (high performance)
    "v5p-8": TPUConfig(tpu_type="tpu-v5p-8"),
}


def get_tpu_config(name: str = "v4-8") -> TPUConfig:
    """Get a predefined TPU configuration by name."""
    if name not in TPU_CONFIGS:
        available = ", ".join(TPU_CONFIGS.keys())
        raise ValueError(f"Unknown TPU config '{name}'. Available: {available}")
    return TPU_CONFIGS[name]


def get_modal_credentials() -> dict:
    """
    Get Modal credentials from environment variables.
    
    Returns dict with 'token_id' and 'token_secret' if available.
    """
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    
    if token_id and token_secret:
        return {
            "token_id": token_id,
            "token_secret": token_secret,
        }
    return {}


def create_tpu_image(config: Optional[TPUConfig] = None):
    """
    Create a Modal image configured for JAX on TPU.
    
    Args:
        config: TPU configuration (uses default if None)
    
    Returns:
        modal.Image configured for TPU
    """
    import modal
    
    if config is None:
        config = TPUConfig()
    
    image = (
        modal.Image.debian_slim(python_version=config.python_version)
        .pip_install(
            f"jax[tpu]=={config.jax_version}",
            extra_options="-f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        )
        .pip_install(
            "flax>=0.8.0",
            "optax>=0.2.0",
            "numpy>=1.24.0",
            "tqdm>=4.65.0",
        )
    )
    
    return image

