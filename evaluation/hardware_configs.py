"""
Hardware Configuration Presets

Pre-defined hardware configurations for common TPU types.
"""

from .evaluate_kernel import HardwareConfig


# TPU v5e configurations
TPU_V5E_4 = HardwareConfig(
    tpu_type="v5e-4",
    tpu_zone="us-central1-b",
    tpu_project="jaxbench",
)

TPU_V5E_8 = HardwareConfig(
    tpu_type="v5e-8",
    tpu_zone="us-central1-b",
    tpu_project="jaxbench",
)

# TPU v4 configurations
TPU_V4_8 = HardwareConfig(
    tpu_type="v4-8",
    tpu_zone="us-central1-b",
    tpu_project="jaxbench",
)

TPU_V4_16 = HardwareConfig(
    tpu_type="v4-16",
    tpu_zone="us-central1-b",
    tpu_project="jaxbench",
)

TPU_V4_32 = HardwareConfig(
    tpu_type="v4-32",
    tpu_zone="us-central1-b",
    tpu_project="jaxbench",
)

# TPU v6e configurations
TPU_V6E_1 = HardwareConfig(
    tpu_type="v6e-1",
    tpu_zone="us-central1-b",
    tpu_project="jaxbench",
)


# Default configuration
DEFAULT_CONFIG = TPU_V5E_4


def get_config(tpu_type: str) -> HardwareConfig:
    """Get hardware config by TPU type string."""
    configs = {
        "v5e-4": TPU_V5E_4,
        "v5e-8": TPU_V5E_8,
        "v4-8": TPU_V4_8,
        "v4-16": TPU_V4_16,
        "v4-32": TPU_V4_32,
        "v6e-1": TPU_V6E_1,
    }
    if tpu_type not in configs:
        raise ValueError(f"Unknown TPU type: {tpu_type}. Available: {list(configs.keys())}")
    return configs[tpu_type]
