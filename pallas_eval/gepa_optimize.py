"""GEPA prompt optimization for Pallas kernel generation.

Uses GEPA to iteratively improve the system prompt that LLMs use to generate
TPU Pallas kernels from JAX code. Evaluates on a diverse set of ~20 workloads.

Usage:
    python -m pallas_eval.gepa_optimize                          # full run
    python -m pallas_eval.gepa_optimize --model gpt53            # optimize for GPT-5.3
    python -m pallas_eval.gepa_optimize --model gemini3          # optimize for Gemini
    python -m pallas_eval.gepa_optimize --max-evals 50           # limit budget
    python -m pallas_eval.gepa_optimize --dry-run                # test setup without running
"""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from gepa.optimize_anything import (
    optimize_anything,
    GEPAConfig,
    EngineConfig,
    ReflectionConfig,
)

from pallas_eval.gepa_evaluator import make_evaluator
from pallas_eval.prompts import SYSTEM_PROMPT
from pallas_eval.tpu import run_ssh, scp_to_tpu

logger = logging.getLogger("pallas_eval.gepa")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PALLAS_EVAL_DIR = Path(__file__).resolve().parent
REMOTE_BASE = "/tmp/pallas_eval"

# 20 diverse workloads: 8 level1, 8 level2, 4 priority
TRAINING_SET = [
    {"name": "1_Square_matrix_multiplication_", "path": "jaxkernelbench/level1/1_Square_matrix_multiplication_.py", "suite": "jaxkernelbench", "level": "level1", "category": "matmul"},
    {"name": "9_Tall_skinny_matrix_multiplication_", "path": "jaxkernelbench/level1/9_Tall_skinny_matrix_multiplication_.py", "suite": "jaxkernelbench", "level": "level1", "category": "matmul"},
    {"name": "3_Batched_matrix_multiplication", "path": "jaxkernelbench/level1/3_Batched_matrix_multiplication.py", "suite": "jaxkernelbench", "level": "level1", "category": "matmul"},
    {"name": "50_conv_standard_2D__square_input__square_kernel", "path": "jaxkernelbench/level1/50_conv_standard_2D__square_input__square_kernel.py", "suite": "jaxkernelbench", "level": "level1", "category": "conv"},
    {"name": "97_ScaledDotProductAttention", "path": "jaxkernelbench/level1/97_ScaledDotProductAttention.py", "suite": "jaxkernelbench", "level": "level1", "category": "attention"},
    {"name": "40_LayerNorm", "path": "jaxkernelbench/level1/40_LayerNorm.py", "suite": "jaxkernelbench", "level": "level1", "category": "norm"},
    {"name": "19_ReLU", "path": "jaxkernelbench/level1/19_ReLU.py", "suite": "jaxkernelbench", "level": "level1", "category": "elementwise"},
    {"name": "48_Mean_reduction_over_a_dimension", "path": "jaxkernelbench/level1/48_Mean_reduction_over_a_dimension.py", "suite": "jaxkernelbench", "level": "level1", "category": "reduction"},
    {"name": "1_Conv2D_ReLU_BiasAdd", "path": "jaxkernelbench/level2/1_Conv2D_ReLU_BiasAdd.py", "suite": "jaxkernelbench", "level": "level2", "category": "fused"},
    {"name": "66_Matmul_Dropout_Softmax", "path": "jaxkernelbench/level2/66_Matmul_Dropout_Softmax.py", "suite": "jaxkernelbench", "level": "level2", "category": "fused"},
    {"name": "28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply", "path": "jaxkernelbench/level2/28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py", "suite": "jaxkernelbench", "level": "level2", "category": "fused"},
    {"name": "6_Conv3d_Softmax_MaxPool_MaxPool", "path": "jaxkernelbench/level2/6_Conv3d_Softmax_MaxPool_MaxPool.py", "suite": "jaxkernelbench", "level": "level2", "category": "fused"},
    {"name": "3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU", "path": "jaxkernelbench/level2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py", "suite": "jaxkernelbench", "level": "level2", "category": "fused"},
    {"name": "41_Gemm_BatchNorm_GELU_ReLU", "path": "jaxkernelbench/level2/41_Gemm_BatchNorm_GELU_ReLU.py", "suite": "jaxkernelbench", "level": "level2", "category": "fused"},
    {"name": "22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish", "path": "jaxkernelbench/level2/22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish.py", "suite": "jaxkernelbench", "level": "level2", "category": "fused"},
    {"name": "84_Gemm_BatchNorm_Scaling_Softmax", "path": "jaxkernelbench/level2/84_Gemm_BatchNorm_Scaling_Softmax.py", "suite": "jaxkernelbench", "level": "level2", "category": "fused"},
    {"name": "flash_attention", "path": "priority_kernels/flash_attention/baseline.py", "suite": "priority_kernels", "level": None, "category": "attention"},
    {"name": "gemm", "path": "priority_kernels/gemm/baseline.py", "suite": "priority_kernels", "level": None, "category": "matmul"},
    {"name": "rms_norm", "path": "priority_kernels/rms_norm/baseline.py", "suite": "priority_kernels", "level": None, "category": "norm"},
    {"name": "gqa_attention", "path": "priority_kernels/gqa_attention/baseline.py", "suite": "priority_kernels", "level": None, "category": "attention"},
]


def build_dataset(workloads: list[dict]) -> list[dict]:
    """Load source code for each workload and build GEPA dataset."""
    dataset = []
    for w in workloads:
        abs_path = PROJECT_ROOT / w["path"]
        if not abs_path.exists():
            logger.warning(f"Workload not found: {abs_path}")
            continue
        source_code = abs_path.read_text()
        dataset.append({
            "name": w["name"],
            "source_code": source_code,
            "suite": w["suite"],
            "level": w.get("level"),
            "category": w.get("category", "unknown"),
            "original_path": str(abs_path),
        })
    return dataset


def setup_tpu():
    """Ensure TPU remote dirs exist and harness is deployed."""
    logger.info("Setting up TPU for GEPA evaluation...")
    run_ssh(f"mkdir -p {REMOTE_BASE}/originals {REMOTE_BASE}/generated", timeout=15)
    scp_to_tpu(str(PALLAS_EVAL_DIR / "eval_harness.py"), f"{REMOTE_BASE}/eval_harness.py")
    run_ssh("pip install -q numpy 2>/dev/null", timeout=60)
    logger.info("TPU ready.")


def main():
    parser = argparse.ArgumentParser(description="Optimize Pallas prompts with GEPA")
    parser.add_argument("--model", choices=["gpt53", "gemini3"], default="gpt53",
                        help="Which LLM to optimize the prompt for (default: gpt53)")
    parser.add_argument("--max-evals", type=int, default=100,
                        help="Max evaluation budget for GEPA (default: 100)")
    parser.add_argument("--reflection-lm", default="openai/gpt-5.3-chat-latest",
                        help="LLM for GEPA's reflection/mutation step")
    parser.add_argument("--minibatch-size", type=int, default=3,
                        help="Number of workloads shown per reflection step")
    parser.add_argument("--run-dir", default=None,
                        help="Directory to save/resume GEPA state (enables checkpointing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build dataset and print config without running")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    dataset = build_dataset(TRAINING_SET)
    logger.info(f"Built dataset: {len(dataset)} workloads")

    # Split 14 train / 6 val
    trainset = dataset[:14]
    valset = dataset[14:]
    logger.info(f"Train: {len(trainset)} | Val: {len(valset)}")

    if args.dry_run:
        print(f"\nSeed prompt ({len(SYSTEM_PROMPT)} chars):")
        print(SYSTEM_PROMPT[:200] + "...")
        print(f"\nTraining workloads ({len(trainset)}):")
        for w in trainset:
            print(f"  [{w['suite']}] {w['name']} ({w['category']})")
        print(f"\nValidation workloads ({len(valset)}):")
        for w in valset:
            print(f"  [{w['suite']}] {w['name']} ({w['category']})")
        print(f"\nModel: {args.model}")
        print(f"Max evals: {args.max_evals}")
        print(f"Reflection LM: {args.reflection_lm}")
        print(f"Minibatch size: {args.minibatch_size}")
        return

    setup_tpu()

    evaluator = make_evaluator(model=args.model)

    run_dir = args.run_dir or str(PALLAS_EVAL_DIR / "gepa_runs" / f"{args.model}")

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_evals,
            parallel=False,  # TPU is single-device, serialize evals
            run_dir=run_dir,
            display_progress_bar=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=args.reflection_lm,
            reflection_minibatch_size=args.minibatch_size,
        ),
    )

    logger.info(f"Starting GEPA optimization for {args.model}")
    logger.info(f"Budget: {args.max_evals} evals | Reflection LM: {args.reflection_lm}")
    logger.info(f"Run dir: {run_dir}")

    result = optimize_anything(
        seed_candidate={"system_prompt": SYSTEM_PROMPT},
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=(
            "Optimize the system prompt so that LLMs generate correct and fast "
            "Pallas TPU kernels from JAX code. The generated code MUST use "
            "jax.experimental.pallas (not vanilla JAX). It must compile and run "
            "on TPU v6e with JAX 0.6.2, produce numerically correct outputs "
            "(allclose with atol=0.01), and ideally be faster than the baseline."
        ),
        background=(
            "Pallas is JAX's kernel language for TPU (Mosaic backend). "
            "Common LLM errors: using GPU-only APIs (pl.load/pl.store, static_argnums), "
            "wrong PrefetchScalarGridSpec constructor args, Python if/else on traced values, "
            "incorrect block shapes (must be divisible by (8,128) for bf16), "
            "1D tensors (TPU requires 2D+), and confusing Triton-style with Mosaic-style. "
            "The prompt must teach these constraints clearly."
        ),
        config=config,
    )

    # Save the optimized prompt
    best_prompt = result.best_candidate["system_prompt"]
    output_path = PALLAS_EVAL_DIR / "gepa_runs" / f"{args.model}" / "best_prompt.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(best_prompt)

    logger.info(f"\n{'='*60}")
    logger.info(f"GEPA optimization complete!")
    logger.info(f"Best prompt saved to: {output_path}")
    logger.info(f"Best prompt length: {len(best_prompt)} chars")
    logger.info(f"{'='*60}")
    print(f"\nOptimized prompt:\n{best_prompt}")


if __name__ == "__main__":
    main()
