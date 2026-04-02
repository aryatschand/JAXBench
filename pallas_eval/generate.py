"""Stage 1: Generate Pallas kernels from JAX workloads using GPT 5.3 and Gemini 3.1 Pro.

Usage:
    python -m pallas_eval.generate                          # all workloads, both models
    python -m pallas_eval.generate --model gpt53            # GPT 5.3 only
    python -m pallas_eval.generate --model gemini3          # Gemini 3.1 Pro only
    python -m pallas_eval.generate --suite jaxkernelbench   # only jaxkernelbench
    python -m pallas_eval.generate --suite priority         # only priority_kernels
    python -m pallas_eval.generate --dry-run                # list workloads, don't call APIs
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import pallas_eval.prompts as prompts_module
from pallas_eval.prompts import (
    JAXKERNELBENCH_PROMPT,
    PRIORITY_KERNEL_PROMPT,
)

logger = logging.getLogger("pallas_eval.generate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = Path(__file__).resolve().parent / "generated"
JAXKERNELBENCH_DIR = PROJECT_ROOT / "jaxkernelbench"
PRIORITY_DIR = PROJECT_ROOT / "priority_kernels"

MODELS = {
    "gpt53": {
        "display": "GPT-5.3",
        "api": "openai",
        "model_id": "gpt-5.3-chat-latest",
    },
    "gemini3": {
        "display": "Gemini-3.1-Pro",
        "api": "gemini",
        "model_id": "gemini-3.1-pro-preview",
    },
}


# ---------------------------------------------------------------------------
# Workload discovery
# ---------------------------------------------------------------------------

def discover_jaxkernelbench() -> list[dict]:
    """Return list of {name, path, level} for all jaxkernelbench workloads."""
    workloads = []
    for level in ["level1", "level2"]:
        level_dir = JAXKERNELBENCH_DIR / level
        if not level_dir.exists():
            continue
        for f in sorted(level_dir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            workloads.append({
                "name": f.stem,
                "path": f,
                "level": level,
                "suite": "jaxkernelbench",
            })
    return workloads


def discover_priority_kernels() -> list[dict]:
    """Return list of {name, path} for all priority_kernels baselines."""
    workloads = []
    for entry in sorted(PRIORITY_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        baseline = entry / "baseline.py"
        if baseline.exists():
            workloads.append({
                "name": entry.name,
                "path": baseline,
                "level": None,
                "suite": "priority_kernels",
            })
    return workloads


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------

def call_openai(source_code: str, prompt_template: str, model_id: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user_msg = prompt_template.format(source_code=source_code)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": prompts_module.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_completion_tokens=16384,
    )
    return resp.choices[0].message.content


def call_gemini(source_code: str, prompt_template: str, model_id: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name=model_id,
        system_instruction=prompts_module.SYSTEM_PROMPT,
    )
    user_msg = prompt_template.format(source_code=source_code)
    resp = model.generate_content(
        user_msg,
        generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=16384),
    )
    return resp.text


CALLERS = {
    "openai": call_openai,
    "gemini": call_gemini,
}


def extract_python(raw: str) -> str:
    """Strip markdown fences if the LLM wrapped the code."""
    match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_one(workload: dict, model_key: str) -> dict:
    """Generate Pallas code for one workload with one model. Returns status dict."""
    model_cfg = MODELS[model_key]
    source_code = workload["path"].read_text()

    if workload["suite"] == "jaxkernelbench":
        prompt_template = JAXKERNELBENCH_PROMPT
        out_subdir = f"jaxkernelbench_{workload['level']}"
    else:
        prompt_template = PRIORITY_KERNEL_PROMPT
        out_subdir = "priority_kernels"

    out_dir = GENERATED_DIR / model_key / out_subdir
    out_file = out_dir / f"{workload['name']}.py"

    if out_file.exists():
        logger.info(f"  SKIP {workload['name']} ({model_cfg['display']}) — already generated")
        return {"name": workload["name"], "model": model_key, "status": "skipped"}

    caller = CALLERS[model_cfg["api"]]
    try:
        t0 = time.time()
        raw = caller(source_code, prompt_template, model_cfg["model_id"])
        elapsed = time.time() - t0
        code = extract_python(raw)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file.write_text(code + "\n")
        logger.info(f"  OK   {workload['name']} ({model_cfg['display']}) — {elapsed:.1f}s, {len(code)} chars")
        return {"name": workload["name"], "model": model_key, "status": "success", "elapsed": round(elapsed, 1)}
    except Exception as e:
        logger.error(f"  FAIL {workload['name']} ({model_cfg['display']}): {e}")
        return {"name": workload["name"], "model": model_key, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Generate Pallas kernels via LLMs")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Run only this model (default: both)")
    parser.add_argument("--suite", choices=["jaxkernelbench", "priority", "all"], default="all",
                        help="Which workload suite(s)")
    parser.add_argument("--dry-run", action="store_true", help="List workloads without calling APIs")
    parser.add_argument("--output", default=None, help="Save generation log JSON to this path")
    parser.add_argument("--prompt-file", default=None,
                        help="Use a custom system prompt from this file (e.g. GEPA-optimized)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

    workloads = []
    if args.suite in ("jaxkernelbench", "all"):
        workloads.extend(discover_jaxkernelbench())
    if args.suite in ("priority", "all"):
        workloads.extend(discover_priority_kernels())

    model_keys = [args.model] if args.model else list(MODELS.keys())

    if args.prompt_file:
        custom_prompt = Path(args.prompt_file).read_text().strip()
        logger.info(f"Using custom prompt from {args.prompt_file} ({len(custom_prompt)} chars)")
        prompts_module.SYSTEM_PROMPT = custom_prompt

    logger.info(f"Workloads: {len(workloads)}  |  Models: {[MODELS[m]['display'] for m in model_keys]}")
    if args.dry_run:
        for w in workloads:
            print(f"  [{w['suite']}] {w['name']}")
        print(f"\nTotal: {len(workloads)} workloads x {len(model_keys)} models = {len(workloads)*len(model_keys)} generations")
        return

    log = []
    for model_key in model_keys:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {MODELS[model_key]['display']}")
        logger.info(f"{'='*60}")
        for i, w in enumerate(workloads, 1):
            logger.info(f"[{i}/{len(workloads)}] {w['suite']}/{w['name']}")
            result = generate_one(w, model_key)
            log.append(result)

    # Summary
    ok = sum(1 for r in log if r["status"] == "success")
    skip = sum(1 for r in log if r["status"] == "skipped")
    fail = sum(1 for r in log if r["status"] == "error")
    logger.info(f"\nDone: {ok} generated, {skip} skipped, {fail} failed  (total {len(log)})")

    log_path = args.output or str(GENERATED_DIR / "generation_log.json")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    logger.info(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
