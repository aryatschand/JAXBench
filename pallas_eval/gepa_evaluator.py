"""GEPA evaluator for Pallas kernel prompt optimization.

Evaluates a system prompt candidate by:
  1. Calling an LLM with the candidate prompt + a JAX workload
  2. SCP-ing the generated Pallas code + original to the TPU
  3. Running the eval harness on TPU
  4. Returning a score based on correctness and speedup
"""

import json
import os
import re
import tempfile
import time
import traceback
from pathlib import Path

import gepa.optimize_anything as oa
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from pallas_eval.tpu import run_ssh, scp_to_tpu, clear_tpu_state

PALLAS_EVAL_DIR = Path(__file__).resolve().parent
REMOTE_BASE = "/tmp/pallas_eval"
REMOTE_HARNESS = f"{REMOTE_BASE}/eval_harness.py"


def extract_python(raw: str) -> str:
    """Strip markdown fences if the LLM wrapped the code."""
    match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt53") -> str:
    """Call an LLM with the given system+user prompts."""
    if model == "gpt53":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-5.3-chat-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=16384,
        )
        return resp.choices[0].message.content
    elif model == "gemini3":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        m = genai.GenerativeModel(
            model_name="gemini-3.1-pro-preview",
            system_instruction=system_prompt,
        )
        resp = m.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=16384),
        )
        return resp.text
    else:
        raise ValueError(f"Unknown model: {model}")


def run_on_tpu(name: str, suite: str, original_path: str, generated_code: str,
               timeout: int = 120) -> dict:
    """SCP code to TPU, run eval harness, return parsed result."""
    remote_orig = f"{REMOTE_BASE}/originals/{name}_original.py"
    remote_gen = f"{REMOTE_BASE}/generated/{name}_gepa.py"

    scp_to_tpu(original_path, remote_orig)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(generated_code)
        tmp_path = f.name

    try:
        scp_to_tpu(tmp_path, remote_gen)
    finally:
        os.unlink(tmp_path)

    clear_tpu_state()
    time.sleep(0.5)

    cmd = (
        f"PJRT_DEVICE=TPU python3 {REMOTE_HARNESS} "
        f"--original {remote_orig} --generated {remote_gen} "
        f"--suite {suite} --name {name}"
    )

    output = run_ssh(cmd, timeout=timeout)

    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    error_msg = output.strip()[-500:] if output.strip() else "No output"
    return {"name": name, "status": "error", "error": error_msg}


def check_uses_pallas(code: str) -> bool:
    """Verify the generated code actually uses Pallas, not just vanilla JAX."""
    pallas_indicators = [
        "pallas_call",
        "pl.pallas_call",
        "from jax.experimental import pallas",
        "from jax.experimental.pallas",
        "BlockSpec",
        "GridSpec",
        "PrefetchScalarGridSpec",
    ]
    return any(indicator in code for indicator in pallas_indicators)


def build_user_prompt(source_code: str, suite: str) -> str:
    """Build the user prompt for a workload (template part stays fixed)."""
    if suite == "jaxkernelbench":
        return (
            "Below is a JAX workload file. It defines a Model class with a forward() method, "
            "along with get_inputs() and get_init_inputs() functions.\n\n"
            "Your task: rewrite the forward() computation as a Pallas TPU kernel that is faster "
            "than the vanilla JAX version while producing the same outputs.\n\n"
            "You MUST keep the EXACT same file interface:\n"
            "- Same Model class with same __init__ signature\n"
            "- Same forward() method signature and return shape\n"
            "- Same get_inputs() and get_init_inputs() functions\n"
            "- The forward() method should call your Pallas kernel internally\n\n"
            f"Here is the original JAX code:\n\n```python\n{source_code}\n```\n\n"
            "Write the complete replacement Python file using Pallas kernels."
        )
    else:
        return (
            "Below is a JAX workload file for a priority kernel benchmark. It defines CONFIG, "
            "create_inputs(), workload(), and benchmark() functions.\n\n"
            "Your task: rewrite the workload() function using a Pallas TPU kernel that is faster "
            "than the vanilla JAX version while producing the same outputs.\n\n"
            "You MUST keep the EXACT same file interface:\n"
            "- Same CONFIG dict\n"
            "- Same create_inputs() function\n"
            "- Same workload() function signature and return shape/dtype\n"
            "- Same benchmark() function\n"
            "- The workload() function should call your Pallas kernel internally\n\n"
            f"Here is the original JAX code:\n\n```python\n{source_code}\n```\n\n"
            "Write the complete replacement Python file using Pallas kernels."
        )


def evaluate(candidate: dict, example: dict, model: str = "gpt53") -> tuple[float, dict]:
    """GEPA evaluator: generate Pallas code with candidate prompt, eval on TPU.

    Args:
        candidate: dict with key "system_prompt"
        example: dict with keys "name", "source_code", "suite", "original_path"
        model: which LLM to use for generation ("gpt53" or "gemini3")

    Returns:
        (score, side_info) tuple for GEPA
    """
    system_prompt = candidate["system_prompt"]
    name = example["name"]
    suite = example["suite"]
    source_code = example["source_code"]
    original_path = example["original_path"]

    user_prompt = build_user_prompt(source_code, suite)

    side_info = {"Workload": name, "Suite": suite}

    try:
        t0 = time.time()
        raw = call_llm(system_prompt, user_prompt, model=model)
        gen_time = time.time() - t0
        generated_code = extract_python(raw)
        side_info["GenerationTime"] = f"{gen_time:.1f}s"
    except Exception as e:
        side_info["Error"] = f"LLM call failed: {e}"
        return 0.0, side_info

    if not check_uses_pallas(generated_code):
        side_info["Error"] = "Generated code does NOT use Pallas — just vanilla JAX."
        side_info["GeneratedCodeSnippet"] = generated_code[:300]
        return 0.0, side_info

    side_info["UsesPallas"] = True

    try:
        result = run_on_tpu(name, suite, original_path, generated_code, timeout=120)
    except Exception as e:
        side_info["Error"] = f"TPU execution failed: {e}"
        side_info["Traceback"] = traceback.format_exc()[-300:]
        return 0.0, side_info

    if result.get("status") == "error":
        error = result.get("error", "unknown")
        tb = result.get("traceback", "")
        side_info["Error"] = error[:300]
        if tb:
            side_info["Traceback"] = tb[:300]
        return 0.0, side_info

    correct = result.get("correct", False)
    speedup = result.get("speedup", 0.0)
    orig_ms = result.get("original_ms", 0)
    gen_ms = result.get("generated_ms", 0)

    side_info["Correct"] = correct
    side_info["Speedup"] = f"{speedup:.2f}x"
    side_info["OriginalMs"] = f"{orig_ms:.2f}ms"
    side_info["GeneratedMs"] = f"{gen_ms:.2f}ms"
    side_info["MaxDiff"] = result.get("max_diff", "?")

    if not correct:
        side_info["CorrectnessReason"] = result.get("correctness_reason", "unknown")

    # Scoring: correctness is the gate, speedup is the reward
    # 0.0 = wrong, 0.5 = correct but slower, up to 2.0+ = correct and fast
    if correct:
        score = max(speedup, 0.1)
    else:
        score = 0.0

    side_info["scores"] = {
        "correctness": 1.0 if correct else 0.0,
        "speedup": speedup if correct else 0.0,
    }

    return score, side_info


def make_evaluator(model: str = "gpt53"):
    """Create a GEPA-compatible evaluator closure for a specific LLM."""
    def _eval(candidate: dict, example: dict) -> tuple[float, dict]:
        return evaluate(candidate, example, model=model)
    return _eval
