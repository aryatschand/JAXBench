"""
JAXBench Pipeline - Translate KernelBench PyTorch to JAX.

Main orchestration script that:
1. Reads KernelBench Level 1 PyTorch workloads
2. Translates to JAX using LLM (Opus or Gemini)
3. Validates on GCP TPU
4. Saves successful translations to jaxbench/level1/
"""

import os
import json
import time
import glob
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Import our modules
from src.llm_client import LLMClient
from src.translator import PyTorchToJAXTranslator, extract_code_from_response
from src.validator import BatchValidator

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELBENCH_DIR = os.path.join(BASE_DIR, "KernelBench", "KernelBench", "level1")
OUTPUT_DIR = os.path.join(BASE_DIR, "jaxbench", "level1")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


@dataclass
class TaskResult:
    """Result of translating one task."""
    task_id: str
    task_name: str
    pytorch_file: str
    success: bool
    attempts: int
    jax_code: Optional[str]
    compilation_success: bool
    correctness_success: bool
    max_diff: Optional[float]
    jax_ms: Optional[float]
    pytorch_ms: Optional[float]
    speedup: Optional[float]
    error: str
    provider: str


def get_level1_tasks(limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Get KernelBench Level 1 tasks.
    
    Returns list of dicts with 'task_id', 'task_name', 'filepath', 'code'
    """
    tasks = []
    
    # Get all Python files in level1
    pattern = os.path.join(KERNELBENCH_DIR, "*.py")
    files = glob.glob(pattern)
    
    # Sort numerically by task ID
    def get_task_num(f):
        try:
            return int(os.path.basename(f).split("_")[0])
        except:
            return 999
    files = sorted(files, key=get_task_num)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract task number and name
        # Format: "1_Square_matrix_multiplication_.py"
        parts = filename.replace(".py", "").split("_", 1)
        task_id = parts[0]
        task_name = parts[1] if len(parts) > 1 else filename
        
        with open(filepath, 'r') as f:
            code = f.read()
        
        tasks.append({
            "task_id": task_id,
            "task_name": task_name,
            "filepath": filepath,
            "filename": filename,
            "code": code,
        })
    
    if limit:
        tasks = tasks[:limit]
    
    return tasks


def save_jax_code(task: Dict, jax_code: str):
    """Save translated JAX code to output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path = os.path.join(OUTPUT_DIR, task["filename"])
    
    # Add header comment
    header = f'''"""
JAXBench Level 1 - {task["task_name"]}

Translated from KernelBench PyTorch to JAX.
Original: {task["filename"]}
"""

'''
    
    with open(output_path, 'w') as f:
        f.write(header + jax_code)
    
    return output_path


def run_pipeline(
    num_tasks: int = 10,
    provider: str = "bedrock",
    model: Optional[str] = None,
    max_attempts: int = 3,
    tpu_type: str = "v6e-1",
):
    """
    Run the JAXBench translation pipeline.
    
    Args:
        num_tasks: Number of tasks to process
        provider: LLM provider ("bedrock" or "gemini")
        model: Model name (optional)
        max_attempts: Max translation attempts per task
        tpu_type: TPU type for validation
    """
    print("=" * 70)
    print("JAXBench Translation Pipeline")
    print("=" * 70)
    print(f"Provider: {provider}")
    print(f"Model: {model or 'default'}")
    print(f"Tasks: {num_tasks}")
    print(f"Max attempts: {max_attempts}")
    print(f"TPU type: {tpu_type}")
    print("=" * 70)
    
    # Initialize components
    translator = PyTorchToJAXTranslator(provider=provider, model=model)
    validator = BatchValidator(tpu_type=tpu_type)
    
    # Get tasks
    tasks = get_level1_tasks(limit=num_tasks)
    print(f"\nLoaded {len(tasks)} tasks from KernelBench Level 1")
    
    # Results tracking
    results: List[TaskResult] = []
    
    # Process each task
    for i, task in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(tasks)}] Task {task['task_id']}: {task['task_name']}")
        print(f"{'='*60}")
        
        result = TaskResult(
            task_id=task["task_id"],
            task_name=task["task_name"],
            pytorch_file=task["filename"],
            success=False,
            attempts=0,
            jax_code=None,
            compilation_success=False,
            correctness_success=False,
            max_diff=None,
            jax_ms=None,
            pytorch_ms=None,
            speedup=None,
            error="",
            provider=provider,
        )
        
        pytorch_code = task["code"]
        jax_code = None
        last_error = ""
        
        for attempt in range(max_attempts):
            result.attempts = attempt + 1
            print(f"\n  Attempt {attempt + 1}/{max_attempts}")
            
            # Translate
            if attempt == 0:
                print("  Translating PyTorch to JAX...")
                jax_code, success, error = translator.translate(pytorch_code)
            else:
                print(f"  Refining based on error: {last_error[:80]}...")
                jax_code, success, error = translator.refine(pytorch_code, jax_code, last_error)
            
            if not success:
                last_error = error
                print(f"  Translation failed: {error[:100]}")
                continue
            
            print(f"  Generated {len(jax_code)} chars of JAX code")
            result.jax_code = jax_code
            
            # Validate on TPU
            print("  Validating on TPU...")
            validation_tasks = [{
                "task_id": task["task_id"],
                "jax_code": jax_code,
                "pytorch_code": pytorch_code,
            }]
            
            val_results = validator.validate_batch(validation_tasks, timeout_per_task=120)
            
            if not val_results or "error" in val_results[0]:
                last_error = val_results[0].get("error", "Unknown validation error") if val_results else "No validation results"
                print(f"  Validation error: {last_error[:100]}")
                continue
            
            val = val_results[0]
            
            # Check compilation
            result.compilation_success = val.get("compilation", {}).get("success", False)
            if not result.compilation_success:
                last_error = val.get("compilation", {}).get("error", "Compilation failed")
                print(f"  Compilation failed: {last_error[:100]}")
                continue
            
            print("  ✓ Compilation successful")
            
            # Check correctness
            result.correctness_success = val.get("correctness", {}).get("success", False)
            result.max_diff = val.get("correctness", {}).get("max_diff")
            
            if not result.correctness_success:
                last_error = val.get("correctness", {}).get("error", "Correctness check failed")
                print(f"  Correctness failed: {last_error[:100]}")
                continue
            
            print(f"  ✓ Correctness passed (max_diff: {result.max_diff})")
            
            # Get performance
            result.jax_ms = val.get("performance", {}).get("jax_ms")
            result.pytorch_ms = val.get("performance", {}).get("pytorch_ms")
            result.speedup = val.get("performance", {}).get("speedup")
            
            print(f"  ✓ Performance: JAX={result.jax_ms}ms, speedup={result.speedup}x")
            
            # Success!
            result.success = True
            result.error = ""
            
            # Save the JAX code
            output_path = save_jax_code(task, jax_code)
            print(f"  ✓ Saved to: {output_path}")
            
            break
        
        if not result.success:
            result.error = last_error
            print(f"  ✗ Failed after {result.attempts} attempts")
        
        results.append(result)
    
    # Cleanup TPU
    print("\nCleaning up TPU...")
    validator.cleanup()
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"translation_{provider}_{int(time.time())}.json")
    
    results_data = {
        "provider": provider,
        "model": model,
        "num_tasks": len(tasks),
        "tpu_type": tpu_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": len(results),
            "successful": sum(1 for r in results if r.success),
            "compiled": sum(1 for r in results if r.compilation_success),
            "correct": sum(1 for r in results if r.correctness_success),
        },
        "results": [asdict(r) for r in results],
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {results_data['summary']['successful']}")
    print(f"Compiled: {results_data['summary']['compiled']}")
    print(f"Correct: {results_data['summary']['correct']}")
    print(f"\nResults saved to: {results_path}")
    
    # Print per-task results
    print("\nPer-task results:")
    for r in results:
        status = "✓" if r.success else "✗"
        perf = f"JAX={r.jax_ms}ms, {r.speedup}x" if r.jax_ms else "N/A"
        print(f"  {status} {r.task_id}: {r.task_name[:30]:30} | {perf}")
    
    return results_data


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="JAXBench Translation Pipeline")
    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks to process")
    parser.add_argument("--provider", choices=["bedrock", "gemini"], default="bedrock",
                        help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--attempts", type=int, default=3, help="Max attempts per task")
    parser.add_argument("--tpu", type=str, default="v6e-1", help="TPU type")
    
    args = parser.parse_args()
    
    run_pipeline(
        num_tasks=args.tasks,
        provider=args.provider,
        model=args.model,
        max_attempts=args.attempts,
        tpu_type=args.tpu,
    )


if __name__ == "__main__":
    main()

