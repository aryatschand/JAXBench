"""Stage 3: Aggregate evaluation results into a summary report.

Usage:
    python -m pallas_eval.report
    python -m pallas_eval.report --input pallas_eval/results/eval_results.json
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_report(data: dict) -> str:
    results = data["results"]

    lines = ["# Pallas Evaluation Report\n"]

    for model_key in sorted(set(r.get("model", "?") for r in results)):
        model_results = [r for r in results if r.get("model") == model_key]
        model_name = {"gpt53": "GPT-5.3", "gemini3": "Gemini-3.1-Pro"}.get(model_key, model_key)

        lines.append(f"\n## {model_name}\n")

        for suite in ["jaxkernelbench", "priority_kernels"]:
            suite_results = [r for r in model_results if r.get("suite") == suite]
            if not suite_results:
                continue

            successes = [r for r in suite_results if r.get("status") == "success"]
            errors = [r for r in suite_results if r.get("status") != "success"]
            correct = [r for r in successes if r.get("correct")]
            faster = [r for r in successes if r.get("speedup", 0) > 1]
            correct_and_faster = [r for r in correct if r.get("speedup", 0) > 1]

            suite_label = suite.replace("_", " ").title()
            lines.append(f"### {suite_label}\n")
            lines.append(f"| Metric | Count |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Total | {len(suite_results)} |")
            lines.append(f"| Ran successfully | {len(successes)} |")
            lines.append(f"| Errors (crash/timeout) | {len(errors)} |")
            lines.append(f"| Correct output | {len(correct)} |")
            lines.append(f"| Faster than baseline | {len(faster)} |")
            lines.append(f"| Correct AND faster | {len(correct_and_faster)} |")
            lines.append("")

            if suite == "jaxkernelbench":
                for level in ["level1", "level2"]:
                    lvl_results = [r for r in successes if r.get("level") == level]
                    if not lvl_results:
                        continue
                    lvl_correct = sum(1 for r in lvl_results if r.get("correct"))
                    lvl_faster = sum(1 for r in lvl_results if r.get("speedup", 0) > 1)
                    lines.append(f"**{level}**: {len(lvl_results)} ran, {lvl_correct} correct, {lvl_faster} faster\n")

            if successes:
                speedups = [r["speedup"] for r in successes if "speedup" in r]
                if speedups:
                    import statistics
                    lines.append(f"Speedup stats (over ran): median={statistics.median(speedups):.2f}x, "
                                 f"mean={statistics.mean(speedups):.2f}x, "
                                 f"max={max(speedups):.2f}x, min={min(speedups):.2f}x\n")

            lines.append("| Workload | Correct | Orig (ms) | Gen (ms) | Speedup |")
            lines.append("|----------|---------|-----------|----------|---------|")
            for r in sorted(successes, key=lambda x: x.get("speedup", 0), reverse=True):
                c = "yes" if r.get("correct") else "NO"
                lines.append(
                    f"| {r['name']} | {c} | {r.get('original_ms',0):.2f} | "
                    f"{r.get('generated_ms',0):.2f} | {r.get('speedup',0):.2f}x |"
                )
            if errors:
                for r in errors:
                    err = r.get("error", "")[:40]
                    lines.append(f"| {r['name']} | ERROR | — | — | {err} |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--input", default=str(RESULTS_DIR / "eval_results.json"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "report.md"))
    args = parser.parse_args()

    data = load_results(args.input)
    report = build_report(data)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(report)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
