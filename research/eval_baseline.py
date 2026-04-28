# ABOUTME: Measure unoptimized-prompt baseline scores for SLMs on the val set.
# ABOUTME: Reuses QueryRouter + routing_metric from dspy_optimizer; no DSPy compile.
"""
Baseline eval — what score does a model get with the default signature
docstring and no few-shot demos, i.e. with **no DSPy optimization at all**?

This is the reference point that the Stage 4 MIPROv2/BFRS/COPRO numbers
need in order for uplift vs. default prompting to be interpretable.

Usage:
    python research/eval_baseline.py \
        --model qwen3.5:0.8b --task-base http://127.0.0.1:11435

Outputs a single number to stdout; also appends a line to
`research/artifacts/baseline_val_scores.jsonl` so the numbers accumulate
across runs.
"""
import argparse
import json
import time
from pathlib import Path

import dspy

from dspy_optimizer import (
    QueryRouter,
    routing_metric,
    load_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an unoptimized QueryRouter on the val set."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model ID (e.g. gemma4:e2b, qwen3.5:0.8b).",
    )
    parser.add_argument(
        "--task-base",
        default="http://localhost:11434",
        help="Ollama base URL for the model. Default: 11434.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Parallelism for Evaluate. Default: 4.",
    )
    args = parser.parse_args()

    # Qwen thinking-mode workaround (mirrors dspy_optimizer.py behavior).
    extra = {"think": False} if args.model.lower().startswith("qwen") else {}

    lm = dspy.LM(
        model=f"ollama_chat/{args.model}",
        api_base=args.task_base,
        temperature=0.0,
        **extra,
    )
    dspy.configure(lm=lm)
    print(f"LM: {args.model} via {args.task_base}")

    val_path = Path(__file__).parent / "data" / "generated" / "val.jsonl"
    val_examples = load_examples(str(val_path))
    print(f"Val examples: {len(val_examples)}")

    router = QueryRouter()  # uncompiled — default signature, no demos.

    print(f"\nEvaluating baseline (num_threads={args.num_threads})...")
    evaluator = dspy.Evaluate(
        devset=val_examples,
        metric=routing_metric,
        num_threads=args.num_threads,
        display_progress=True,
    )
    start = time.time()
    result = evaluator(router)
    elapsed = time.time() - start
    score = float(result) if not isinstance(result, (int, float)) else result
    print(f"\nBaseline val score: {score:.4f}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Append to running log so we can accumulate across models.
    out_path = Path(__file__).parent / "artifacts" / "baseline_val_scores.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps({
            "model": args.model,
            "val_score": round(score, 4),
            "elapsed_sec": round(elapsed, 1),
            "task_base": args.task_base,
        }) + "\n")
    print(f"Logged to {out_path}")


if __name__ == "__main__":
    main()
