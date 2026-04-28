# ABOUTME: Cross-ablation across (instruction, demos) to attribute MIPROv2 vs BFRS gap.
# ABOUTME: Mixes optimizer artifacts and re-evaluates on the val set; result goes to artifacts/cross_ablation.jsonl.
"""
Cross-ablation: is the MIPROv2-vs-BFRS gap driven by instruction or by demos?

We already have four single-optimizer cells on the val set (600):

    cell                         instruction    demos     val_score
    ---------------------------  -------------  --------  -----------
    (default, none)              default        none      baseline_*
    (default, BFRS demos)        default        BFRS      BFRS_*
    (MIPRO instr, MIPRO demos)   MIPRO          MIPRO     MIPRO_*
    (COPRO instr, none)          COPRO          none      COPRO_*

This script fills in two more cells per model:

    (MIPRO instr, BFRS demos)    MIPRO          BFRS      ← NEW
    (default, MIPRO demos)       default        MIPRO     ← NEW

Putting it all together lets us separate the instruction contribution
from the demo contribution, which so far has been conflated.

Usage:
    python research/eval_cross_ablation.py \
        --model gemma4:e2b --task-base http://127.0.0.1:11435 \
        --mipro-state research/artifacts/optimized_router_state_gemma4_e2b.json \
        --bfrs-state  research/artifacts/optimized_router_state_gemma4_e2b_bfrs.json
"""
import argparse
import json
import time
from pathlib import Path

import dspy

from dspy_optimizer import QueryRouter, routing_metric, load_examples


def build_router(
    instruction: str | None,
    demos: list | None,
) -> QueryRouter:
    """Build a QueryRouter with an optional instruction override and optional
    prebuilt demos.

    Args:
        instruction (str | None): New signature docstring. `None` keeps the
            default docstring defined in `RouteQuery`.
        demos (list | None): List of demo dicts (as produced by DSPy's
            program state serialization) or None to leave demos empty.

    Returns:
        QueryRouter: Router ready to evaluate.
    """
    r = QueryRouter()
    if instruction is not None:
        r.router.predict.signature = r.router.predict.signature.with_instructions(
            instruction
        )
    if demos is not None:
        r.router.predict.demos = list(demos)
    return r


def load_signature_and_demos(state_path: str) -> tuple[str, list]:
    """Load a saved DSPy program state and extract (instruction, demos).

    Args:
        state_path (str): Path to an optimized_router_state_*.json.

    Returns:
        tuple[str, list]: (instruction_string, list_of_demo_dicts)
    """
    r = QueryRouter()
    r.load(state_path)
    return (
        r.router.predict.signature.instructions,
        list(r.router.predict.demos),
    )


def evaluate_router(router: QueryRouter, val_examples: list, num_threads: int = 4) -> float:
    evaluator = dspy.Evaluate(
        devset=val_examples,
        metric=routing_metric,
        num_threads=num_threads,
        display_progress=True,
    )
    result = evaluator(router)
    return float(result) if not isinstance(result, (int, float)) else result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-ablate instruction × demos across MIPROv2 and BFRS artifacts."
    )
    parser.add_argument("--model", required=True, help="Ollama model ID.")
    parser.add_argument("--task-base", default="http://localhost:11434")
    parser.add_argument("--mipro-state", required=True, help="MIPROv2 state path.")
    parser.add_argument("--bfrs-state", required=True, help="BFRS state path.")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

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

    mipro_instr, mipro_demos = load_signature_and_demos(args.mipro_state)
    bfrs_instr, bfrs_demos = load_signature_and_demos(args.bfrs_state)
    print(f"MIPRO state: instruction={len(mipro_instr)} chars, demos={len(mipro_demos)}")
    print(f"BFRS  state: instruction={len(bfrs_instr)} chars, demos={len(bfrs_demos)}")
    # BFRS's instruction is the default signature docstring (unchanged by BFRS).

    cells = [
        # (label, instruction, demos)
        ("mipro_instr_x_bfrs_demos", mipro_instr, bfrs_demos),
        ("default_instr_x_mipro_demos", None, mipro_demos),
    ]

    log_path = Path(__file__).parent / "artifacts" / "cross_ablation.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for label, instr, demos in cells:
        print(f"\n--- cell: {label} ---")
        router = build_router(instr, demos)
        t0 = time.time()
        score = evaluate_router(router, val_examples, args.num_threads)
        elapsed = time.time() - t0
        print(f"score: {score:.4f}   elapsed: {elapsed:.1f}s")
        with log_path.open("a") as f:
            f.write(json.dumps({
                "model": args.model,
                "cell": label,
                "instruction_source": "mipro" if instr == mipro_instr else ("default" if instr is None else "other"),
                "demos_source": "bfrs" if demos == bfrs_demos else ("mipro" if demos == mipro_demos else "none"),
                "val_score": round(score, 4),
                "elapsed_sec": round(elapsed, 1),
            }) + "\n")
    print(f"\nLogged results to {log_path}")


if __name__ == "__main__":
    main()
