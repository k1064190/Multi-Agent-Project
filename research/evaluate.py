# ABOUTME: Evaluation harness for the 3-Way LLM Router.
# ABOUTME: Generates confusion matrix, per-class P/R/F1, and latency distribution reports.

"""
Evaluation Harness for LLM Router

Runs the optimized router against held-out test data and produces:
- Confusion matrix visualization (PNG)
- Per-class precision, recall, F1 scores
- Latency distribution (P50, P95, P99)
- Comprehensive benchmark report (JSON)
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import dspy
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROUTE_LABELS = ["local", "cloud", "search"]


def load_test_data(filepath: str) -> list[dict[str, Any]]:
    """Load test data from JSONL file.

    Args:
        filepath (str): Path to the JSONL test file.

    Returns:
        list[dict]: List of test sample dictionaries.
    """
    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def run_router_evaluation(
    test_samples: list[dict[str, Any]],
    router_state_path: str | None = None,
) -> dict[str, Any]:
    """Run a router on test samples and collect predictions + latencies.

    If `router_state_path` is provided, loads the DSPy-optimized router
    state (post-MIPROv2). If None, uses the default (unoptimized)
    `QueryRouter` — useful as a baseline for measuring DSPy uplift.

    Args:
        test_samples (list[dict]): Test samples with 'query' and 'route_label' keys.
        router_state_path (str | None): Path to the saved DSPy router state JSON.
                                         If None, runs the default router without
                                         optimization (baseline mode).

    Returns:
        dict: Contains 'predictions' list and 'latencies_ms' list.
    """
    # Import and load the router module
    from research.dspy_optimizer import QueryRouter

    router = QueryRouter()
    if router_state_path is not None:
        router.load(router_state_path)
        print("  Mode: optimized (DSPy-compiled router loaded)")
    else:
        print("  Mode: baseline (default router, no DSPy optimization)")

    predictions = []
    latencies_ms = []

    for i, sample in enumerate(test_samples):
        query = sample["query"]

        start = time.perf_counter()
        try:
            result = router(query=query)
            pred_label = result.route_label.strip().lower()
            if pred_label not in ROUTE_LABELS:
                pred_label = "cloud"
        except Exception as e:
            print(f"  WARNING: Error on sample {i}: {e}")
            pred_label = "cloud"
        elapsed_ms = (time.perf_counter() - start) * 1000

        predictions.append(pred_label)
        latencies_ms.append(elapsed_ms)

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i + 1}/{len(test_samples)} samples...")

    return {"predictions": predictions, "latencies_ms": latencies_ms}


def compute_metrics(
    gold_labels: list[str],
    pred_labels: list[str],
    latencies_ms: list[float],
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    Args:
        gold_labels (list[str]): Ground truth route labels.
        pred_labels (list[str]): Predicted route labels.
        latencies_ms (list[float]): Per-query latency in milliseconds.

    Returns:
        dict: Metrics including accuracy, weighted F1, per-class report,
              confusion matrix, and latency percentiles.
    """
    accuracy = accuracy_score(gold_labels, pred_labels)
    weighted_f1 = f1_score(gold_labels, pred_labels, average="weighted", labels=ROUTE_LABELS)
    macro_f1 = f1_score(gold_labels, pred_labels, average="macro", labels=ROUTE_LABELS)

    # Per-class report
    cls_report = classification_report(
        gold_labels, pred_labels,
        labels=ROUTE_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(gold_labels, pred_labels, labels=ROUTE_LABELS)

    # Latency stats
    lat = np.array(latencies_ms)
    latency_stats = {
        "mean_ms": float(np.mean(lat)),
        "median_ms": float(np.median(lat)),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "min_ms": float(np.min(lat)),
        "max_ms": float(np.max(lat)),
    }

    # Privacy leak analysis: count privacy-sensitive queries routed to cloud
    # (requires original samples, done in the main function)

    return {
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            label: {
                "precision": round(cls_report[label]["precision"], 4),
                "recall": round(cls_report[label]["recall"], 4),
                "f1": round(cls_report[label]["f1-score"], 4),
                "support": int(cls_report[label]["support"]),
            }
            for label in ROUTE_LABELS
            if label in cls_report
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ROUTE_LABELS,
        "latency": latency_stats,
        "total_samples": len(gold_labels),
    }


def analyze_privacy_leaks(
    test_samples: list[dict[str, Any]],
    pred_labels: list[str],
) -> dict[str, Any]:
    """Analyze privacy-sensitive queries that were routed to cloud.

    Args:
        test_samples (list[dict]): Test samples with is_privacy_sensitive field.
        pred_labels (list[str]): Predicted route labels.

    Returns:
        dict: Privacy leak statistics.
    """
    total_sensitive = 0
    leaked_to_cloud = 0
    leak_examples = []

    for sample, pred in zip(test_samples, pred_labels):
        is_sensitive = sample.get("is_privacy_sensitive", False)
        if isinstance(is_sensitive, str):
            is_sensitive = is_sensitive.lower() == "true"

        if is_sensitive:
            total_sensitive += 1
            if pred == "cloud":
                leaked_to_cloud += 1
                if len(leak_examples) < 5:
                    leak_examples.append({
                        "query": sample["query"],
                        "gold": sample["route_label"],
                        "predicted": pred,
                    })

    leak_rate = leaked_to_cloud / max(total_sensitive, 1)

    return {
        "total_sensitive_queries": total_sensitive,
        "leaked_to_cloud": leaked_to_cloud,
        "leak_rate": round(leak_rate, 4),
        "leak_examples": leak_examples,
    }


def save_confusion_matrix_plot(
    cm: list[list[int]],
    labels: list[str],
    output_path: str,
) -> None:
    """Save confusion matrix as a PNG heatmap.

    Args:
        cm (list[list[int]]): Confusion matrix values.
        labels (list[str]): Class labels for axes.
        output_path (str): Path to save the PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("WARNING: matplotlib/seaborn not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("LLM Router Confusion Matrix")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


def save_latency_distribution_plot(
    latencies_ms: list[float],
    output_path: str,
) -> None:
    """Save latency distribution as a histogram PNG.

    Args:
        latencies_ms (list[float]): Per-query latencies in milliseconds.
        output_path (str): Path to save the PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed. Skipping latency plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies_ms, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(np.median(latencies_ms), color="red", linestyle="--", label=f"P50: {np.median(latencies_ms):.0f}ms")
    ax.axvline(np.percentile(latencies_ms, 95), color="orange", linestyle="--", label=f"P95: {np.percentile(latencies_ms, 95):.0f}ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Router Inference Latency Distribution")
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Latency distribution saved to {output_path}")


def main():
    """Main evaluation pipeline.

    Loads test data, runs the router (either DSPy-optimized or baseline),
    computes metrics, analyzes privacy leaks, generates plots, and saves
    report. Use --baseline to evaluate the default unoptimized router
    for comparison against the MIPROv2-optimized version.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate LLM routing classifier on held-out test set"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline (default unoptimized router) instead of DSPy-optimized. "
             "Results saved under baseline_* filenames for comparison.",
    )
    parser.add_argument(
        "--model",
        default="gemma4:31b",
        help="Model ID for the task LM. Ollama: gemma4:e2b, qwen3.5:0.8b, etc. "
             "vLLM: the --served-model-name you launched it with. Must match "
             "the model the artifact was optimized for.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama_chat", "openai"],
        default="ollama_chat",
        help="LiteLLM provider prefix for the task LM. 'openai' works for "
             "vLLM / any OpenAI-compatible server.",
    )
    parser.add_argument(
        "--task-base",
        default=None,
        help="Base URL for the task LM. For Ollama, host:port. For vLLM, "
             "include the /v1 suffix. Defaults to $OLLAMA_BASE_URL or "
             "http://localhost:11434.",
    )
    parser.add_argument(
        "--artifact-state",
        default=None,
        help="Path to the optimized DSPy router state JSON. If omitted, uses "
             "research/artifacts/optimized_router_state.json (most recent run).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional identifier appended to result filenames "
             "(e.g., 'gemma4_e2b_bfrs'). Prevents overwriting previous reports.",
    )
    args = parser.parse_args()

    # Configure DSPy LM for the target task model.
    # Default base URL depends on provider.
    if args.task_base:
        task_base = args.task_base
    elif args.provider == "openai":
        task_base = "http://localhost:8000/v1"
    else:
        task_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    extra: dict = {}
    if args.provider == "openai":
        extra["api_key"] = "EMPTY"
    elif args.model.lower().startswith("qwen"):
        extra["think"] = False
    lm = dspy.LM(
        model=f"{args.provider}/{args.model}",
        api_base=task_base,
        temperature=0.0,
        **extra,
    )
    dspy.configure(lm=lm)
    print(f"Task LM: {args.model} via {task_base} (provider={args.provider})")

    # Paths
    base_dir = Path(__file__).parent
    test_path = str(base_dir / "data" / "generated" / "test.jsonl")
    router_state_path = args.artifact_state or str(
        base_dir / "artifacts" / "optimized_router_state.json"
    )
    results_dir = base_dir / "results"

    # Determine filenames based on mode + optional tag
    tag = args.tag or Path(router_state_path).stem.replace("optimized_router_state_", "")
    tag_suffix = f"_{tag}" if tag and tag != "optimized_router_state" else ""
    if args.baseline:
        mode_label = f"BASELINE (no DSPy optimization) — {args.model}"
        report_filename = f"baseline_report{tag_suffix}.json"
        confusion_filename = f"baseline_confusion_matrix{tag_suffix}.png"
        latency_filename = f"baseline_latency_distribution{tag_suffix}.png"
        state_for_eval = None  # Default unoptimized router
    else:
        mode_label = f"OPTIMIZED — {args.model} / state={Path(router_state_path).name}"
        report_filename = f"benchmark_report{tag_suffix}.json"
        confusion_filename = f"confusion_matrix{tag_suffix}.png"
        latency_filename = f"latency_distribution{tag_suffix}.png"
        state_for_eval = router_state_path

    print("=" * 60)
    print(f"Evaluation mode: {mode_label}")
    print("=" * 60)

    # Validate paths
    if not Path(test_path).exists():
        print(f"ERROR: Test data not found at {test_path}")
        print("Run prep_dataset.py first.")
        return

    if not args.baseline and not Path(router_state_path).exists():
        print(f"ERROR: Router state not found at {router_state_path}")
        print("Run dspy_optimizer.py first, or use --baseline to skip.")
        return

    # Load test data
    print("Loading test data...")
    test_samples = load_test_data(test_path)
    gold_labels = [s["route_label"] for s in test_samples]
    print(f"  {len(test_samples)} test samples loaded")

    # Run evaluation
    print("\nRunning router on test set...")
    eval_results = run_router_evaluation(test_samples, state_for_eval)
    pred_labels = eval_results["predictions"]
    latencies_ms = eval_results["latencies_ms"]

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(gold_labels, pred_labels, latencies_ms)

    # Privacy leak analysis
    print("\nAnalyzing privacy leaks...")
    privacy = analyze_privacy_leaks(test_samples, pred_labels)
    metrics["privacy_analysis"] = privacy

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"\nPer-class metrics:")
    for label in ROUTE_LABELS:
        if label in metrics["per_class"]:
            cls = metrics["per_class"][label]
            print(f"  {label:8s} — P: {cls['precision']:.3f}  R: {cls['recall']:.3f}  F1: {cls['f1']:.3f}  (n={cls['support']})")
    print(f"\nLatency:")
    lat = metrics["latency"]
    print(f"  P50: {lat['p50_ms']:.0f}ms  P95: {lat['p95_ms']:.0f}ms  P99: {lat['p99_ms']:.0f}ms")
    print(f"\nPrivacy:")
    print(f"  Sensitive queries: {privacy['total_sensitive_queries']}")
    print(f"  Leaked to cloud:  {privacy['leaked_to_cloud']} ({privacy['leak_rate']:.1%})")

    # Save report
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = str(results_dir / report_filename)
    metrics["mode"] = "baseline" if args.baseline else "optimized"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Save plots
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        ROUTE_LABELS,
        str(results_dir / confusion_filename),
    )
    save_latency_distribution_plot(
        latencies_ms,
        str(results_dir / latency_filename),
    )

    # Check success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)
    f1_pass = metrics["weighted_f1"] >= 0.85
    leak_pass = privacy["leak_rate"] < 0.1
    print(f"  Weighted F1 >= 0.85: {'PASS' if f1_pass else 'FAIL'} ({metrics['weighted_f1']:.4f})")
    print(f"  Privacy leak < 10%:  {'PASS' if leak_pass else 'FAIL'} ({privacy['leak_rate']:.1%})")

    # If both reports exist, generate comparison markdown
    baseline_report = results_dir / "baseline_report.json"
    optimized_report = results_dir / "benchmark_report.json"
    if baseline_report.exists() and optimized_report.exists():
        _write_comparison_report(baseline_report, optimized_report, results_dir)


def _write_comparison_report(
    baseline_path: Path,
    optimized_path: Path,
    output_dir: Path,
) -> None:
    """Write a Markdown report comparing baseline vs optimized router results.

    Args:
        baseline_path (Path): Path to baseline_report.json.
        optimized_path (Path): Path to benchmark_report.json (optimized).
        output_dir (Path): Directory to write the comparison.md file.
    """
    with open(baseline_path) as f:
        base = json.load(f)
    with open(optimized_path) as f:
        opt = json.load(f)

    def delta(opt_val: float, base_val: float) -> str:
        diff = opt_val - base_val
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.4f}"

    lines = []
    lines.append("# Baseline vs DSPy-Optimized Router Comparison")
    lines.append("")
    lines.append("Held-out test set performance: default router (no prompt optimization)")
    lines.append("vs DSPy MIPROv2-optimized router. Both use Gemma 4 31B as the inference LM.")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Optimized | Δ |")
    lines.append("|--------|----------|-----------|---|")
    lines.append(
        f"| Accuracy    | {base['accuracy']:.4f} | {opt['accuracy']:.4f} | {delta(opt['accuracy'], base['accuracy'])} |"
    )
    lines.append(
        f"| Weighted F1 | {base['weighted_f1']:.4f} | {opt['weighted_f1']:.4f} | {delta(opt['weighted_f1'], base['weighted_f1'])} |"
    )
    lines.append(
        f"| Macro F1    | {base['macro_f1']:.4f} | {opt['macro_f1']:.4f} | {delta(opt['macro_f1'], base['macro_f1'])} |"
    )
    lines.append("")
    lines.append("## Per-class F1")
    lines.append("")
    lines.append("| Class | Baseline F1 | Optimized F1 | Δ | Support |")
    lines.append("|-------|-------------|--------------|---|---------|")
    for cls in ROUTE_LABELS:
        b = base["per_class"].get(cls, {})
        o = opt["per_class"].get(cls, {})
        b_f1 = b.get("f1", 0.0)
        o_f1 = o.get("f1", 0.0)
        lines.append(
            f"| {cls} | {b_f1:.4f} | {o_f1:.4f} | {delta(o_f1, b_f1)} | {o.get('support', '-')} |"
        )
    lines.append("")
    lines.append("## Privacy Leak Rate")
    lines.append("")
    b_priv = base.get("privacy_analysis", {})
    o_priv = opt.get("privacy_analysis", {})
    lines.append("| | Baseline | Optimized |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Sensitive queries | {b_priv.get('total_sensitive_queries', 0)} | {o_priv.get('total_sensitive_queries', 0)} |"
    )
    lines.append(
        f"| Leaked to cloud   | {b_priv.get('leaked_to_cloud', 0)} | {o_priv.get('leaked_to_cloud', 0)} |"
    )
    lines.append(
        f"| Leak rate         | {b_priv.get('leak_rate', 0):.1%} | {o_priv.get('leak_rate', 0):.1%} |"
    )
    lines.append("")
    lines.append("## Latency (ms)")
    lines.append("")
    b_lat = base.get("latency", {})
    o_lat = opt.get("latency", {})
    lines.append("| Percentile | Baseline | Optimized |")
    lines.append("|---|---|---|")
    for p in ["p50_ms", "p95_ms", "p99_ms"]:
        lines.append(f"| {p.upper()} | {b_lat.get(p, 0):.0f} | {o_lat.get(p, 0):.0f} |")
    lines.append("")
    lines.append("Note: Latency similarity between modes is expected — both run Gemma 4 31B.")
    lines.append("The routing quality improvement comes from the DSPy-optimized prompt,")
    lines.append("not from changing the inference LM.")
    lines.append("")

    output_path = output_dir / "comparison.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nComparison report saved to {output_path}")


if __name__ == "__main__":
    main()
