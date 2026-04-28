# ABOUTME: DSPy prompt optimization pipeline for the 3-Way LLM Router.
# ABOUTME: Defines routing signature, custom privacy-aware metric, and runs MIPROv2 optimization.

"""
DSPy Optimizer for LLM Router

Defines a ChainOfThought routing module that classifies queries into
local/cloud/search, with a custom metric that penalizes privacy leaks
and unnecessary cloud routing. Optimizes via MIPROv2 or
BootstrapFewShotWithRandomSearch.

Output: optimized_prompt.json artifact for downstream LangChain integration.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import random as _random

import dspy
import litellm

# COPRO's proposer calls pass `n=<breadth>` to request multiple completions
# per call. The ollama_chat LiteLLM provider does not implement `n` and
# the upstream Ollama HTTP API itself only returns a single `message` per
# request, so even dropping `n` silently (`litellm.drop_params=True`)
# collapses breadth down to 1. `drop_params` is still useful for other
# unsupported params, but for COPRO breadth we use `OllamaBreadthLM`
# below, which loops n times sequentially with jittered temperature/seed.
litellm.drop_params = True


class OllamaBreadthLM(dspy.LM):
    """dspy.LM subclass that implements `n>1` against Ollama by looping.

    Ollama's /api/chat returns one completion per HTTP request, so
    asking for n>1 either errors (UnsupportedParamsError) or gets
    silently dropped by LiteLLM — both collapse to breadth=1. For
    COPRO, where a small breadth is the whole point of the optimizer,
    we intercept `n` and issue n sequential calls with jittered
    temperature and distinct random seeds so the results are actually
    diverse, then concatenate the completions into a single list.

    Used only for the prompt proposer when running COPRO on Ollama.
    MIPROv2 and BFRS do not need it.
    """

    def __call__(self, prompt=None, messages=None, **kwargs):
        n = int(kwargs.pop("n", 1) or 1)
        if n <= 1:
            return super().__call__(prompt=prompt, messages=messages, **kwargs)

        base_temp = float(kwargs.pop("temperature", 0.7) or 0.7)
        rng = _random.Random(kwargs.pop("seed", 42))
        results = []
        for i in range(n):
            # Small linear jitter keeps samples distinct without letting
            # temperature wander far from the caller's intent.
            jittered_temp = max(0.3, min(1.4, base_temp + (i - n / 2) * 0.1))
            call_kwargs = dict(kwargs)
            call_kwargs["temperature"] = jittered_temp
            call_kwargs["seed"] = rng.randint(0, 2**31 - 1)
            completions = super().__call__(
                prompt=prompt, messages=messages, **call_kwargs
            )
            if isinstance(completions, list):
                results.extend(completions)
            else:
                results.append(completions)
        return results


# --- DSPy Signatures and Modules ---


class RouteQuery(dspy.Signature):
    """Classify a user query into the best routing destination.

    Given a user query, determine whether it should be handled by:
    - local: Simple queries answerable by a small on-device model
    - cloud: Complex queries requiring a powerful LLM
    - search: Queries needing real-time information from the web

    Consider privacy: queries with personal/sensitive data should prefer
    local routing to avoid sending private information to cloud services.
    """

    query: str = dspy.InputField(desc="The user's natural language query")
    reasoning: str = dspy.OutputField(
        desc="Brief reasoning for the routing decision (1-2 sentences)"
    )
    route_label: str = dspy.OutputField(
        desc="Routing destination: must be exactly one of 'local', 'cloud', or 'search'"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0"
    )


class QueryRouter(dspy.Module):
    """DSPy module that routes queries using ChainOfThought reasoning.

    Uses ChainOfThought to produce step-by-step reasoning before
    outputting the final routing decision.
    """

    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(RouteQuery)

    def forward(self, query: str) -> dspy.Prediction:
        """Route a single query.

        Args:
            query (str): The user query to classify.

        Returns:
            dspy.Prediction: Prediction with route_label, reasoning, confidence.
        """
        result = self.router(query=query)
        # Normalize route_label to lowercase and validate
        label = result.route_label.strip().lower()
        if label not in ("local", "cloud", "search"):
            # Attempt to extract valid label from the text
            for valid in ("local", "cloud", "search"):
                if valid in label:
                    label = valid
                    break
            else:
                label = "cloud"  # Default fallback
        result.route_label = label
        return result


# --- Privacy Detection (mirrors prep_dataset.py) ---

PRIVACY_KEYWORDS = {
    "pii": [
        "my name", "my email", "my phone", "my address", "social security",
        "passport", "driver's license", "date of birth",
    ],
    "health": [
        "my symptoms", "my medication", "my doctor", "my diagnosis",
        "my health", "my prescription",
    ],
    "financial": [
        "my bank", "my salary", "my account", "my balance", "my credit",
        "my transaction", "my income",
    ],
    "location": [
        "my location", "where i am", "my home address", "track me",
    ],
}


def is_privacy_sensitive(query: str) -> bool:
    """Check if a query contains privacy-sensitive content.

    Args:
        query (str): The user query to check.

    Returns:
        bool: True if query contains privacy-sensitive keywords.
    """
    query_lower = query.lower()
    for keywords in PRIVACY_KEYWORDS.values():
        for kw in keywords:
            if kw in query_lower:
                return True
    return False


# --- Custom Metric ---


def routing_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Per-example routing metric with privacy and cost penalties.

    Scoring:
        +1.0 for correct route label
        -0.5 penalty if privacy-sensitive query is routed to cloud
        -0.3 penalty if simple query is routed to cloud (cost waste)

    Args:
        example (dspy.Example): Ground truth example with route_label,
                                is_privacy_sensitive, complexity fields.
        prediction (dspy.Prediction): Model prediction with route_label.
        trace: DSPy trace object (unused, required by DSPy API).

    Returns:
        float: Score in range [0.0, 1.0]. Higher is better.
    """
    pred_label = prediction.route_label.strip().lower()
    gold_label = example.route_label.strip().lower()

    # Base score: correctness
    score = 1.0 if pred_label == gold_label else 0.0

    # Privacy penalty: sending sensitive data to cloud
    privacy_flag = getattr(example, "is_privacy_sensitive", False)
    if isinstance(privacy_flag, str):
        privacy_flag = privacy_flag.lower() == "true"
    if privacy_flag and pred_label == "cloud":
        score -= 0.5

    # Cost penalty: routing simple queries to expensive cloud
    complexity = getattr(example, "complexity", "simple")
    if complexity == "simple" and pred_label == "cloud":
        score -= 0.3

    return max(0.0, score)


# --- Data Loading ---


def load_examples(filepath: str) -> list[dspy.Example]:
    """Load JSONL dataset into DSPy Examples.

    Args:
        filepath (str): Path to a JSONL file with routing samples.

    Returns:
        list[dspy.Example]: List of DSPy examples with input/label fields set.
    """
    examples = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            ex = dspy.Example(
                query=data["query"],
                route_label=data["route_label"],
                is_privacy_sensitive=data.get("is_privacy_sensitive", False),
                complexity=data.get("complexity", "simple"),
            ).with_inputs("query")
            examples.append(ex)
    return examples


# --- Optimization ---


def run_optimization(
    train_path: str,
    val_path: str,
    output_path: str,
    optimizer_type: str = "mipro",
    max_demos: int = 8,
    num_candidates: int = 10,
) -> dict[str, Any]:
    """Run DSPy prompt optimization on the routing task.

    Args:
        train_path (str): Path to train.jsonl.
        val_path (str): Path to val.jsonl (used as dev set by MIPROv2).
        output_path (str): Path to save the optimized prompt JSON artifact.
        optimizer_type (str): Which DSPy optimizer to run.
            "mipro" — MIPROv2, joint instruction + few-shot search.
            "bfrs"  — BootstrapFewShotWithRandomSearch, few-shot only.
            "copro" — COPRO, coordinate-ascent instruction search only.
        max_demos (int): Maximum number of few-shot demonstrations.
        num_candidates (int): Number of prompt candidates to evaluate.

    Returns:
        dict: Optimization results including best metric score and config.
    """
    print("Loading training data...")
    train_examples = load_examples(train_path)
    val_examples = load_examples(val_path)
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val: {len(val_examples)} examples")

    # Initialize the router module
    router = QueryRouter()

    # Select optimizer
    if optimizer_type == "mipro":
        # Honor global auto mode if set, otherwise default to light
        auto_mode = globals().get("_CURRENT_AUTO_MODE", "light")
        print(f"\nRunning MIPROv2 optimization (auto={auto_mode})...")
        # Subsample training set to limit local inference time.
        # Stratified subsample to keep class balance. For seed set (150 samples)
        # no subsampling happens since max_train > 150.
        max_train = 500
        if len(train_examples) > max_train:
            from collections import defaultdict
            by_class = defaultdict(list)
            for ex in train_examples:
                by_class[ex.route_label].append(ex)
            per_class = max_train // len(by_class)
            subsampled = []
            import random as _rand
            _rand.seed(42)
            for label, exs in by_class.items():
                _rand.shuffle(exs)
                subsampled.extend(exs[:per_class])
            _rand.shuffle(subsampled)
            print(f"  Subsampled trainset: {len(train_examples)} -> {len(subsampled)} (stratified)")
            train_examples = subsampled

        # If a separate prompt proposer model was configured, use it.
        # Otherwise MIPROv2 defaults to the task LM for both roles.
        prompt_lm = globals().get("_CURRENT_PROMPT_LM", None)
        mipro_kwargs = {
            "metric": routing_metric,
            "auto": auto_mode,
            "num_threads": 4,
        }
        if prompt_lm is not None:
            mipro_kwargs["prompt_model"] = prompt_lm
            print(f"  Prompt proposer: separate LM configured")
        optimizer = dspy.MIPROv2(**mipro_kwargs)
        optimized_router = optimizer.compile(
            router,
            trainset=train_examples,
            valset=val_examples,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=max_demos,
        )
    elif optimizer_type == "bfrs":
        print("\nRunning BootstrapFewShotWithRandomSearch optimization...")
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=routing_metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=max_demos,
            num_candidate_programs=num_candidates,
            num_threads=4,
        )
        optimized_router = optimizer.compile(
            router,
            trainset=train_examples,
            valset=val_examples,
        )
    elif optimizer_type == "copro":
        # Coordinate-ascent instruction search. No few-shot demos, only
        # instruction-field rewrites proposed by the prompt model. Useful
        # as an ablation against MIPROv2 — isolates the instruction
        # contribution from the demo contribution.
        breadth = 6
        depth = 2
        print(
            f"\nRunning COPRO optimization (breadth={breadth}, depth={depth})..."
        )
        prompt_lm = globals().get("_CURRENT_PROMPT_LM", None)
        copro_kwargs = {
            "metric": routing_metric,
            "breadth": breadth,
            "depth": depth,
        }
        if prompt_lm is not None:
            copro_kwargs["prompt_model"] = prompt_lm
            print(f"  Prompt proposer: separate LM configured")
        optimizer = dspy.COPRO(**copro_kwargs)
        # COPRO uses the trainset internally for eval during coordinate
        # ascent, and does not take a separate valset at compile time.
        optimized_router = optimizer.compile(
            router,
            trainset=train_examples,
            eval_kwargs={"num_threads": 4, "display_progress": True},
        )
    else:
        raise ValueError(
            f"Unknown optimizer_type: {optimizer_type!r}. "
            f"Expected one of: mipro, bfrs, copro"
        )

    # Evaluate on validation set. Under COPRO, weak SLMs have produced
    # very long outputs that tripped Ollama's 10-minute request deadline;
    # a 4-thread pool then deadlocked cleaning up the 500s. Drop to a
    # single worker for COPRO specifically so any one hang stalls only
    # itself and is recoverable.
    eval_threads = 1 if optimizer_type == "copro" else 4
    print(
        f"\nEvaluating optimized router on validation set "
        f"(num_threads={eval_threads})..."
    )
    evaluator = dspy.Evaluate(
        devset=val_examples,
        metric=routing_metric,
        num_threads=eval_threads,
        display_progress=True,
    )
    val_result = evaluator(optimized_router)
    # DSPy 3.x returns EvaluationResult; extract the numeric score
    val_score = float(val_result) if not isinstance(val_result, (int, float)) else val_result
    print(f"Validation score: {val_score:.4f}")

    # Extract and save the optimized prompt artifact
    artifact = _extract_prompt_artifact(optimized_router, val_score, optimizer_type)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"\nOptimized prompt saved to {output_path}")

    # Also save the full DSPy program state. The canonical file is
    # `optimized_router_state.json` (last-run pointer). We additionally
    # write a stem-suffixed copy that mirrors the artifact filename so
    # each (model, optimizer) pair has its own preserved state.
    out_dir = Path(output_path).parent
    canonical_state_path = str(out_dir / "optimized_router_state.json")
    stem = Path(output_path).stem.replace("optimized_prompt_", "")
    suffixed_state_path = str(out_dir / f"optimized_router_state_{stem}.json")
    optimized_router.save(canonical_state_path)
    optimized_router.save(suffixed_state_path)
    print(f"DSPy program state saved to {canonical_state_path}")
    print(f"DSPy program state also preserved at {suffixed_state_path}")

    return {"val_score": val_score, "artifact_path": output_path}


def _extract_prompt_artifact(
    optimized_router: QueryRouter,
    val_score: float,
    optimizer_type: str,
) -> dict[str, Any]:
    """Extract a portable prompt artifact from an optimized DSPy module.

    In DSPy 3.x, ChainOfThought wraps an internal Predict whose state
    (demos and optimized instructions) lives at `.router.predict`.
    We walk the module tree to find the Predict with demos/instructions.

    Args:
        optimized_router (QueryRouter): The DSPy-optimized router module.
        val_score (float): Validation metric score.
        optimizer_type (str): Name of the optimizer used.

    Returns:
        dict: Prompt artifact with system_prompt, few_shot_examples, etc.
    """
    few_shot_examples: list[dict[str, Any]] = []
    system_prompt: str = ""

    # Walk candidate paths where DSPy may store optimized state.
    # ChainOfThought wraps a Predict at `.predict` in DSPy 3.x.
    candidates = []
    router_obj = optimized_router.router
    candidates.append(router_obj)
    if hasattr(router_obj, "predict"):
        candidates.append(router_obj.predict)

    for predictor in candidates:
        # Extract few-shot demos
        demos = getattr(predictor, "demos", None)
        if demos and not few_shot_examples:
            for demo in demos:
                # DSPy demos are Example objects; access via attribute or dict
                get = (
                    demo.get if hasattr(demo, "get") else lambda k, d="": getattr(demo, k, d)
                )
                few_shot_examples.append({
                    "query": get("query", ""),
                    "reasoning": get("reasoning", ""),
                    "route_label": get("route_label", ""),
                    "confidence": float(get("confidence", 0.9) or 0.9),
                })

        # Extract optimized instructions from the signature
        sig = getattr(predictor, "signature", None)
        if sig is not None:
            instr = getattr(sig, "instructions", None)
            if instr and not system_prompt:
                system_prompt = instr
        # DSPy 2.x had extended_signature
        if hasattr(predictor, "extended_signature"):
            ext_sig = predictor.extended_signature
            ext_instr = getattr(ext_sig, "instructions", None)
            if ext_instr and not system_prompt:
                system_prompt = ext_instr

    # Fallback to a sensible default if nothing was extracted
    if not system_prompt:
        system_prompt = (
            "You are a query routing classifier. Given a user query, classify it into "
            "one of three categories:\n"
            "- local: Simple queries answerable by a small on-device model\n"
            "- cloud: Complex queries requiring a powerful LLM\n"
            "- search: Queries needing real-time information from the web\n\n"
            "IMPORTANT: If the query contains personal or sensitive information, "
            "prefer routing to 'local' to protect privacy."
        )

    return {
        "system_prompt": system_prompt,
        "few_shot_examples": few_shot_examples,
        "output_format": "JSON with keys: route_label, confidence, reasoning",
        "dspy_version": dspy.__version__ if hasattr(dspy, "__version__") else "3.x",
        "optimizer": optimizer_type,
        "metrics": {
            "val_score": round(val_score, 4),
        },
    }


def main():
    """Main optimization entry point.

    CLI flags:
        --model: Ollama model ID (e.g. gemma4:31b, gemma4:e2b, qwen3.5:0.8b).
                 Default: gemma4:31b.
        --seed-set: Use the 150-query seed set for training (stratified).
                    Default: use train.jsonl.
        --auto: MIPROv2 auto mode (light/medium/heavy). Default: light.

    Artifacts are suffixed with the model name so multiple SLMs can
    coexist (e.g. optimized_prompt_gemma4_e2b.json).
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimize router prompt with DSPy MIPROv2 for a target LLM"
    )
    parser.add_argument(
        "--model",
        default="gemma4:31b",
        help="Ollama model ID. Examples: gemma4:31b (reference), gemma4:e2b, qwen3.5:0.8b",
    )
    parser.add_argument(
        "--seed-set",
        action="store_true",
        help="Use 150-query seed set (seed_set.jsonl) instead of train.jsonl. "
             "Recommended for SLM optimization — small + diverse.",
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium"],
        default="light",
        help="MIPROv2 auto mode (affects number of trials).",
    )
    parser.add_argument(
        "--task-base",
        default="http://localhost:11434",
        help="Ollama base URL for the task LM (student). Default: 11434.",
    )
    parser.add_argument(
        "--prompt-model",
        default=None,
        help="Separate prompt-proposer model (Ollama ID). If unset, uses the "
             "task model for both roles (standard MIPROv2 default).",
    )
    parser.add_argument(
        "--prompt-base",
        default="http://localhost:11434",
        help="Base URL for the prompt proposer. For Ollama, a host:port. "
             "For vLLM / any OpenAI-compatible server, include the /v1 suffix "
             "(e.g., http://127.0.0.1:8000/v1).",
    )
    parser.add_argument(
        "--prompt-provider",
        choices=["ollama_chat", "openai"],
        default="ollama_chat",
        help="LiteLLM provider prefix for the prompt proposer. "
             "'ollama_chat' for Ollama (no native `n` support — breadth gets "
             "looped sequentially via OllamaBreadthLM for COPRO). "
             "'openai' for vLLM / any OpenAI-compatible server (native `n`, "
             "proper breadth search).",
    )
    parser.add_argument(
        "--optimizer",
        choices=["mipro", "bfrs", "copro"],
        default="mipro",
        help="DSPy optimizer. mipro=MIPROv2 (instruction + demos, Bayesian). "
             "bfrs=BootstrapFewShotWithRandomSearch (demos only). "
             "copro=COPRO (instructions only, coordinate ascent).",
    )
    args = parser.parse_args()

    # Qwen 3.5 and other thinking-capable models route output to a `thinking`
    # field by default, leaving `content` empty. DSPy parses content, so we
    # disable thinking for any model whose name starts with "qwen".
    def _ollama_extra(model_name: str) -> dict:
        return {"think": False} if model_name.lower().startswith("qwen") else {}

    # When running COPRO on Ollama, cap the task LM's generation length
    # so a verbose proposed instruction cannot push a weak SLM into
    # runaway output that trips Ollama's 10-minute request timeout.
    task_extra: dict = dict(_ollama_extra(args.model))
    if args.optimizer == "copro":
        task_extra.setdefault("max_tokens", 512)

    # Configure Task LM (the student, what gets optimized)
    task_lm = dspy.LM(
        model=f"ollama_chat/{args.model}",
        api_base=args.task_base,
        temperature=0.0,
        **task_extra,
    )
    dspy.configure(lm=task_lm)
    print(f"Task LM:   {args.model} via {args.task_base}")
    if args.optimizer == "copro":
        print(f"           COPRO safety cap: max_tokens={task_extra['max_tokens']}")

    # Optional: Configure Prompt Model (separate, stronger model for proposal)
    if args.prompt_model:
        # Pick the right LM class based on provider + optimizer:
        # - vLLM/OpenAI-compat: `dspy.LM` with `openai/...` model prefix.
        #   Native `n` support, no wrapper needed.
        # - Ollama + COPRO:     `OllamaBreadthLM` (see class docstring).
        # - Ollama + other:     plain `dspy.LM`.
        if args.prompt_provider == "openai":
            prompt_lm_cls = dspy.LM
            prompt_model_str = f"openai/{args.prompt_model}"
            prompt_extra: dict = {"api_key": "EMPTY"}
        else:
            prompt_lm_cls = (
                OllamaBreadthLM if args.optimizer == "copro" else dspy.LM
            )
            prompt_model_str = f"ollama_chat/{args.prompt_model}"
            prompt_extra = dict(_ollama_extra(args.prompt_model))

        prompt_lm = prompt_lm_cls(
            model=prompt_model_str,
            api_base=args.prompt_base,
            temperature=0.7,  # Higher temp for diverse instruction candidates
            **prompt_extra,
        )
        globals()["_CURRENT_PROMPT_LM"] = prompt_lm
        if args.prompt_provider == "openai":
            kind = "vLLM/OpenAI-compat proposer (native n)"
        elif prompt_lm_cls is OllamaBreadthLM:
            kind = "Ollama breadth-aware proposer"
        else:
            kind = "Ollama separate proposer"
        print(f"Prompt LM: {args.prompt_model} via {args.prompt_base} ({kind})")
    else:
        print(f"Prompt LM: same as task LM (default)")

    # Paths — trainset depends on --seed-set flag
    data_dir = Path(__file__).parent / "data" / "generated"
    if args.seed_set:
        train_path = str(data_dir / "seed_set.jsonl")
        val_path = str(data_dir / "val.jsonl")  # Keep val from original split
        print(f"Training on seed set: {train_path}")
    else:
        train_path = str(data_dir / "train.jsonl")
        val_path = str(data_dir / "val.jsonl")
        print(f"Training on full train set: {train_path}")

    # Output path includes model name suffix for multi-model comparison.
    # Non-mipro optimizers also embed the optimizer in the filename so
    # artifacts from different strategies can coexist for the same model.
    safe_model_name = args.model.replace(":", "_").replace("/", "_").replace(".", "")
    if args.optimizer == "mipro":
        artifact_stem = f"optimized_prompt_{safe_model_name}"
    else:
        artifact_stem = f"optimized_prompt_{safe_model_name}_{args.optimizer}"
    output_path = str(
        Path(__file__).parent / "artifacts" / f"{artifact_stem}.json"
    )

    # Check if data exists
    if not Path(train_path).exists():
        print(f"ERROR: Training data not found at {train_path}")
        if args.seed_set:
            print("Run research/data/prep_seed_set.py first.")
        else:
            print("Run research/data/prep_dataset.py first.")
        return

    # Override auto mode by re-patching run_optimization's internal call
    global _CURRENT_AUTO_MODE
    _CURRENT_AUTO_MODE = args.auto

    # Run optimization
    start_time = time.time()
    results = run_optimization(
        train_path=train_path,
        val_path=val_path,
        output_path=output_path,
        optimizer_type=args.optimizer,
        max_demos=8,
        num_candidates=10,
    )
    elapsed = time.time() - start_time

    print(f"\nOptimization completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Validation score: {results['val_score']:.4f}")
    print(f"Artifact saved to: {results['artifact_path']}")


if __name__ == "__main__":
    main()
