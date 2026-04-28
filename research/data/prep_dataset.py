# ABOUTME: Dataset preparation script for the 3-Way LLM Routing Chatbot.
# ABOUTME: Generates synthetic queries via Gemini Flash and maps CLINC150 intents to routing labels.

"""
Dataset Preparation for LLM Router Training

Generates a hybrid dataset of synthetic queries and mapped CLINC150 samples,
labeled with routing categories (local, cloud, search), privacy sensitivity,
and query complexity.

Output schema per sample:
    {
        "query": str,           # The user query text
        "route_label": str,     # One of: "local", "cloud", "search"
        "is_privacy_sensitive": bool,  # Whether query contains PII/health/financial/location data
        "complexity": str,      # "simple" or "complex"
        "source": str           # "synthetic" or "clinc150"
    }
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Any

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Privacy detection keywords by category
PRIVACY_KEYWORDS = {
    "pii": [
        "my name", "my email", "my phone", "my address", "social security",
        "passport", "driver's license", "date of birth", "my age",
        "contact info", "personal information",
    ],
    "health": [
        "my symptoms", "my medication", "my doctor", "my diagnosis",
        "my health", "my prescription", "blood pressure", "my condition",
        "medical record", "my allergy",
    ],
    "financial": [
        "my bank", "my salary", "my account", "my balance", "my credit",
        "my transaction", "my income", "my tax", "my payment", "my debt",
        "my investment", "my portfolio",
    ],
    "location": [
        "my location", "where i am", "my home address", "my current position",
        "track me", "my commute", "where i live", "my neighborhood",
    ],
}

# Complexity indicators for "complex" classification
COMPLEXITY_INDICATORS = [
    "explain", "compare", "analyze", "write", "generate", "create",
    "summarize", "translate", "debug", "optimize", "design", "implement",
    "evaluate", "argue", "discuss", "elaborate", "describe in detail",
    "step by step", "pros and cons", "differences between",
]

SYNTHETIC_PROMPT_TEMPLATE = """You are generating training data for an on-device AI router on a SAMSUNG GALAXY SMARTPHONE.

The router classifies user queries into three routes based on what the Galaxy phone can handle:

## ROUTE DEFINITIONS (Samsung Galaxy context)

**local** — Handled by the on-device LLM or built-in Galaxy apps/services:
- Galaxy built-in apps: Weather, Clock (alarm/timer/stopwatch), Calculator, Calendar, Contacts, Messages, Gallery, Notes, Voice Recorder, Samsung Health, Samsung Pay
- Device settings and controls: volume, brightness, WiFi, Bluetooth, display mode, DND
- Simple Q&A answerable without real-time web data (basic facts, unit conversion, math)
- Short casual chat, greetings, acknowledgments, fragments ("hi", "thanks", "yes")
- Personal/private data: health info, financial data, contacts, location history, medical records
- Simple reminders, todos, voice memos
- On-device translation (short phrases)
- Example: "what is the weather" is LOCAL — Galaxy Weather app handles it via on-device cached data + device sensors
- Example: "my blood pressure is 140/90" is LOCAL — Samsung Health records locally

**cloud** — Needs a powerful LLM (too complex for on-device small LLM):
- Long creative writing, essay generation, story writing, poem composition
- Complex code generation, debugging, architecture design
- Multi-step reasoning, analysis, detailed explanations
- Long document summarization, translation of long texts
- Open-ended brainstorming, philosophical discussion
- Technical deep-dives requiring broad knowledge
- Example: "write a 500-word essay on climate change" — too large for on-device model
- Example: "explain quantum entanglement step by step with examples" — needs strong reasoning

**search** — Needs real-time external web data (not in Galaxy built-in services):
- Current news, events, breaking stories
- Live stock prices, crypto prices, exchange rates
- Sports scores, match results, league standings
- Restaurant reviews, business info, hours of operation
- Flight status, train schedules (external transit)
- Product reviews, comparison shopping, current prices
- Real-time traffic to specific destinations
- Example: "what are Samsung's latest earnings" is SEARCH — requires live data
- Example: "best sushi place near me right now" is SEARCH — external review data

## KEY DISTINCTIONS (resolve common confusions)

- "weather" alone → LOCAL (Galaxy Weather app, on-device)
- "weather forecast in Tokyo next week" → SEARCH (external location + future prediction needs fresh data)
- "my schedule today" → LOCAL (Galaxy Calendar)
- "what's happening in the news today" → SEARCH (current events)
- "calculate 15% tip on $42" → LOCAL (math)
- "write python code to calculate tip" → CLOUD (code generation)
- "translate 'hello' to French" → LOCAL (simple phrase)
- "translate this 3-page document to Korean" → CLOUD (long text)

## DIVERSITY REQUIREMENTS

Generate {count} diverse {route_type} queries. Mix:
- LENGTHS: one-word commands ("mute"), short utterances ("what time is it"), full sentences, multi-sentence queries
- PHRASINGS: commands, questions, polite requests, statements, fragments, casual chat
- STYLES: formal and informal, complete and incomplete, typed and voice-dictated (with filler words)
- AMBIGUITY: include ~15% edge cases that seem like other routes but are actually {route_type}
- PERSONAL DATA: ~10% with sensitive info (names, health, finance, location)
- Include SHORT CLINC-style utterances (3-6 words) mixed with longer queries
- NEVER repeat queries

## OUTPUT FORMAT

Output ONLY a JSON array of strings, no other text.

## EXAMPLE for "local" (notice the range — very short to medium, diverse contexts, Galaxy-aware)

[
  "what is the weather",
  "set alarm for 7am",
  "tell me the temperature outside",
  "mute",
  "what's 15% of 230",
  "my blood pressure reading is 140 over 90 log it in samsung health",
  "thx",
  "how many days until june 12",
  "translate hello to french",
  "turn on dark mode",
  "remind me to call mom at 5pm",
  "what's on my calendar today",
  "show me photos from last weekend",
  "convert 5.7 miles to kilometers",
  "what time is sunset today",
  "my medication refill is due next tuesday"
]

Now generate {count} realistic {route_type} queries for a Samsung Galaxy user.
"""


def detect_privacy_sensitive(query: str) -> bool:
    """Check if a query contains privacy-sensitive content.

    Args:
        query (str): The user query text to check.

    Returns:
        bool: True if the query contains keywords from any privacy category
              (PII, health, financial, location).
    """
    query_lower = query.lower()
    for keywords in PRIVACY_KEYWORDS.values():
        for keyword in keywords:
            if keyword in query_lower:
                return True
    return False


def classify_complexity(query: str) -> str:
    """Classify query complexity as 'simple' or 'complex'.

    A query is 'complex' if it has >20 tokens AND contains multi-step
    reasoning indicators. Otherwise it is 'simple'.

    Args:
        query (str): The user query text to classify.

    Returns:
        str: Either "simple" or "complex".
    """
    tokens = query.split()
    if len(tokens) <= 20:
        return "simple"

    query_lower = query.lower()
    for indicator in COMPLEXITY_INDICATORS:
        if indicator in query_lower:
            return "complex"
    return "simple"


def generate_synthetic_queries(
    route_type: str, count: int, api_key: str, batch_size: int = 250
) -> list[dict[str, Any]]:
    """Generate synthetic queries using Gemini Flash API in batches.

    Uses batched generation (default 250 queries per API call) to avoid
    response truncation on large counts. Higher temperature on later
    batches encourages diversity across the full dataset.

    Args:
        route_type (str): One of "local", "cloud", "search".
        count (int): Total number of queries to generate.
        api_key (str): Google AI API key for Gemini Flash.
        batch_size (int): Queries per API call. Default 250.

    Returns:
        list[dict]: List of sample dicts with keys: query, route_label,
                    is_privacy_sensitive, complexity, source.
    """
    if genai is None:
        raise ImportError(
            "google-generativeai is required. Install with: "
            "pip install google-generativeai"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")

    samples: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    batch_idx = 0

    while len(samples) < count:
        batch_idx += 1
        remaining = count - len(samples)
        this_batch = min(batch_size, remaining + 20)  # Overshoot slightly for dedup

        prompt = SYNTHETIC_PROMPT_TEMPLATE.format(
            route_type=route_type, count=this_batch
        )
        # Vary temperature per batch for diversity
        temp = 0.3 + 0.1 * (batch_idx - 1)

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=min(temp, 0.9),
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.text.strip()
            queries = json.loads(raw_text)
        except Exception as e:
            print(f"    Batch {batch_idx} failed: {e}")
            if batch_idx > 10:
                break
            continue

        if not isinstance(queries, list):
            continue

        added_this_batch = 0
        for query_text in queries:
            if not isinstance(query_text, str) or not query_text.strip():
                continue
            q = query_text.strip()
            if q.lower() in seen_queries:
                continue
            seen_queries.add(q.lower())
            samples.append({
                "query": q,
                "route_label": route_type,
                "is_privacy_sensitive": detect_privacy_sensitive(q),
                "complexity": classify_complexity(q),
                "source": "synthetic",
            })
            added_this_batch += 1
            if len(samples) >= count:
                break

        print(f"    Batch {batch_idx}: +{added_this_batch} unique ({len(samples)}/{count})")

        if added_this_batch == 0 and batch_idx > 3:
            print(f"    No new unique queries, stopping at {len(samples)}")
            break

    return samples[:count]


def load_clinc150_mapped(mapping_path: str) -> list[dict[str, Any]]:
    """Load CLINC150 dataset and map intents to routing labels.

    Uses the HuggingFace datasets library to load CLINC150 and maps each
    intent to local/cloud/search using the mapping file.

    Args:
        mapping_path (str): Path to clinc150_mapping.json.

    Returns:
        list[dict]: List of sample dicts mapped to routing labels.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("WARNING: 'datasets' library not installed. Skipping CLINC150.")
        return []

    with open(mapping_path) as f:
        mapping = json.load(f)

    # Build intent_name -> route_label lookup
    intent_to_route = {}
    for route_label in ["local", "cloud", "search"]:
        for intent_name in mapping.get(route_label, []):
            intent_to_route[intent_name] = route_label

    dataset = load_dataset("clinc_oos", "plus", trust_remote_code=True)
    # CLINC150 uses integer intent IDs; we need the intent names
    intent_names = dataset["train"].features["intent"].names

    samples = []
    for split in ["train", "validation", "test"]:
        for example in dataset[split]:
            intent_id = example["intent"]
            # Skip out-of-scope samples (intent_id for "oos")
            if intent_id >= len(intent_names):
                continue
            intent_name = intent_names[intent_id]
            if intent_name in intent_to_route:
                query = example["text"]
                samples.append({
                    "query": query,
                    "route_label": intent_to_route[intent_name],
                    "is_privacy_sensitive": detect_privacy_sensitive(query),
                    "complexity": classify_complexity(query),
                    "source": "clinc150",
                })

    return samples


def split_dataset(
    samples: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/val/test sets (80/10/10).

    Args:
        samples (list[dict]): All samples to split.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_samples, val_samples, test_samples)
    """
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def save_jsonl(samples: list[dict], filepath: str) -> None:
    """Save samples as JSONL file.

    Args:
        samples (list[dict]): List of sample dictionaries to save.
        filepath (str): Output file path.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {filepath}")


def print_dataset_stats(samples: list[dict], name: str) -> None:
    """Print distribution statistics for a dataset split.

    Args:
        samples (list[dict]): List of sample dictionaries.
        name (str): Name of the split (e.g., "train", "val", "test").
    """
    print(f"\n--- {name} ({len(samples)} samples) ---")

    route_counts = {}
    privacy_count = 0
    complexity_counts = {"simple": 0, "complex": 0}
    source_counts = {}

    for s in samples:
        route_counts[s["route_label"]] = route_counts.get(s["route_label"], 0) + 1
        if s["is_privacy_sensitive"]:
            privacy_count += 1
        complexity_counts[s["complexity"]] = complexity_counts.get(s["complexity"], 0) + 1
        source_counts[s["source"]] = source_counts.get(s["source"], 0) + 1

    print(f"  Routes: {route_counts}")
    print(f"  Privacy-sensitive: {privacy_count} ({100*privacy_count/max(len(samples),1):.1f}%)")
    print(f"  Complexity: {complexity_counts}")
    print(f"  Sources: {source_counts}")


def main():
    """Main dataset preparation pipeline.

    Generates synthetic queries via Gemini Flash API across three routing
    categories (local, cloud, search), splits into train/val/test, and
    saves as JSONL files. Synthetic-only dataset: CLINC150 was found to
    have ambiguous intent-to-route mappings that hurt accuracy.
    """
    output_dir = Path(__file__).parent / "generated"

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set. Synthetic generation requires the API.")
        return

    all_samples = []
    samples_per_class = 2000

    # Synthetic generation via Gemini
    print(f"Generating synthetic queries via Gemini ({samples_per_class} per class)...")
    for route_type in ["local", "cloud", "search"]:
        print(f"  Generating {samples_per_class} '{route_type}' queries...")
        try:
            samples = generate_synthetic_queries(route_type, samples_per_class, api_key)
            all_samples.extend(samples)
            print(f"  Generated {len(samples)} '{route_type}' queries")
        except Exception as e:
            print(f"  ERROR generating '{route_type}': {e}")

    if not all_samples:
        print("\nERROR: No samples generated. Check API key and quota.")
        return

    print(f"\nTotal samples: {len(all_samples)}")

    # Step 3: Split dataset
    train, val, test = split_dataset(all_samples)

    # Step 4: Save splits
    save_jsonl(train, str(output_dir / "train.jsonl"))
    save_jsonl(val, str(output_dir / "val.jsonl"))
    save_jsonl(test, str(output_dir / "test.jsonl"))

    # Step 5: Print statistics
    print_dataset_stats(train, "Train")
    print_dataset_stats(val, "Val")
    print_dataset_stats(test, "Test")

    print(f"\nDataset preparation complete. Files saved to {output_dir}/")


if __name__ == "__main__":
    main()
