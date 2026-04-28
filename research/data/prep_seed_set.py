# ABOUTME: Seed set generator for the 3-Way LLM Router — small, diverse, high-quality.
# ABOUTME: Generates 150 hand-verified-ready queries (50/class) across 4 difficulty levels.

"""
Seed Set Generator

Generates a small but deliberately diverse set of routing queries intended
to be hand-verified and used as the DSPy optimization training set. The
design hypothesis: a high-quality small seed (with hard edge cases) lets
DSPy learn a truly robust prompt, which then drives bulk data generation
for distillation.

Per class (50 queries total):
    - easy     (8, 16%):  obvious, unambiguous baseline cases
    - medium   (20, 40%): realistic daily queries with mild ambiguity
    - hard     (15, 30%): known failure patterns (A/B/C/D from error analysis)
    - extreme  (7, 14%):  adversarial, multi-intent, long-form, tricky cases

Total: 150 queries (3 classes x 50). Output preserves `difficulty` and
`edge_case_type` fields so the evaluation harness can break down performance
by difficulty.
"""

import json
import os
import random
from pathlib import Path
from typing import Any

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# --- Privacy detection (reused across scripts) ---
PRIVACY_KEYWORDS = {
    "pii": ["my name", "my email", "my phone", "my address", "social security",
            "passport", "driver's license", "date of birth"],
    "health": ["my symptoms", "my medication", "my doctor", "my diagnosis",
               "my health", "my prescription", "blood pressure"],
    "financial": ["my bank", "my salary", "my account", "my balance",
                  "my credit", "my income", "my debt", "my loan", "401k"],
    "location": ["my location", "my home address", "my neighborhood",
                 "track me", "where i live"],
}


def detect_privacy_sensitive(query: str) -> bool:
    """Return True if query contains privacy-sensitive keywords.

    Args:
        query (str): Query text to check.

    Returns:
        bool: True if matched any privacy keyword.
    """
    q = query.lower()
    for kws in PRIVACY_KEYWORDS.values():
        for kw in kws:
            if kw in q:
                return True
    return False


# --- Seed generation prompts per (difficulty, class) ---

BASE_CONTEXT = """You are generating training queries for a Samsung Galaxy phone AI router.
The router classifies queries into three routes:
- local: Handled by Galaxy built-in apps (Weather, Clock, Calendar, Samsung Health,
  Contacts, Gallery, etc.) OR by a small on-device LLM. Includes simple chat,
  greetings, short Q&A, and privacy-sensitive data (health, finance, PII, location).
- cloud: Requires a powerful LLM. Long creative writing, code generation,
  detailed multi-step reasoning, large document summarization or translation.
- search: Requires real-time external web data not in Galaxy built-in services.
  Current news, live stock prices, restaurant reviews, flight status, product prices.

Key distinctions (canonical rulings):
- "what is the weather" → local (Galaxy Weather app handles this on-device)
- "what time is it / what date is it" → local (Galaxy Clock, no web needed)
- "weather forecast in Tokyo next week" → search (future + non-local)
- "translate 'hello' to French" → local (simple phrase)
- "translate this 3-page document" → cloud (long generation)
"""

DIFFICULTY_SPECS = {
    "easy": {
        "local": "Very obvious local queries. Short, direct, unambiguous use of Galaxy built-in features. No personal/sensitive content required. 1-10 words. Examples: 'mute', 'set timer for 5 minutes', 'what's 20% of 50'.",
        "cloud": "Very obvious cloud queries. Clearly require long generation or deep reasoning. No ambiguity. Explicitly ask for long output or complex analysis. Examples: 'Write a 1000-word essay on industrial revolution', 'Debug this 50-line Python function'.",
        "search": "Very obvious search queries. Explicitly ask for current, live, or external web data that Galaxy cannot provide locally. Examples: 'Show me today's stock price of Samsung Electronics', 'What are today's CNN headlines'.",
    },
    "medium": {
        "local": "Realistic local queries. Mix of short commands, personal data tasks, simple chat, Galaxy app tasks. Include some with casual phrasing, fragments, voice-dictated filler. ~5% should involve sensitive data (health/financial/PII).",
        "cloud": "Realistic cloud queries. Mix of creative writing requests, coding help, translation of medium texts, analytical explanations, brainstorming. Length varies. Make clear they need a large LLM.",
        "search": "Realistic search queries about current info: weather forecasts for other cities, live scores, recent news events, business hours, product prices, restaurant reviews.",
    },
    "hard": {
        "local": (
            "HARD edge cases for LOCAL. These should LOOK like search or cloud but actually be local. "
            "Focus: "
            "(1) Time/date queries with words 'now', 'current', 'right now' — Galaxy Clock handles these. "
            "Examples: 'what time is it in London right now', 'what is the current month'. "
            "(2) Basic weather queries — Galaxy Weather app. Examples: 'is it going to rain', 'temperature outside'. "
            "(3) Simple conversions and math that might seem to need a model. "
            "(4) Privacy-heavy personal data commands (Samsung Health logging, contacts)."
        ),
        "cloud": (
            "HARD edge cases for CLOUD. These should LOOK like local but need cloud's deep reasoning. "
            "Focus: "
            "(1) Multi-paragraph synthesis or comparison questions (e.g. 'differences between X and Y' that need depth). "
            "(2) Open-ended 'how do I...' questions requiring detailed explanation. "
            "(3) Creative short prompts that actually need long generation (e.g. 'write a poem about autumn that uses all five senses'). "
            "(4) Translations of non-trivial sentences with nuance."
        ),
        "search": (
            "HARD edge cases for SEARCH. These should LOOK like local or cloud but need real-time external data. "
            "Focus: "
            "(1) Weather/time queries that add location or future specificity that Galaxy Weather can't cover "
            "('weather in Hanoi on Thursday'). "
            "(2) Local business info requiring real-time query ('is this restaurant open now'). "
            "(3) Product/price comparison requiring live market data. "
            "(4) Recent events / this-week news."
        ),
    },
    "extreme": {
        "local": (
            "EXTREME adversarial LOCAL cases. Tricky, long, or multi-intent queries that still belong to LOCAL. "
            "Examples of what to generate: "
            "- Multi-part personal data task: 'Remember that my Metformin 500mg refill is due every Tuesday "
            "at 7am, and set a reminder for 30 min before, and add it to my calendar as weekly recurring'. "
            "- Voice-dictated filler with hidden simple local task: 'uhh hey so like I need to um set an alarm "
            "for like 6 45 tomorrow morning because I have a meeting'. "
            "- Confused phrasing hiding a simple on-device task: 'can you tell me what the temperature "
            "outside is right now without using the internet or anything'. "
            "- Long personal-data recall: 'My blood pressure reading this morning was 138 over 88, resting "
            "heart rate 72, please log all of this to Samsung Health with today's date'."
        ),
        "cloud": (
            "EXTREME adversarial CLOUD cases. Long, multi-step, or technically deep queries that belong to CLOUD. "
            "Include the design tension of privacy-sensitive generative requests (these are legitimate cloud "
            "per design, user accepts the trade-off). Examples: "
            "- Privacy + long generation: 'I'm anxious about my $150k medical debt — write me a 500-word "
            "emotionally resonant negotiation script I can use on the phone with the billing department'. "
            "- Deep technical: 'Compare the thermodynamic efficiency differences between PEM fuel cells "
            "and SOFCs with at least 3 worked equations and discuss material constraints'. "
            "- Multi-step code: 'Write Python to train a character-level LSTM on my journal entries and "
            "generate new text samples every 10 epochs, including plot of loss curves'. "
            "- Long-form creative with constraints: 'Write a sonnet in Shakespearean form from the "
            "perspective of a dying star, using ocean metaphors throughout'."
        ),
        "search": (
            "EXTREME adversarial SEARCH cases. Queries that mix multiple routing signals but genuinely need "
            "real-time external data. Examples: "
            "- Privacy hint embedded: 'What are the top-rated hiking trails within 20 miles of my house?' "
            "(search-dominant even though it needs location). "
            "- Multi-condition real-time query: 'Find 3 restaurants near Gangnam Station that are open past "
            "midnight right now, have at least 4.2 stars, and don't require reservations'. "
            "- Real-time + niche knowledge: 'Is the Han River frozen right now based on last 48 hours of "
            "temperatures in Seoul?'. "
            "- Market-data question: 'Show me today's intraday chart for TSLA and the top 3 news headlines "
            "that drove its movement'."
        ),
    },
}

COUNTS = {"easy": 8, "medium": 20, "hard": 15, "extreme": 7}

# Keywords that identify each pattern type for edge_case_type labeling
PATTERN_KEYWORDS = {
    "A_time_date": ["time", "date", "month", "day", "year", "now", "current"],
    "B_privacy_mention": ["my home", "my address", "my house", "my location"],
    "C_privacy_generative": ["my debt", "my loan", "my salary", "negotiation", "budget"],
    "D_multi_para_synthesis": ["compare", "differences", "explain", "analyze"],
}


def _classify_edge_case(query: str, difficulty: str) -> str | None:
    """Best-effort edge case pattern labeling for hard/extreme queries.

    Args:
        query (str): The query text.
        difficulty (str): The difficulty level.

    Returns:
        str | None: Pattern name (A_time_date / B_privacy_mention / etc.)
                    or None if no pattern matched.
    """
    if difficulty not in ("hard", "extreme"):
        return None
    q_lower = query.lower()
    for pattern, kws in PATTERN_KEYWORDS.items():
        if any(kw in q_lower for kw in kws):
            return pattern
    return None


def _build_prompt(route_type: str, difficulty: str, count: int) -> str:
    """Build the Gemini generation prompt for a (route_type, difficulty) bucket.

    Args:
        route_type (str): One of "local", "cloud", "search".
        difficulty (str): One of "easy", "medium", "hard", "extreme".
        count (int): Number of queries to generate.

    Returns:
        str: The full prompt to send to Gemini.
    """
    spec = DIFFICULTY_SPECS[difficulty][route_type]
    return (
        BASE_CONTEXT
        + f"\n\n## Task\n\nGenerate exactly {count} diverse {route_type.upper()} "
        + f"queries at DIFFICULTY={difficulty.upper()}.\n\n"
        + f"## {difficulty.upper()} spec for {route_type.upper()}\n\n"
        + spec
        + "\n\n## Diversity requirements\n\n"
        + "- Vary LENGTH: include 1-word commands, short utterances, medium sentences, long multi-sentence queries.\n"
        + "- Vary PHRASING: questions, statements, commands, polite requests, fragments.\n"
        + "- Each query must be distinctly different from the others — no near-duplicates.\n"
        + f"- Output ONLY a JSON array of {count} strings, no other text.\n"
    )


def generate_bucket(
    route_type: str, difficulty: str, count: int, api_key: str
) -> list[dict[str, Any]]:
    """Generate queries for one (route_type, difficulty) bucket via Gemini.

    Args:
        route_type (str): One of "local", "cloud", "search".
        difficulty (str): One of "easy", "medium", "hard", "extreme".
        count (int): Number of queries requested.
        api_key (str): Google AI API key.

    Returns:
        list[dict]: Samples with query, route_label, difficulty, is_privacy_sensitive,
                    edge_case_type, source="seed".
    """
    if genai is None:
        raise ImportError("google-generativeai is required")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")
    prompt = _build_prompt(route_type, difficulty, count)

    # Temperature varies by difficulty: easy is deterministic, extreme pushes diversity
    temp_map = {"easy": 0.1, "medium": 0.4, "hard": 0.7, "extreme": 0.9}
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temp_map[difficulty],
            response_mime_type="application/json",
        ),
    )

    try:
        queries = json.loads(response.text.strip())
    except Exception as e:
        print(f"    ERROR parsing JSON for {route_type}/{difficulty}: {e}")
        return []

    if not isinstance(queries, list):
        return []

    samples = []
    seen = set()
    for q_text in queries:
        if not isinstance(q_text, str):
            continue
        q = q_text.strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        samples.append({
            "query": q,
            "route_label": route_type,
            "difficulty": difficulty,
            "is_privacy_sensitive": detect_privacy_sensitive(q),
            "edge_case_type": _classify_edge_case(q, difficulty),
            "source": "seed",
        })

    return samples[:count]


def main():
    """Generate the 150-query seed set and save to research/data/generated/."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set.")
        return

    all_samples = []

    for route_type in ["local", "cloud", "search"]:
        print(f"\n=== {route_type.upper()} (target: 50) ===")
        for difficulty, count in COUNTS.items():
            print(f"  Generating {count} {difficulty} {route_type} queries...")
            samples = generate_bucket(route_type, difficulty, count, api_key)
            print(f"    Got {len(samples)} unique")
            all_samples.extend(samples)

    # Shuffle for good measure
    random.seed(42)
    random.shuffle(all_samples)

    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "seed_set.jsonl"
    with open(output_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_samples)} samples to {output_path}")

    # Stats
    from collections import Counter
    route_counts = Counter(s["route_label"] for s in all_samples)
    diff_counts = Counter(s["difficulty"] for s in all_samples)
    privacy_count = sum(1 for s in all_samples if s["is_privacy_sensitive"])
    edge_counts = Counter(s["edge_case_type"] for s in all_samples if s["edge_case_type"])

    print(f"\nRoutes: {dict(route_counts)}")
    print(f"Difficulty: {dict(diff_counts)}")
    print(f"Privacy-sensitive: {privacy_count} ({100*privacy_count/max(len(all_samples),1):.1f}%)")
    print(f"Edge case types (hard/extreme only): {dict(edge_counts)}")


if __name__ == "__main__":
    main()
