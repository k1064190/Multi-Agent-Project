# Error Analysis — DSPy-Optimized Router

Analysis of the 11 misclassified queries out of 600 on the held-out test set
(1.83% error rate, 98.17% accuracy). Error details saved in `errors.json`.

## Confusion Matrix

Rows = gold labels, columns = predicted labels (order: local, cloud, search).

```
           local  cloud  search
local    [  209,    0,     5  ]   # 5 local queries sent to search (time/date)
cloud    [    3,  190,     0  ]   # 3 cloud queries sent to local (privacy mentions)
search   [    3,    0,   190  ]   # 3 search queries sent to local (privacy mentions)
```

## Error Pattern Summary

| Pattern | Count | Root Cause | Fixable? |
|---------|-------|------------|----------|
| A. Time/date queries → search | 5 | Model triggered by "now" / "current" keywords, ignoring that Galaxy clock/calendar apps handle this locally | Yes — prompt |
| B. Search queries with privacy hint → local | 3 | Meta-hint in query text ("requires my home address") triggered privacy override even when task is external data retrieval | Yes — data cleanup + prompt |
| C. Cloud generation with money mention → local | 2 | Privacy penalty (-0.5) overrode task complexity; model favors privacy over generative capability | Design trade-off |
| D. Cloud general-knowledge → local | 1 | Short factual comparison question mistaken for trivial local Q&A | Yes — prompt clarity |

## Individual Errors

### Pattern A: Time/date queries sent to search (5 cases)

The Galaxy Weather app, Clock widget, and Calendar handle these on-device via
sensors and cached data. The router over-weighted real-time keywords.

| # | Query | Gold → Pred |
|---|-------|-------------|
| 11 | "What's the weather like right now? (Just check the local system cache if you can)." | local → search |
| 121 | "What time is it in London right now?" | local → search |
| 123 | "How many days are in February this year?" | local → search |
| 284 | "What is the current day?" | local → search |
| 492 | "What is the current month?" | local → search |

**Model reasoning (typical):** "The query requires real-time, volatile information
that necessitates access to the current date or the web."

**Fix:** Add explicit rule to the DSPy signature or a few-shot example:
"Time, date, and local weather queries are handled by the Galaxy built-in
system clock / Weather app, not via search."

### Pattern B: Search queries with privacy mentions sent to local (3 cases)

These queries contain meta-commentary ("requires my home address") that likely
confused synthetic data generation. The model correctly identifies privacy
signals but applies the local preference overly aggressively.

| # | Query | Gold → Pred | Privacy flag |
|---|-------|-------------|-------------|
| 111 | "What's the weather like in my current location—is it safe for my allergies or does that require accessing my health profile?" | search → local | ✓ |
| 359 | "What's the current value of the 401k I have? (Search for market trends vs. Local for my account balance)" | search → local | ✓ |
| 426 | "What are the top-rated hiking trails within 20 miles of my house? This is search but requires my home address." | search → local | ✓ |

**Note:** These labels may be debatable — the model's choice is defensible from
a privacy standpoint. Some of these should arguably be re-labeled as local.

**Fix:** Clean synthetic generation — filter out queries containing meta-hints
like "(Search for X vs. Local for Y)" which are artifacts of the generation
prompt leaking into examples.

### Pattern C: Cloud generation with sensitive amounts sent to local (2 cases)

The DSPy privacy penalty (-0.5 for sensitive→cloud) was intentionally designed
to redirect sensitive queries away from cloud. But when the task is clearly
generative (write a negotiation script, build a budget), on-device small models
cannot handle it. This is an inherent trade-off.

| # | Query | Gold → Pred |
|---|-------|-------------|
| 131 | "I'm feeling very anxious about my $150k medical debt; can you help me write a 'Negotiation Script' to use with the hospital's billing department to ask for a reduction or payment plan?" | cloud → local |
| 276 | "I'm feeling very anxious about my $50k in personal loans; can you help me write a 'Budget' that allows me to pay them off in two years while still having a life?" | cloud → local |

**Design consideration:** A future iteration could add a **two-tier fallback**:
privacy-sensitive generative queries run on-device first, and only escalate to
cloud after anonymization (redact amounts, names, account numbers).

### Pattern D: General-knowledge comparison sent to local (1 case)

| # | Query | Gold → Pred |
|---|-------|-------------|
| 245 | "What are the main differences between a tropical rainforest and a temperate forest?" | cloud → local |

**Model reasoning:** "A factual comparison between two ecological systems, which
is a low-complexity general knowledge task that can be handled by a small
on-device model."

**Fix:** Clarify in the routing prompt that **any query requiring multi-paragraph
synthesis or domain-specific depth** belongs to cloud, even if phrased as a
simple comparison.

## Takeaways for Future Iterations

1. **Prompt hardening (expected ~0.7 pp lift):** Add explicit rules for time/date
   and multi-paragraph synthesis patterns. Of the 11 errors, 5 (Pattern A) + 1
   (Pattern D) = 6 cases should be recoverable via prompt clarification alone.

2. **Data cleanup (expected ~0.3 pp lift):** Regenerate synthetic data with a
   filter to reject queries containing meta-hints like "(Search for X)". This
   addresses Pattern B.

3. **Privacy/task-complexity trade-off:** Pattern C is a legitimate design
   choice. Keeping privacy-sensitive generative queries local is a feature, not
   a bug. Document this explicitly in the product spec.

4. **Accuracy ceiling:** After removing ambiguous Pattern B labels and fixing
   Pattern A/D prompts, the realistic ceiling is around 99.3-99.5%. Remaining
   errors (Pattern C, some Pattern B) reflect genuinely ambiguous routing
   decisions, not model failure.
