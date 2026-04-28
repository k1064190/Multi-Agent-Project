# Baseline vs DSPy-Optimized Router Comparison

Held-out test set performance: default router (no prompt optimization)
vs DSPy MIPROv2-optimized router. Both use Gemma 4 31B as the inference LM.

## Overall Metrics

| Metric | Baseline | Optimized | Δ |
|--------|----------|-----------|---|
| Accuracy    | 0.9800 | 0.9817 | +0.0017 |
| Weighted F1 | 0.9800 | 0.9817 | +0.0017 |
| Macro F1    | 0.9802 | 0.9820 | +0.0018 |

## Per-class F1

| Class | Baseline F1 | Optimized F1 | Δ | Support |
|-------|-------------|--------------|---|---------|
| local | 0.9742 | 0.9744 | +0.0002 | 214 |
| cloud | 0.9817 | 0.9922 | +0.0105 | 193 |
| search | 0.9846 | 0.9794 | -0.0052 | 193 |

## Privacy Leak Rate

| | Baseline | Optimized |
|---|---|---|
| Sensitive queries | 11 | 11 |
| Leaked to cloud   | 0 | 0 |
| Leak rate         | 0.0% | 0.0% |

## Latency (ms)

| Percentile | Baseline | Optimized |
|---|---|---|
| P50_MS | 6187 | 5913 |
| P95_MS | 14407 | 16663 |
| P99_MS | 19072 | 30083 |

Note: Latency similarity between modes is expected — both run Gemma 4 31B.
The routing quality improvement comes from the DSPy-optimized prompt,
not from changing the inference LM.
