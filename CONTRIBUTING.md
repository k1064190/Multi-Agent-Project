<!-- ABOUTME: Repo conventions for the router DSPy optimization research scope. -->
<!-- ABOUTME: Mirrors how the development tree was actually run during Stage 1-5 + Stage 7. -->

# Contributing

This document captures the conventions used during the project's
research cycles (Stage 1-5 + Stage 7). They were not improvised —
they evolved across multiple stages and were validated by post-execution
code reviews each time.

---

## Environment

Python 3.11 via micromamba (or any environment manager that gives
you isolated 3.11). The development tree used:

```
$HOME/micromamba/envs/llm_router/bin/python
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` uses loose `>=` pins. If you need exact
reproducibility for a specific run, freeze locally with `pip freeze`
and pin in your fork.

---

## Commit format

```
feat(stage-X.Y): <one-line subject>

<body explaining the why, not the what>
```

Other prefixes used in the development tree:

- `fix(stage-X.Y):` — bug fix landed during a stage
- `docs(stage-X.Y):` — Korean retro or doc-only change
- `refactor(...):` — non-behavioral cleanup
- `chore(...):` — tooling, deps

The `(stage-X.Y)` scope ties each commit back to a numbered stage.

Never use `--no-verify`, `--no-hooks`, `--no-pre-commit-hook`. If a
hook fails, fix the underlying issue rather than bypassing it.

---

## Stage retrospectives

Every stage closes with a Korean retrospective at
`docs/stage_<n>/<name>.md`. The four mandatory sections are:

| Section | Korean | Content |
|---------|--------|---------|
| Why | `왜` | The motivation / question the stage was answering |
| What | `무엇` | Concrete outputs (runs, artifacts, scores, decisions) |
| How | `어떻게` | Methodology (commands, knobs, gotchas, reproduction) |
| Retrospect | `회고` | What went well, what went wrong, what to change |

The retrospectives are deliberately in Korean even though the rest
of the codebase is in English. They are a personal reasoning track,
not user-facing documentation. Keep them honest — failed runs and
wrong premises are *more* valuable than greatest-hits highlight reels.

---

## Code style

Match existing patterns in each file. Some tactical conventions:

- **ABOUTME comments**: every code file starts with two lines
  (`<!-- ABOUTME: ... -->` for markdown, `# ABOUTME: ...` for python)
  describing what the file does.
- **Function docstrings**: document every argument with type and
  semantic meaning. Include array/tensor shapes where relevant.
- **No mock or placeholder code**: implementations either work or
  they aren't merged.
- **Surgical changes**: every changed line should trace directly to
  the user's request or stage scope. No drive-by cleanups.

---

## Architecture references

If you are touching the routing classifier, read
`docs/stage_4/alt_optimizers_and_ablation.md` +
`docs/stage_5/vllm_migration_and_test_eval.md`. The BFRS-vs-MIPROv2
decomposition is non-obvious.

Do not skip the retros — they record the *failures* that shaped the
current design, which the code alone does not show.
