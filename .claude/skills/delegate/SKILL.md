---
name: delegate
description: Plan-delegate-supervise workflow for Sherm Quanty — Opus 5 supervises (plans, decomposes, verifies, integrates) and delegates to three agents: opus-builder (Opus, heavy build work), sonnet-analyst (Sonnet 5, analysis/validation), and haiku-scout (Haiku 4.5, fast mechanical lookups and checks). Use this whenever the user asks to delegate work, split a task across agents/models, "use the delegation system", run build and analysis in parallel, or hands over any multi-part task combining build work (code, engine, harness, notebook) with analysis work (selection study, validation, benchmark) — even if they don't say the word "delegate".
---

# Delegate — plan, delegate, supervise

This skill encodes the working pattern this project uses for multi-part tasks.
**Opus 5 is the supervisor**: it plans, decomposes, delegates, verifies, and
integrates. It never just forwards an agent's results — it checks them.

Three delegate agents, matched to the weight of the work:

| Agent | Model | Use for |
|---|---|---|
| `opus-builder` | Opus (medium effort) | Substantial build work: engine modules, harnesses, notebooks, refactors |
| `sonnet-analyst` | Sonnet 5 | Analysis & validation: selection studies, statistics, validation notebooks, verifying a builder's output |
| `haiku-scout` | Haiku 4.5 | Fast mechanical tasks: locating files/symbols, summarizing a file, running an existing script, checking an invariant, collecting results |

Routing rule of thumb: if the task needs *design judgment*, it's a builder or
analyst job. If it's already fully specified and just needs doing, it's a scout
job — sending it to a heavy model wastes time and money.

## 1. Plan (supervisor)

Decompose the request into workstreams that can run **independently and in
parallel**, and assign each to the lightest agent that can do it well.

If two workstreams depend on each other, either sequence them or — better —
make the dependency a clearly-marked placeholder the supervisor swaps in at
integration time (e.g. a `FEATURES_LEAN` constant the analysis will determine
later). Placeholders keep the streams parallel.

Tell the user the plan in one short block (who gets what, what happens when
results land) before launching.

## 2. Delegate

Launch all independent agents **in one tool block** so they run concurrently,
via the Agent tool with `subagent_type: "opus-builder" | "sonnet-analyst" |
"haiku-scout"`. Agents start cold, so each brief must be self-contained:

```
You are the {BUILD | ANALYSIS/VALIDATION | SCOUT} agent... (one line of role context)
WORKSPACE: <scratchpad path>                 ← deliverables go here
REPO (READ-ONLY): /home/user/SHERM-QUANTY    ← never modified without approval
CONTEXT: <2-6 sentences: what exists, what was already learned, why this task>
YOUR TASK: <numbered, concrete requirements; name exact files to read first>
VERIFY: <the exact self-check the agent must pass before reporting>
REPORT BACK: <the fields you want: paths, verification evidence, results, deviations>
```

State these constraints in every brief: scratchpad-only output; the repo is
read-only; no look-ahead in anything financial (a signal at bar t uses only
bars <= t); yfinance is firewalled in this environment, so synthetic fallback
paths are the expected test route.

## 3. Supervise (when notifications arrive)

Completion notifications are the only source of truth about an agent's results
— never predict or pre-report them. For each result:

1. **Verify at least one load-bearing claim yourself** — run the deliverable or
   spot-check the code. Agents self-verify; the supervisor confirms. This is
   where fabricated or overstated passes get caught.
2. **Cross-check between workstreams** — naming mismatches (e.g. `ret_2h` vs
   `ret`), interface drift, conflicting assumptions. Parallel work breaks here,
   and the supervisor is the only one who sees both sides.
3. **Integrate** — swap real results into placeholders, wire outputs together,
   re-run the integrated whole end-to-end.
4. To iterate with an agent that already has context, continue it with
   SendMessage instead of spawning a fresh one.

## 4. Report

Only after verification and integration: report to the user — outcome first,
each agent's contribution summarized, then the integrated result and what's
next. State plainly what survived your checks; explicitly flag anything you
could not verify.

## Standing project rules the supervisor enforces

- **Nothing graduates to the repo without user approval.** Experiments live in
  the scratchpad; the user decides what gets committed. (Bugfixes the user
  already approved are exempt.)
- **Synthetic-data results are illustrative only** — label them as such, and
  route real-data runs through the user's cloud notebook (Colab / Kaggle /
  Antigravity).
- **Stability over peak performance** in any model/config selection: worst-fold
  behaviour, fold variance, and parameter-count-vs-data (rows/param) outrank a
  flashy mean.
- **Assert invariants that must hold by construction** (probabilities partition
  to 1, every label reachable, no future data in a bar-t signal). First
  principles catches silent bugs that eyeballing charts does not.
