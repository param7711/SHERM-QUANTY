---
name: delegate
description: Plan-delegate-supervise workflow for Sherm Quanty — the supervisor (Fable) plans and decomposes the work, delegates build tasks to the opus-builder agent and analysis/validation tasks to the sonnet-analyst agent, then verifies and integrates their outputs before reporting. Use this whenever the user asks to delegate work, split a task across agents/models, "use the delegation system", run build and analysis in parallel, or hands over any multi-part task with both a build component (code, engine, harness, notebook) and an analysis component (selection study, validation, benchmark) — even if they don't say the word "delegate".
---

# Delegate — plan, delegate, supervise

This skill encodes the working pattern this project uses for multi-part tasks:
the session model acts as **supervisor** (plans, delegates, verifies,
integrates); **opus-builder** executes build work; **sonnet-analyst** executes
analysis/validation work. The supervisor never just forwards results — it
checks them.

## 1. Plan (supervisor)

Decompose the request into workstreams that can run **independently and in
parallel**. For each, decide:

- **Build** (create/modify substantial code: engines, harnesses, notebooks,
  refactors) → `opus-builder`
- **Analysis/validation** (selection studies, statistics, validation
  harnesses, verification of a builder's output, reports) → `sonnet-analyst`

If two workstreams depend on each other's outputs, either sequence them or —
better — make the dependency a clearly-marked placeholder the supervisor swaps
in at integration time (e.g. a `FEATURES_LEAN` constant the analysis will later
determine). Placeholders keep the streams parallel.

Tell the user the plan (one short block: who gets what, what you'll do when
results land) before launching.

## 2. Delegate

Launch all independent agents **in one tool block** so they run concurrently,
via the Agent tool with `subagent_type: "opus-builder"` or
`"sonnet-analyst"`. Agents start cold — the brief must be self-contained:

```
You are the {BUILD | ANALYSIS/VALIDATION} agent... (one line of role context)
WORKSPACE: <scratchpad path>            ← deliverables go here
REPO (READ-ONLY): /home/user/SHERM-QUANTY   ← never modified without approval
CONTEXT: <2-6 sentences: what exists, what was already learned, why this task>
YOUR TASK: <numbered, concrete requirements; name exact files to read first>
VERIFY: <the exact self-check the agent must pass before reporting>
REPORT BACK: <the fields you want: paths, verification evidence, results, deviations>
```

Hard constraints to state in every brief: scratchpad-only output; repo is
read-only; no look-ahead in anything financial (bar-t signals use only bars
<= t); yfinance is firewalled here so synthetic fallback paths are the expected
test route.

## 3. Supervise (when notifications arrive)

Completion notifications are the only source of truth about an agent's results
— never predict or pre-report them. For each result:

1. **Verify at least one load-bearing claim yourself** — run the deliverable,
   or spot-check the code (the "did it really run clean?" check). Agents
   self-verify, but the supervisor confirms.
2. **Cross-check between workstreams** — naming mismatches (e.g. `ret_2h` vs
   `ret`), interface drift, conflicting assumptions. This is where parallel
   work breaks; the supervisor is the only one who sees both sides.
3. **Integrate** — swap real results into placeholders, wire outputs together,
   re-run the integrated whole end-to-end.
4. To iterate with an agent that already has context, continue it with
   SendMessage instead of spawning a fresh one.

## 4. Report

Only after verification and integration: report to the user — outcome first,
each agent's contribution summarized (quotable from their reports), then the
integrated result and what's next. Findings that survived your checks are
stated plainly; anything you couldn't verify is flagged as such.

## Standing project rules the supervisor enforces

- **Nothing graduates to the repo without user approval.** Experiments live in
  the scratchpad; the user decides what gets committed. (Bugfixes the user
  already approved are exempt.)
- **Synthetic-data results are illustrative only** — label them, and route
  real-data runs through the user's cloud notebook (Colab/Kaggle/Antigravity).
- **Stability over peak performance** in any model/config selection: worst-fold
  behavior, fold variance, and parameter-count-vs-data (rows/param) outrank a
  flashy mean.
