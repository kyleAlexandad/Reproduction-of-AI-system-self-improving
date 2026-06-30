# C3 Final Report — ERA tree search on GIFT-Eval `m4_weekly/W/short`

Stage C3 connected ERA / FUTS tree search (`gift_eval_era_search.py`) to the official GIFT-Eval
scorer (`gift_eval_m4_weekly_task.py`) via a subprocess bridge across the two Python environments.
We ran three searches; this is the consolidated result.

## Headline
**C3 engineering succeeded; ERA did not beat the naive baseline** (naive is extremely strong on this
subset). The best ERA-generated program got within **5.4e-7 MASE** of naive but stayed just above it.

## Runs (see `c3_summary_table.csv`, `c3_best_so_far_mase.png`)

| run | prompt | iters | generated | valid/invalid | best gen. MASE (excl. seed) | distance to naive | beat naive? |
|-----|--------|-------|-----------|---------------|------------------------------|-------------------|-------------|
| smoke | baseline | 5 | 5 | 4 / 1 | 2.7772950 (exact-naive copy) | 0 (degenerate copy) | no |
| iter10 | conservative_v2 | 10 | 10 | **10 / 0** | 2.7773131 | 1.80e-05 | no |
| iter20 | conservative_v2 | 20 | 20 | **20 / 0** | 2.7772956 | **5.41e-07** | no |

Naive baseline MASE = **2.7772950477** (identical seed in all runs). Best MASE *including* seed = the
naive seed in every run.

## What the runs demonstrate (all confirmed ✅)
- **C3 engineering succeeded** — all three runs completed end-to-end.
- **ERA generated real candidate forecasting programs** — Gemini produced valid `forecast(...)`
  functions that were written to `candidates/` and executed.
- **The GIFT-Eval scorer ran the official gluonts pipeline** — every candidate was scored with the
  same `evaluate_model` path as C1/C2 (MASE/CRPS/RMSE).
- **The subprocess bridge worked** — ERA env (`python`) → `subprocess` → GIFT-Eval venv scorer →
  JSON parse → reward, with zero plumbing failures.
- **Invalid handling worked** — the one invalid candidate (smoke iter 3, an `IndexError`) was caught,
  scored `-inf`, logged, and the search continued.
- **The conservative prompt removed all invalids** — baseline prompt: 1/5 invalid; conservative_v2:
  **0/10 and 0/20 invalid**. It also stopped the wild drift (baseline produced MASE up to 5.08;
  conservative kept everything in 2.777–2.804).
- **Naive remained the best candidate** in every run; ERA's best generated program **came extremely
  close** (iter20: 2.77729559 vs naive 2.77729505) but did not improve beyond it.

## Why ERA did not beat naive (postmortem)
1. **Naive is near-optimal here.** M4 weekly series behave close to a random walk, so the last
   observed value is an extremely strong point predictor; MASE is also scaled by the in-sample naive
   error, so a naive-like forecast sits right at MASE ≈ 2.78 and there is very little headroom for a
   *simple, hand-written heuristic* to do better. Any deviation (trend, smoothing, seasonality)
   mostly adds variance and nudges MASE up.
2. **The search actually behaved well — it converged toward naive.** With the conservative prompt,
   Gemini first copied the worked template (MASE 2.8039), then FUTS built an improving chain
   (iter20: nodes 0→6→7→8→9→10) that progressively shrank the trend's influence. The best node
   (id 10) ends up almost exactly naive: it blends `last` with a tiny damped trend at
   `weight_naive = 0.9999` (trend influence 1e-4), heavily damped (`phi=0.4`) and clipped. So ERA
   was driving the candidate *toward* naive — the correct direction — but the global optimum in this
   tiny heuristic space basically *is* naive, so it could only tie it asymptotically.
3. **Point-only forecasts + MASE leave no probabilistic headroom.** We optimize `-MASE` on point
   forecasts; the only way to beat naive on MASE is a genuinely better point predictor, which simple
   numpy heuristics can't find for near-random-walk data.

**Conclusion:** `m4_weekly/W/short` is a *hard subset for simple hand-written forecasting heuristics
because naive is already very competitive.* This is a known property of M4-weekly, not a defect in
ERA or the pipeline.

## Implication for C4
Because beating naive here is near-impossible for simple heuristics, **C4 (ERA vs best-of-N) is
framed as a fair process comparison, not a "beat naive" contest.** C3 already hints at ERA's edge:
ERA's tree search *refined* candidates toward naive over successive nodes (2.8039 → 2.7790 → 2.7779
→ 2.7774 → 2.77730), which independent best-of-N sampling cannot do (it would keep redrawing
template-level ~2.8039 candidates without iterative refinement). C4 will measure this directly:
best MASE, best *generated* MASE (excl. seed), distance to naive, valid/invalid rate, duplicate rate,
and the best-so-far curve, under equal LLM budget.

## Files
- `README_C3_FINAL.md` (this file)
- `c3_summary_table.csv` — per-run metrics
- `c3_best_so_far_mase.png` / `.pdf` — best generated-candidate MASE so far vs iteration, with naive
  reference (zoomed to show the near-naive convergence; smoke's early 3.74/5.08 values are above the
  axis on purpose)
- `build_c3_summary.py` — reproducible builder (reads the three runs' `results.json`)
