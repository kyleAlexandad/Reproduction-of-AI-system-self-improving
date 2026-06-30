# C4 — Fair ERA vs Best-of-N on GIFT-Eval `m4_weekly/W/short` (CODE READY, NOT RUN)

C4 compares ERA tree search against best-of-N independent sampling on the GIFT-Eval task, under an
**equal LLM budget**. Per instructions, the experiment has **not** been run here — this folder holds
the script docs + the exact manual commands. (`gift_eval_compare_era_vs_bon.py` lives in
`implementation/`.)

## Why this comparison is fair
Both methods are identical except for parent selection:

| | shared | Method A: ERA | Method B: best-of-N |
|---|---|---|---|
| model | gemini-2.5-flash | same | same |
| prompt | `conservative_v2` | same | same |
| initial seed | naive last-value (scored ONCE, shared) | same | same |
| scorer / reward | C2 scorer, reward = -MASE | same | same |
| budget | — | N Gemini calls | N Gemini calls |
| **parent** | — | **tree-selected (FUTS/PUCT)** → can refine prior candidates | **always the same naive seed** → never reuses prior candidates |

So any difference in outcome is attributable to ERA's tree-search refinement vs independent sampling,
not to model/prompt/seed/scorer differences. (Same design as the toy-task `compare_era_vs_bon.py`.)

## Framing (important)
C3 showed naive is near-optimal on this subset (best ERA candidate came within 5.4e-7 MASE of naive
but did not beat it). **C4 is therefore a PROCESS comparison, not a "beat naive" contest.** Even if
neither method beats naive, ERA can still be the better process if it gets **closer to naive**, has
**fewer invalid candidates**, or produces **more stable / less duplicated** candidates.

## How to run (MANUAL — normal terminal, not the agent sandbox)
See `commands_used.sh`. Smoke (N=10 → 20 Gemini calls):
```bash
cd /Users/zhangweikun/era/implementation
unset GOOGLE_API_KEY
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL=gemini-2.5-flash
python gift_eval_compare_era_vs_bon.py --N 10 --model gemini-2.5-flash \
    --out_dir saved_runs/gift_eval_c4_era_vs_bon_N10
```
Then optionally `--N 20` into `saved_runs/gift_eval_c4_era_vs_bon_N20`.

## Outputs you should expect
```
saved_runs/gift_eval_c4_era_vs_bon_N10/
├── results.json          # full metrics + per-candidate records for both methods
├── summary.csv           # one row per method (+ a naive row)
├── progress_era.csv      # ERA per-candidate rows (incl. seed id 0)
├── progress_bon.csv      # best-of-N per-candidate rows (incl. seed id 0)
├── era_candidates/       # ERA candidate .py files (cand_000=seed, cand_001..N)
├── bon_candidates/       # best-of-N candidate .py files
├── candidate_logs/       # per-candidate scorer outputs (era_NNN/, bon_NNN/, seed/)
├── era_vs_bon_mase.png   # best-so-far MASE (generated, excl. seed) for both methods + naive line
├── README_C4.md          # this file
└── commands_used.sh
```

## Metrics reported (per method, in `summary.csv` / `results.json`)
- naive baseline MASE
- ERA best MASE **including** the initial seed
- ERA best **generated** MASE **excluding** the seed
- best-of-N best MASE
- whether **ERA beat best-of-N** (`winner_by_best_generated_MASE`)
- whether **either** method beat naive (`either_beat_naive`)
- valid / invalid counts for ERA and for best-of-N
- closest distance to naive for each method (`distance_to_naive_excl_seed`)
- number of exact/near-duplicate candidates (`n_duplicate_programs`, `n_unique_valid_MASE`)
- best-so-far curves (the plot + the `*_so_far` arrays in `results.json`)

## What would count as a meaningful C4 result
Because naive is near-optimal, success is a clear **process** difference under equal budget, e.g.:
- ERA's best generated candidate is **closer to naive** than best-of-N's (smaller `distance_to_naive`);
- ERA has **fewer invalid** candidates and/or **fewer duplicates**;
- ERA's best-so-far curve **descends** (refinement) while best-of-N stays flat at template level
  (~2.804) — which is exactly the behaviour C3 hinted at.
A tie or a tiny edge either way is still informative; beating naive is a bonus, not the bar.

## Status
`gift_eval_compare_era_vs_bon.py` is written, passes `python -m py_compile`, imports cleanly, and
reuses the verified C3 scorer + prompt builder. **Gemini / ERA / best-of-N NOT executed here.**
Ready for the manual C4 run.
