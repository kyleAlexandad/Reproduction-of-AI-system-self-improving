# C3.1 Postmortem — why the first 5 GIFT-Eval candidates did not beat naive

Run analyzed: `saved_runs/gift_eval_c3_era_smoke/` (5-iteration smoke, `gemini-2.5-flash`,
prompt = original "baseline"). Task: `m4_weekly/W/short`, horizon 13, **reward = -MASE**
(higher = better). Initial naive: **MASE 2.777295**. Best after 5 iters: **still the naive seed**
(improvement 0.0).

## Per-candidate analysis

| id | parent | valid | MASE | reward | idea | why it lost to naive |
|----|--------|-------|------|--------|------|----------------------|
| 0 (seed) | – | ✅ | **2.7773** | -2.7773 | last-value naive | the anchor |
| 1 | 0 | ✅ | 3.7388 | -3.7388 | seasonal-naive (hardcoded season=52) + mean-of-last-5 level + damped linear trend | abandons the last-value anchor for a value ~52 weeks ago **plus** a compounding `trend*(h+1)*0.95^h` term; on near-random-walk weekly data both drift away from `last` |
| 2 | 0 | ✅ | 5.0807 | -5.0807 | SES level over the WHOLE series (α=0.3) + **multiplicative** seasonal factors (last 52 / mean) + linear trend | full-history SES pulls the level toward the global mean (far from `last`); multiplicative weekly factors amplify noise → large drift |
| 3 | 0 | ❌ | – | -inf | damped trend + blended seasonal-naive | **IndexError** (see below) |
| 4 | 0 | ✅ | 2.7773 | -2.7773 | seasonal-trend using `metadata['season_length']` | `season_length` came back as **1**, so `h % 1 == 0` and `context[-1]` everywhere → the formula **collapsed to naive** (an accidental tie, not a real improvement) |
| 5 | 4 | ✅ | 4.1018 | -4.1018 | additive seasonal-trend, season_length from metadata (=1) | used `mean(last 4)` as the level (off the `last` anchor) + a compounding damped trend from a single recent diff → drift |

## The invalid candidate (id 3) — exact bug
```
INVALID: candidate raised IndexError: index 2179 is out of bounds for axis 0 with size 2179
```
Candidate 3 computed a *forward* index into history:
```python
season_length = metadata.get('season_length', 52)            # <-- returned 1, not 52
seasonal_naive_idx_in_history = len(context) - season_length + h   # = len-1+h
if seasonal_naive_idx_in_history >= 0:                        # only checks the LOWER bound
    seasonal_val = context[seasonal_naive_idx_in_history]     # h>=1 -> index == len -> crash
```
Two compounding mistakes: (a) it trusted `metadata['season_length']` as the *data's* seasonal
period, but our scorer passes the **metric seasonality** (`get_seasonality("W") == 1`); (b) the
guard only checked `>= 0`, never the upper bound. With `season_length = 1`, the index hits exactly
`len(context)` at `h = 1` → out of bounds.

## Root causes (cross-cutting)
1. **The prompt was open-ended.** It listed *moving average, seasonal lag, trend extrapolation* as
   co-equal ideas and never told the model that **naive is a strong baseline (MASE 2.7773)** or that
   MA (3.42) and seasonal naive (9.58) are *worse* here. So Gemini "improved" by adding machinery
   that moves the forecast off the last-value anchor — exactly the wrong direction for
   random-walk-like weekly series.
2. **`metadata['season_length'] == 1` is misleading.** It is the metric seasonality (for MASE
   scaling), not the series' seasonal period. Candidates that used it as a lag either crashed
   (id 3) or silently collapsed/distorted (ids 4, 5).
3. **Unsafe forward indexing** (`context[len - k + h]`) caused the crash; end-relative slicing
   (`context[-k:]`) with bound checks would have been safe.
4. **Drift-inducing constructs**: compounding trend terms `trend*(h+1)*φ^h`, full-history SES
   levels, and multiplicative seasonal factors all pull away from `last` on noisy weekly data.

## Search behaviour (not the problem)
FUTS worked correctly: expansions selected parent 0 (the naive root) and its tie (id 4) — the best
nodes — and only id 5 expanded from id 4. The failure was **candidate quality**, not the tree
search. The right lever is the prompt + keeping naive as the anchor, which is what C3.1 changes.

## Fixes applied in C3.1
- **New default prompt `conservative_v2`** in `gift_eval_era_search.py`. It now:
  - states the measured benchmark facts (naive 2.7773 strong; MA 3.4206 worse; seasonal naive 9.5780
    much worse; series ≈ random walk → last value is the best simple predictor);
  - tells the model to treat naive as the **anchor** and make only **small, safe** corrections
    (small *capped* damped trend, high-naive-weight blends, robust clipping);
  - **warns that `metadata['season_length']` is 1 (metric seasonality), not a seasonal period** —
    do not use it as a lag;
  - adds an **INDEXING SAFETY** hard requirement (no `context[len - k + h]`; slice from the end,
    bound-check both ends);
  - includes a **worked conservative template** to anchor the style.
  - The old prompt is still available as `--prompt_version baseline` for A/B.
- **New flags:** `--prompt_version {baseline,conservative_v2}` (default `conservative_v2`) and
  `--initial_candidate PATH` (default = built-in naive). `prompt_version` + `initial_candidate` are
  recorded in `results.json`.
- **Optional conservative seed** `implementation/initial_candidate_conservative.py` (naive + a tiny
  capped damped trend). Measured on this task: **MASE 2.8039 / RMSE 672.60** vs naive
  **2.7773 / 673.44** — i.e. *slightly worse on MASE* (the reward metric), slightly better on RMSE.
  **So naive remains the default seed** (cleanest, best-MASE anchor); the conservative file is
  offered only as an experiment via `--initial_candidate`.
- **C1/C2 left untouched.** The `season_length=1` confusion is addressed purely in the prompt, so
  the C2 scorer stays bit-for-bit reproducible. (A future option is to stop passing the metric
  `season_length` in candidate metadata, but that would change C2 outputs, so we did not.)

## What would count as success next run
At least one **valid** candidate with **MASE < 2.777295** (even 2.70–2.75 is meaningful). Given how
strong naive is, a tie or tiny win is a good outcome; the conservative prompt is designed to make
"tie-or-slightly-better" much more likely than the previous "drift far above naive".
