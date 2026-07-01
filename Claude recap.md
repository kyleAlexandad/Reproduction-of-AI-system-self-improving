# Claude Recap — ERA Reproduction Project

> Last updated: 2026-07-01 · Repo root: `/Users/zhangweikun/era` (git root) · Code: `/Users/zhangweikun/era/implementation`

---

## 0. Start Here / Current State

This project **reproduces and lightly extends `google-research/era`** (Nature paper *"An AI system to
help scientists write expert-level empirical software"*). ERA = an LLM (Gemini) writes/edits a Python
function → a local sandbox runs it → it is auto-scored → **Flat UCB Tree Search (FUTS)** picks a parent
and iterates. The central research question we test: **is ERA's tree search actually better than
best-of-N independent sampling?**

**Status: essentially complete.** All planned experiments are done across 4 tracks — toy regression
(California Housing), classification (Breast Cancer), GIFT-Eval time-series, and scRNA-seq batch
integration. Everything is **committed and pushed** to `repro/main`.

- **Latest commit:** `0c9ccb2` (README promotes scRNA to Stage 4). **Working tree is clean.**
- **Recommendation:** move to **final report consolidation**, not more scRNA experiments unless
  explicitly requested. Both ERA and best-of-N already reach the reference on the PBMC3k bridge.
- **Official Kaggle/HCA scRNA benchmark remains BLOCKED** (private 3GB dataset + R/kBET), but the
  **PBMC3k public-data bridge is complete** and stands in for it.

---

## 1. Hard Rules for Future Agents

- **Do NOT create new per-stage Markdown files.** Only maintain (1) this `Claude recap.md` and (2) the
  root `README.md` / overall report — and only when asked. (User docs policy, 2026-06-30.)
- **Do not modify `README.md` unless the user explicitly asks.**
- **Do not commit** secrets, Kaggle tokens, data caches, `scRNA-env/`, `gift-eval/`, `.h5ad` files, or
  `__pycache__`. These are gitignored — keep them out.
- **Do not use the ERA env for scanpy/anndata/scib**, and **never import scanpy into the ERA
  controller.** Score scRNA/GIFT-Eval candidates via **subprocess into task-specific venvs**.
- **The user runs anything that spends Gemini calls.** Prepare code + give exact commands; do NOT run
  Gemini / ERA / best-of-N yourself. Keep `N` small; **never use `gemini-2.5-pro`**.
- **GIFT-Eval native libs segfault under the agent sandbox** (`statsforecast`/`numba` exit 139) → GIFT-Eval
  scoring must run in a normal terminal.
- **zsh gotcha:** avoid multiline commands with blank lines or trailing spaces after backslashes.

---

## 2. Environment and Repository

**Three isolated Python environments — never mix them:**

| Env | Path | Purpose |
|-----|------|---------|
| **ERA** (controller) | brew `python` = `/opt/homebrew/opt/python@3.12/libexec/bin/python` (3.12.13, numpy 2.5) | `futs.py`, `llm.py`, all `*_era_search.py` / `*_compare_*.py` controllers |
| **GIFT-Eval** venv | `/Users/zhangweikun/era/gift-eval/.venv/bin/python` (3.12.13, gluonts 0.15.1, numpy 1.26.4) | GIFT-Eval scorers only |
| **scRNA** venv | `/Users/zhangweikun/era/scRNA-env/bin/python` (3.11.15, numpy 1.26.4, anndata 0.11.3, scanpy 1.10.4, scib 1.1.7) | scRNA scorers only (R/kBET deferred, NOT installed) |

**Gemini setup (every new terminal — keys are session-only):**
```bash
unset GOOGLE_API_KEY               # SDK prefers it if both are set
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL=gemini-2.5-flash
```
- **Model choice:** `gemini-2.5-flash` for the harder tasks (GIFT-Eval, scRNA, Breast Cancer);
  `gemini-2.5-flash-lite` for the earlier California Housing toy comparisons. **Pro is banned** (cost).
  `gemini-2.5-flash-image` has zero free-tier quota → never use it.
- **Code uses `google-genai`** (`from google import genai`), not `google-generativeai`.
- `sandbox.py` is a **minimal local executor and is INSECURE** — toy use only.
- **Git:** `origin` = upstream `google-research/era` (no push); `repro` =
  `github.com/kyleAlexandad/Reproduction-of-AI-system-self-improving` (push via macOS keychain).
  Everything is committed to `repro/main` @ `0c9ccb2` (working tree clean).

---

## 3. Completed Experiment Summary

| Stage | Task | Status | Key result | Output dir |
|-------|------|--------|-----------|-----------|
| **1** | California Housing regression (official ERA toy) | ✅ | 30-iter best −0.5776 (from −0.7339) | `saved_runs/playground_s3e1_iter30/` |
| **2** | California Housing ERA vs best-of-N | ✅ | **ERA 3/3**; avg −0.5823 vs −0.6058 | `saved_runs/era_vs_bon/`, `saved_runs/repeated_era_vs_bon/` |
| **3** | Breast Cancer classification (ROC-AUC) | ✅ | **ERA 3/3**; avg 0.9888 vs 0.9808 | `saved_runs/breast_cancer_era_vs_bon/`, `saved_runs/repeated_breast_cancer_era_vs_bon/` |
| **C1–C6** | GIFT-Eval time-series | ✅ | **ERA beats naive AND best-of-N on m4_hourly** | `saved_runs/gift_eval_c*/` |
| **D1/D2** | Synthetic scRNA (smoke + ERA wrapper) | ✅ | ERA loop works; 1.0699→1.2793 | `saved_runs/scrna_d1_synthetic_smoke/`, `scrna_d2a_synthetic_era_smoke/` |
| **D3A-alt** | PBMC3k real AnnData smoke | ✅ | pca 1.0275 < batch-centered 1.0701 | `saved_runs/scrna_d3a_realdata_smoke/` |
| **D3B-alt** | PBMC3k ERA smoke (10-iter) | ✅ | 1.0275→**1.070070**, 11/11 valid | `saved_runs/scrna_d3b_pbmc3k_era_iter10_conservative/` |
| **D3C-alt** | PBMC3k ERA vs best-of-N (N=10) | ✅ | **effective tie** (both ~1.070070) | `saved_runs/scrna_d3c_pbmc3k_era_vs_bon_N10/` |
| — | scRNA consolidated figure | ✅ | one 2-panel arc figure + README §8 | `saved_runs/scrna_summary/` |

---

## 4. California Housing Results (Stages 1–2)

Metric = **negative RMSE** (`−RMSE`, higher/closer-to-0 is better). Model = `gemini-2.5-flash-lite`.

**Stage 1 — search progress** (`playground_s3e1.py`, `compare_era_vs_bon.py`):

| | neg RMSE |
|--|--|
| Initial baseline | **−0.7339** |
| 10-iteration best | ≈ **−0.5785** |
| 30-iteration best | **−0.5776** (barely beats 10 → small toy search space) |

**Stage 2 — ERA vs best-of-N** (equal budget N LLM calls each):
- **Single, N=20** (`saved_runs/era_vs_bon/`): ERA **−0.5735** ✅ > BoN −0.6149. Invalids: **ERA 3/20 vs
  BoN 13/20** (ERA much more reliable).
- **Repeated, N=10 × 3** (`saved_runs/repeated_era_vs_bon/`): **ERA won 3/3**; avg **ERA −0.5823 vs BoN
  −0.6058** (+0.0234 gap). Invalids: ERA 6/30 (20%) vs BoN 10/30 (33%).

Scripts: `playground_s3e1.py`, `compare_era_vs_bon.py`, `repeat_era_vs_bon.py`.

---

## 5. Breast Cancer Results (Stage 3)

Task = sklearn breast-cancer **classification, ROC-AUC** (higher better). Weak baseline =
`DecisionTreeClassifier(max_depth=1)`. Model = `gemini-2.5-flash`.

| | ROC-AUC |
|--|--|
| Baseline (depth-1 stump) | 0.8468 |
| ERA best (10 iterations) | **0.9886** |
| ERA vs BoN, single N=10 | **ERA 0.9943** > BoN 0.9841 (invalids 0/10 both) |
| ERA vs BoN, repeated N=10 × 3 | **ERA 3/3**; avg **0.9888** vs 0.9808 (invalids 0/30 both) |

Margin is small (~0.008 AUC — easy task, BoN also reaches ~0.98) but ERA wins consistently.
Scripts: `playground_breast_cancer.py`, `compare_era_vs_bon_breast_cancer.py`,
`repeated_breast_cancer_era_vs_bon.py`.

---

## 6. GIFT-Eval Results (Stage C)

Real public time-series benchmark (`SalesforceAIResearch/gift-eval`). **Runs in its own venv**; ERA
controller bridges via **subprocess** (never imports gluonts). Reward = **`−MASE`** (MASE is
lower-better). Candidate interface an LLM generates:

```python
def forecast(context, prediction_length, freq, metadata=None) -> np.ndarray  # length == prediction_length
```

**Progression:**
- **C1** — official Naive baseline reproduced the `naive.ipynb` numbers exactly on `m4_weekly/W/short`
  (MASE **2.7773**).
- **C2** — ERA-scorable wrapper (`forecast` → `reward = −MASE`; invalid → `−inf`).
- **C3/C4 (m4_weekly)** — **naive is near-optimal here**, so ERA **tied/matched naive but did not beat
  it**; ERA still **beat the best-of-N generated candidates** (process win, not a naive-beating win).
- **C5 subset scout** — found subsets where naive is NOT dominant (ratio = best-non-naive / naive MASE):

  | Dataset | naive | best non-naive | verdict |
  |--|--|--|--|
  | m4_weekly | 2.7773 | 2.78 (ratio 1.002) | naive dominates ❌ |
  | **m4_hourly** | **11.6077** | seasonal_naive **1.1932** (ratio 0.10) | **strong C6 target** ✅ |
  | hospital | 0.9676 | moving_average 0.8139 (ratio 0.84) | secondary ✅ |
  | covid_deaths | 46.9124 | damped_trend 46.0975 | marginal 🟡 |

- **C6B (m4_hourly) — THE STRONGEST REAL RESULT: ERA clearly beats naive AND best-of-N.** From the weak
  naive seed, ERA **discovered daily seasonality (period 24)**:
  - **ERA-only, 10-iter:** seed naive MASE **11.607687602745548** → ERA best **1.1383830818815497**
    (even beats the seasonal-naive reference 1.1932).
  - **ERA vs BoN, N=10:** ERA **1.1932101877** vs BoN **1.3651106991** (winner ERA; 10/0 valid both).
  - **ERA vs BoN, N=20:** ERA **1.19321019** vs BoN **1.36511070** (same outcome, gap stable in N; ERA
    explored more distinct solutions).

  This is the one real benchmark where **ERA's tree-search advantage clearly holds** — because the strong
  structure (period-24 seasonality) is discoverable and naive is weak.

**Key scripts:** `gift_eval_task.py` (parametric scorer, `--dataset/--freq/--term`),
`gift_eval_era_search.py`, `gift_eval_compare_era_vs_bon.py`, `gift_eval_subset_scout.py`.
(`gift_eval_m4_weekly_task.py` = the original C2 scorer, kept for backward-compat.)
Outputs under `saved_runs/gift_eval_c6b_*` (and c1–c6a dirs). To hand-score a candidate:
`gift-eval/.venv/bin/python -u gift_eval_task.py --dataset m4_hourly --freq H --term short --candidates f.py --out-dir /tmp/x`.

---

## 7. scRNA-seq Results (Stage D)

Upstream ERA task lives at `implementation/notebooks/single_cell_batch_integration.ipynb`. Same
two-env pattern as GIFT-Eval. **Reward is HIGHER-is-better** (opposite of GIFT-Eval's `−MASE`), so
best = MAX, `delta_vs_seed` POSITIVE = improvement.

### 7.1 Official Kaggle/HCA benchmark — BLOCKED
- Official interface an ERA candidate implements:
  ```python
  def eliminate_batch_effect_fn(adata, config) -> AnnData   # returns adata with adata.obsm["X_emb"]
  ```
  Candidate **must NOT use `cell_type`**; only `scanpy` allowed among specialized packages.
- Official score = mean of **12 scIB metrics** (needs R + kBET/LISI); reference gemini-3-pro ERA ≈ 0.677.
- **Dataset blocked:** `vsubhashinigoog/single-cell-batch-integration` is **private + ~3GB**; a valid
  Kaggle token lists public datasets fine but returns **`403 datasets.get denied`** on this one. Manual
  browser download too slow. → Official paper-scale scRNA is a **future blocker, not current mainline**.

### 7.2 Synthetic scRNA (D1/D2A) — DONE
- Reduced **Python-only proxy score** = bio-preservation proxy + batch-mixing proxy (silhouette-based;
  NOT scIB). Candidate interface as above; `cell_type` popped before scoring.
- Baselines (D2A, n_comps=10): **PCA ≈ 1.0699**, **batch-centered reference ≈ 1.3408**.
- **D2A ERA runs (Gemini):** 5-iter PCA **1.0699 → 1.2793**; 10-iter → **1.2525** (valid/invalid 10/1).
  The synthetic LLM→scorer→FUTS loop works.
- Scripts: `scrna_synthetic_smoke.py`, `scrna_synthetic_task.py`, `scrna_era_search.py`.

### 7.3 PBMC3k real-data bridge (D3A-alt) — DONE
- Uses `scanpy.datasets.pbmc3k()` (~5.6 MB) as a public stand-in: 500 cells × 2000 HVGs, **3 artificial
  batches** (`batch_strength 0.8`), proxy `cell_type` = **Leiden clusters** on the clean data.
- Baselines: **`pca` 1.0274980964** < **`batch_centered_pca` 1.070069915 (1.070070)**; 2/2 valid.
- Output: `saved_runs/scrna_d3a_realdata_smoke/`. Script: `scrna_realdata_smoke.py --source scanpy_pbmc3k`.

### 7.4 PBMC3k ERA smoke (D3B-alt) — DONE, STRONG SUCCESS
- Scorer `scrna_realdata_task.py` reproduces D3A-alt baselines exactly (pca 1.027498, bc 1.070070);
  controller `scrna_era_search.py --task pbmc3k`.
- **First 5-iter run:** initial 1.0274980964 → best **1.028066**, **2/4 valid** — 4 invalids from
  Scanpy/ComBat API misuse (`scanpy.external.pp.combat`, `sc.tl.combat`, `regress_out(n_jobs=-1)`).
- **D3B.1 prompt hardening** — new `pbmc3k_conservative_v2` (now default for pbmc3k): bans
  `scanpy.external.pp`, `sc.tl.combat`, unsafe ComBat, external integration APIs, `n_jobs=-1`; encourages
  normalize_total→log1p→PCA + per-batch mean-centering; states measured refs.
- **10-iter conservative run (2026-07-01):** initial **1.0274980964** → best **1.0700699151**,
  **valid/invalid 11/0**. ERA **rediscovered batch-centering** exactly: best candidate =
  normalize_total→log1p→PCA(20)→per-batch mean-centering in the PC embedding space = the reference
  method, matched to full precision.

### 7.5 PBMC3k ERA vs best-of-N (D3C-alt, N=10) — DONE, EFFECTIVE TIE
- `scrna_compare_era_vs_bon.py`; both methods share model/prompt(`pbmc3k_conservative_v2`)/PCA-20
  seed/scorer/reward and spend N Gemini calls each; only parent selection differs.
- Seed reward **1.0274980964**. **ERA best 1.0700699151** (10/0 valid); **BoN best 1.0700699166** (10/0
  valid); both reached the reference. Script reports BoN winner by only **~1.4e-9** = **float32 (ERA) vs
  float64 (BoN) rounding**, not a real difference → **effective tie**. 7/10 BoN draws independently hit
  the ~1.07007 method.
- **Conclusion:** after prompt hardening the PBMC3k bridge is easy — both methods reliably rediscover
  batch-centered PCA, so ERA's tree-search edge doesn't show here (unlike GIFT-Eval m4_hourly).

**scRNA scripts:** `scrna_synthetic_smoke.py`, `scrna_synthetic_task.py`, `scrna_realdata_smoke.py`,
`scrna_realdata_task.py`, `scrna_era_search.py`, `scrna_compare_era_vs_bon.py`, `scrna_plot_summary.py`.

---

## 8. Current Best Overall Conclusions

- **ERA official toy reproduction works** (California Housing, −0.7339 → −0.5776).
- **ERA beats best-of-N on California Housing** repeated runs (3/3).
- **ERA beats best-of-N on Breast Cancer** repeated runs (3/3).
- **ERA clearly beats best-of-N on GIFT-Eval m4_hourly** — and beats naive too (the strongest real
  result; ERA discovered period-24 seasonality).
- **scRNA synthetic and PBMC3k bridge work** end-to-end; ERA rediscovers batch-centered PCA.
- **PBMC3k ERA vs BoN is an effective tie** — the task is easy after prompt hardening (both hit the ref).
- **Official full scRNA benchmark remains blocked** by dataset access + R/kBET complexity.

---

## 9. Recommended Next Steps

1. **Stop adding new scRNA experiments** unless explicitly needed.
2. **Consolidate the final project report / README** (this is the recommended focus).
3. *Optional:* inspect the best D3C candidates and summarize methods.
4. *Optional:* repeated D3C only if the user wants statistics — **low value** (both already hit the ref).
5. *Future:* official Kaggle/HCA scRNA once dataset access + R/kBET are solved. (Lighter upstream
   fallback if scRNA stays blocked: the `flu-cornell-jhu-hierarchsir.ipynb` notebook — pure numpy/pandas,
   no R, has published ERA-vs-BoN numbers.)

---

## 10. Important Commands

```bash
# --- Gemini env (every new terminal) ---
unset GOOGLE_API_KEY
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL=gemini-2.5-flash

cd /Users/zhangweikun/era/implementation

# --- GIFT-Eval m4_hourly ERA vs best-of-N (the key real result; run in a NORMAL terminal) ---
python gift_eval_compare_era_vs_bon.py --dataset m4_hourly --freq H --term short \
  --initial_seed naive --N 10 --model gemini-2.5-flash \
  --out_dir saved_runs/gift_eval_c6b_era_vs_bon_m4_hourly_naive_N10

# --- scRNA PBMC3k ERA 10-iter run (D3B) ---
python scrna_era_search.py --task pbmc3k --source scanpy_pbmc3k --n_cells 500 \
  --iterations 10 --model gemini-2.5-flash --prompt_version pbmc3k_conservative_v2 \
  --out_dir saved_runs/scrna_d3b_pbmc3k_era_iter10_conservative

# --- scRNA PBMC3k ERA vs best-of-N (D3C; 2N = 20 Gemini calls) ---
python scrna_compare_era_vs_bon.py --task pbmc3k --source scanpy_pbmc3k --n_cells 500 \
  --n_batches 3 --batch_strength 0.8 --N 10 --model gemini-2.5-flash \
  --prompt_version pbmc3k_conservative_v2 --initial_seed pca \
  --out_dir saved_runs/scrna_d3c_pbmc3k_era_vs_bon_N10

# --- Hand-score any candidate (no Gemini) ---
/Users/zhangweikun/era/scRNA-env/bin/python -u scrna_realdata_task.py --candidate f.py --out-dir /tmp/x
/Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_task.py \
  --dataset m4_hourly --freq H --term short --candidates f.py --out-dir /tmp/x

# --- Push (user's repo) ---
git add -A && git commit -m "..." && git push repro main
```

---

## 11. Known Pitfalls

- **zsh:** no multiline commands with blank lines / trailing spaces after backslashes.
- **Kaggle scRNA dataset:** API works for public datasets but `vsubhashinigoog/single-cell-batch-integration`
  returns **403 / private denied**; ~3GB manual download too slow.
- **scRNA candidates:** do NOT use `scanpy.external.pp.combat`, `sc.tl.combat`, unsafe ComBat, or
  `n_jobs=-1` — these caused every D3B 5-iter invalid. `pbmc3k_conservative_v2` bans them.
- **Never use scanpy/anndata/scib in the ERA env**, and never import scanpy into the ERA controller.
- **R / kBET not installed** — the official 12-metric scIB score is not part of the current line.
- **GIFT-Eval native libs segfault under the agent sandbox** — run GIFT-Eval scoring in a normal terminal.
- **`sandbox.py` is a local executor and INSECURE** — it runs LLM-generated code directly. Toy use only.
- **Gemini flakiness:** intermittent 429/503; mitigated by retries + `SafeGenerator`, but a sustained
  outage yields all-`−inf`. `futs_test.py` needs `absl-py` (not installed).

---

## 12. File / Directory Map

**Controllers / scorers (`implementation/`):**

| File | Env | Role |
|------|-----|------|
| `futs.py`, `llm.py`, `sandbox.py` | ERA | FUTS core, Gemini wrapper, insecure local executor |
| `playground_s3e1.py`, `compare_era_vs_bon.py`, `repeat_era_vs_bon.py` | ERA | California Housing demo + comparisons |
| `playground_breast_cancer.py`, `compare_era_vs_bon_breast_cancer.py`, `repeated_breast_cancer_era_vs_bon.py` | ERA | Breast Cancer benchmark + comparisons |
| `gift_eval_task.py`, `gift_eval_m4_weekly_task.py`, `gift_eval_subset_scout.py` | GIFT-Eval venv | scorers / subset scout |
| `gift_eval_era_search.py`, `gift_eval_compare_era_vs_bon.py` | ERA | GIFT-Eval ERA search + ERA-vs-BoN (subprocess bridge) |
| `scrna_synthetic_task.py`, `scrna_realdata_task.py` | scRNA venv | scRNA scorers |
| `scrna_synthetic_smoke.py`, `scrna_realdata_smoke.py` | scRNA venv | scRNA smoke tests |
| `scrna_era_search.py`, `scrna_compare_era_vs_bon.py`, `scrna_plot_summary.py` | ERA | scRNA ERA search + ERA-vs-BoN + summary figure |

**Outputs (`implementation/saved_runs/`):** `playground_s3e1_iter30/`, `era_vs_bon/`,
`repeated_era_vs_bon/`, `breast_cancer_era_vs_bon/`, `repeated_breast_cancer_era_vs_bon/`,
`gift_eval_c1_setup/` … `gift_eval_c6b_*/`, `scrna_d1_synthetic_smoke/`, `scrna_d2a_synthetic_era_smoke/`,
`scrna_d3a_realdata_smoke/`, `scrna_d3b_pbmc3k_era_iter10_conservative/`,
`scrna_d3c_pbmc3k_era_vs_bon_N10/`, `scrna_summary/`.

**Docs:** `README.md` (root, front-page report — scRNA is §8 Stage 4), `README_UPSTREAM.md`,
`Claude recap.md` (this file). **Gitignored / external:** `gift-eval/`, `scRNA-env/`,
`implementation/data/scanpy_cache/`, `*.h5ad`, Kaggle tokens (`~/.kaggle/kaggle.json`), `__pycache__/`.
