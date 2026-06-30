"""C5: GIFT-Eval subset scouting — find small subsets where naive is NOT dominant.

Evaluates a handful of CHEAP hand-written baselines across several small GIFT-Eval dataset
configs, using the SAME official gluonts scoring path as C2 (reward = -MASE). The goal is to
locate 1-2 subsets where a simple non-naive method beats or approaches naive, so a future ERA
search (C6) has real headroom to improve beyond naive.

NO Gemini, NO ERA search, NO best-of-N, NO foundation models, NO full benchmark. Just cheap
baselines on small datasets.

------------------------------------------------------------------------------------------
RUN WITH THE GIFT-EVAL VENV (this script imports gluonts/gift_eval via the C2 wrapper):
    cd /Users/zhangweikun/era/implementation
    /Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_subset_scout.py
------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

# Reuse the verified C2 machinery (load_task + the FunctionPredictor adapter + metric helper).
from gift_eval_m4_weekly_task import (
    CandidateError,
    QUANTILE_LEVELS,
    build_function_predictor_cls,
    load_task,
)

# Default scout set: small, frequency-diverse LEAF datasets (no subdir), fast to score.
#   m4_weekly  (W) — near random walk; naive is known to dominate (C3/C4 reference)
#   m4_hourly  (H) — strong daily seasonality (period 24); seasonal naive should beat naive
#   hospital   (M) — monthly count data, seasonality 12
#   covid_deaths (D) — daily, strong trend
DEFAULT_SUBSETS = [
    ("m4_weekly", "short"),
    ("m4_hourly", "short"),
    ("hospital", "short"),
    ("covid_deaths", "short"),
]

# Natural seasonal period by base frequency letter (for the seasonal-naive baseline only;
# this is the DATA period, distinct from the metric seasonality used for MASE scaling).
SEASON_BY_FREQ = {"S": 60, "T": 60, "H": 24, "D": 7, "W": 52, "M": 12, "Q": 4, "A": 1, "Y": 1}


def _base_freq(freq: str) -> str:
    return str(freq).upper().split("-")[0].lstrip("0123456789") or str(freq).upper()


# ----------------------------- baselines (candidate interface) -----------------------------
def b_naive(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if context.size == 0:
        return np.zeros(prediction_length)
    return np.full(prediction_length, float(context[-1]))


def b_moving_average(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if context.size == 0:
        return np.zeros(prediction_length)
    w = int(min(8, context.size))
    return np.full(prediction_length, float(np.mean(context[-w:])))


def b_seasonal_naive(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if context.size == 0:
        return np.zeros(prediction_length)
    season = SEASON_BY_FREQ.get(_base_freq(freq), 1)
    if season > 1 and context.size >= season:
        last_season = context[-season:]
        reps = int(np.ceil(prediction_length / season))
        return np.tile(last_season, reps)[:prediction_length].astype(float)
    return np.full(prediction_length, float(context[-1]))  # fallback: naive


def b_damped_trend(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    n = context.size
    if n == 0:
        return np.zeros(prediction_length)
    last = float(context[-1])
    if n < 4:
        return np.full(prediction_length, last)
    slope = float(np.mean(np.diff(context[-4:])))
    scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
    if not math.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))
    phi = 0.6
    steps = np.arange(1, prediction_length + 1)
    out = last + slope * np.cumsum(phi ** steps)
    lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
    hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
    span = hi - lo
    out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
    return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)


def b_ensemble_naive_trend(context, prediction_length, freq, metadata=None):
    """Mostly naive (0.8) + a small damped-trend correction (0.2)."""
    context = np.asarray(context, dtype=float)
    if context.size == 0:
        return np.zeros(prediction_length)
    last = float(context[-1])
    naive = np.full(prediction_length, last)
    trend = b_damped_trend(context, prediction_length, freq, metadata)
    return 0.8 * naive + 0.2 * trend


BASELINES = {
    "naive": b_naive,
    "moving_average": b_moving_average,
    "seasonal_naive": b_seasonal_naive,
    "damped_trend": b_damped_trend,
    "ensemble_naive_trend": b_ensemble_naive_trend,
}


# ----------------------------- scoring -----------------------------
def score_baseline(name, fn, dataset, season_length, G):
    """Score one baseline on one loaded dataset -> metrics dict. Never raises."""
    FunctionPredictor = build_function_predictor_cls(G)
    rec = {"baseline": name, "valid": False, "error": None, "runtime_s": None,
           "MASE": None, "CRPS": None, "RMSE": None, "reward": None}
    t0 = time.time()
    try:
        predictor = FunctionPredictor(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            forecast_fn=fn,
            season_length=season_length,
        )
        res = G["evaluate_model"](
            predictor, test_data=dataset.test_data, metrics=G["metrics"],
            batch_size=512, axis=None, mask_invalid_label=True,
            allow_nan_forecast=False, seasonality=season_length,
        )
        def m(k):
            v = res[k]
            return float(v.iloc[0] if hasattr(v, "iloc") else v[0])
        mase = m("MASE[0.5]")
        rec.update(valid=True, MASE=mase, CRPS=m("mean_weighted_sum_quantile_loss"),
                   RMSE=m("RMSE[mean]"), reward=-mase)
    except CandidateError as e:
        rec["error"] = f"INVALID: {e}"
    except Exception as e:
        rec["error"] = f"ERROR: {type(e).__name__}: {e}"
    rec["runtime_s"] = round(time.time() - t0, 3)
    return rec


def recommend(naive_mase, best_nonnaive_mase, total_runtime_s, any_invalid):
    """Recommendation level for a subset as a future ERA target."""
    if naive_mase is None or best_nonnaive_mase is None:
        return "not recommended (scoring failed / unstable)"
    ratio = best_nonnaive_mase / naive_mase
    if any_invalid:
        note = " (some baselines invalid)"
    else:
        note = ""
    if total_runtime_s is not None and total_runtime_s > 120:
        return f"not recommended (too slow: {total_runtime_s:.0f}s){note}"
    if ratio < 0.95:
        return f"strong candidate for ERA (a simple method clearly beats naive, ratio={ratio:.3f}){note}"
    if ratio < 1.0:
        return f"possible candidate (a non-naive method slightly beats naive, ratio={ratio:.3f}){note}"
    return f"not recommended (naive dominates, no non-naive method beats it, ratio={ratio:.3f}){note}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="dataset names (leaf or name/freq). Default: the C5 scout set.")
    parser.add_argument("--term", default="short", choices=["short", "medium", "long"])
    parser.add_argument("--out_dir",
                        default=str(Path(__file__).resolve().parent
                                    / "saved_runs" / "gift_eval_c5_subset_scout"))
    args = parser.parse_args()

    if args.datasets:
        subsets = [(d, args.term) for d in args.datasets]
    else:
        subsets = DEFAULT_SUBSETS

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"GIFT_EVAL data = {os.environ.get('GIFT_EVAL')}")
    print(f"Scouting {len(subsets)} subsets with {len(BASELINES)} baselines each.\n")

    rows = []          # flat per (subset, baseline) rows
    subset_summaries = []

    for ds_name, term in subsets:
        print(f"=== {ds_name} (term={term}) ===")
        try:
            dataset, season_length, G = load_task(ds_name, term)
        except Exception as e:
            print(f"  [!] could not load: {e}\n")
            subset_summaries.append({
                "dataset": ds_name, "term": term, "freq": None, "prediction_length": None,
                "n_series": None, "windows": None, "naive_MASE": None,
                "best_baseline": None, "best_nonnaive_MASE": None,
                "improvement_over_naive": None, "ratio_best_nonnaive_to_naive": None,
                "recommendation": f"not recommended (load failed: {type(e).__name__}: {e})",
            })
            continue

        try:
            n_series = int(len(dataset.hf_dataset))
        except Exception:
            n_series = None
        config = f"{ds_name}/{dataset.freq}/{term}"
        print(f"  freq={dataset.freq} pred_len={dataset.prediction_length} "
              f"windows={dataset.windows} season_len(metric)={season_length} n_series={n_series}")

        per_baseline = {}
        subset_runtime = 0.0
        any_invalid = False
        for bname, bfn in BASELINES.items():
            rec = score_baseline(bname, bfn, dataset, season_length, G)
            subset_runtime += (rec["runtime_s"] or 0.0)
            any_invalid = any_invalid or (not rec["valid"])
            per_baseline[bname] = rec
            row = {"dataset": ds_name, "config": config, "freq": dataset.freq,
                   "prediction_length": dataset.prediction_length, "n_series": n_series,
                   "windows": dataset.windows, **rec}
            rows.append(row)
            status = (f"MASE={rec['MASE']:.4f}" if rec["valid"]
                      else f"INVALID ({str(rec['error'])[:40]})")
            print(f"    {bname:22s} {status}  ({rec['runtime_s']}s)")

        naive_mase = per_baseline["naive"]["MASE"] if per_baseline["naive"]["valid"] else None
        nonnaive_valid = {k: v["MASE"] for k, v in per_baseline.items()
                          if k != "naive" and v["valid"] and v["MASE"] is not None}
        best_nonnaive = min(nonnaive_valid, key=nonnaive_valid.get) if nonnaive_valid else None
        best_nonnaive_mase = nonnaive_valid[best_nonnaive] if best_nonnaive else None
        all_valid = {k: v["MASE"] for k, v in per_baseline.items()
                     if v["valid"] and v["MASE"] is not None}
        best_baseline = min(all_valid, key=all_valid.get) if all_valid else None
        improvement = (naive_mase - best_nonnaive_mase) if (
            naive_mase is not None and best_nonnaive_mase is not None) else None
        ratio = (best_nonnaive_mase / naive_mase) if (
            naive_mase and best_nonnaive_mase is not None) else None
        rec_level = recommend(naive_mase, best_nonnaive_mase, subset_runtime, any_invalid)

        subset_summaries.append({
            "dataset": ds_name, "term": term, "config": config, "freq": dataset.freq,
            "prediction_length": dataset.prediction_length, "n_series": n_series,
            "windows": dataset.windows, "naive_MASE": naive_mase,
            "best_baseline": best_baseline, "best_nonnaive_MASE": best_nonnaive_mase,
            "best_nonnaive_baseline": best_nonnaive,
            "improvement_over_naive": improvement,
            "ratio_best_nonnaive_to_naive": ratio,
            "subset_runtime_s": round(subset_runtime, 3),
            "recommendation": rec_level,
        })
        print(f"  -> naive={naive_mase}  best_nonnaive={best_nonnaive} ({best_nonnaive_mase})  "
              f"ratio={ratio}\n  -> {rec_level}\n")

    # --- write CSV (per baseline) ---
    csv_path = out_dir / "scout_results.csv"
    fields = ["dataset", "config", "freq", "prediction_length", "n_series", "windows",
              "baseline", "valid", "MASE", "CRPS", "RMSE", "reward", "runtime_s", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --- write JSON (per baseline rows + per subset summaries) ---
    json_path = out_dir / "scout_results.json"
    json_path.write_text(json.dumps({
        "term": args.term,
        "baselines": list(BASELINES.keys()),
        "rows": rows,
        "subset_summaries": subset_summaries,
    }, indent=2, default=lambda o: None) + "\n")

    # --- plots ---
    try:
        _make_plots(out_dir, rows, subset_summaries)
    except Exception as e:
        print(f"[!] plotting failed (non-fatal): {e}")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print("\n================ SCOUT RECOMMENDATIONS ================")
    for s in subset_summaries:
        print(f"{s['dataset']:14s} {str(s['recommendation'])}")
    print("======================================================")


def _make_plots(out_dir, rows, subset_summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [s["dataset"] for s in subset_summaries if s.get("naive_MASE") is not None]
    baselines = list(BASELINES.keys())
    # MASE-by-subset grouped bars
    valid_rows = {(r["dataset"], r["baseline"]): r["MASE"]
                  for r in rows if r["valid"] and r["MASE"] is not None}
    if datasets:
        x = np.arange(len(datasets))
        width = 0.15
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        for i, b in enumerate(baselines):
            vals = [valid_rows.get((d, b), np.nan) for d in datasets]
            ax.bar(x + (i - len(baselines) / 2) * width + width / 2, vals, width, label=b)
        ax.set_xticks(x); ax.set_xticklabels(datasets)
        ax.set_ylabel("MASE (lower is better)")
        ax.set_title("C5: baseline MASE by GIFT-Eval subset", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "baseline_mase_by_subset.png")
        plt.close(fig)

        # relative-to-naive ratio (per baseline per subset); <1 means beats naive
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        for i, b in enumerate(baselines):
            ratios = []
            for d in datasets:
                naive = valid_rows.get((d, "naive"))
                v = valid_rows.get((d, b))
                ratios.append((v / naive) if (naive and v is not None) else np.nan)
            ax.bar(x + (i - len(baselines) / 2) * width + width / 2, ratios, width, label=b)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="naive (=1.0)")
        ax.set_xticks(x); ax.set_xticklabels(datasets)
        ax.set_ylabel("MASE / naive_MASE  (<1 beats naive)")
        ax.set_title("C5: baseline MASE relative to naive (lower = better than naive)",
                     fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "relative_to_naive.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
