"""C6A: generalized (dataset-parametric) GIFT-Eval scorer.

Generalizes the hardcoded `gift_eval_m4_weekly_task.py` so the SAME ERA / best-of-N pipeline can
score candidates on any small GIFT-Eval config (e.g. `m4_hourly/H/short`, `hospital/M/short`),
not just `m4_weekly/W/short`.

Same candidate interface and reward as C2:
    def forecast(context, prediction_length, freq, metadata=None) -> 1D np.array[prediction_length]
    reward = -MASE   (invalid candidate -> valid=False, reward=-inf)

Reuses the verified C2 machinery (load_task + FunctionPredictor adapter + the official gluonts
`evaluate_model` path) from `gift_eval_m4_weekly_task`, so scoring is identical except the dataset
is selectable.

------------------------------------------------------------------------------------------
RUN WITH THE GIFT-EVAL VENV:
    /Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_task.py \
        --dataset m4_hourly --freq H --term short --candidates /path/to/candidate.py
------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

# Reuse the verified C2 internals (DRY). These are dataset-agnostic / parametric.
from gift_eval_m4_weekly_task import (
    CandidateError,
    build_function_predictor_cls,
    load_candidate_fn,
    load_task,
    _json_safe,
)


def _base_freq(freq: str) -> str:
    return str(freq).upper().split("-")[0].lstrip("0123456789") or str(freq).upper()


def evaluate_candidate(name, forecast_fn, dataset, season_length, G, config_label):
    """Evaluate one candidate fn on a loaded dataset -> metrics dict (reward = -MASE)."""
    FunctionPredictor = build_function_predictor_cls(G)
    result = {
        "candidate": name,
        "dataset": config_label,
        "valid": False,
        "error": None,
        "runtime_s": None,
        "MASE": None,
        "CRPS": None,
        "RMSE": None,
        "MSE_mean": None,
        "MAE": None,
        "reward": float("-inf"),
    }
    t0 = time.time()
    try:
        predictor = FunctionPredictor(
            prediction_length=dataset.prediction_length,
            freq=dataset.freq,
            forecast_fn=forecast_fn,
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
        result.update(
            valid=True, MASE=mase, CRPS=m("mean_weighted_sum_quantile_loss"),
            RMSE=m("RMSE[mean]"), MSE_mean=m("MSE[mean]"), MAE=m("MAE[0.5]"),
            reward=(-mase if math.isfinite(mase) else float("-inf")),
        )
    except CandidateError as e:
        result["error"] = f"INVALID: {e}"
    except Exception as e:
        result["error"] = f"ERROR: {type(e).__name__}: {e}"
    result["runtime_s"] = round(time.time() - t0, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="m4_weekly",
                        help="GIFT-Eval dataset name (leaf, e.g. m4_weekly, m4_hourly, hospital).")
    parser.add_argument("--freq", default=None,
                        help="Frequency label for the config name, e.g. H/W/M. "
                             "Default: inferred from the dataset.")
    parser.add_argument("--term", default="short", choices=["short", "medium", "long"])
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="Candidate .py file paths (each defines forecast()).")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"Loading task: dataset={args.dataset} term={args.term}")
    dataset, season_length, G = load_task(args.dataset, args.term)
    freq_label = args.freq or _base_freq(dataset.freq)
    config_label = f"{args.dataset}/{freq_label}/{args.term}"
    print(f"config={config_label}  freq={dataset.freq}  pred_len={dataset.prediction_length}  "
          f"windows={dataset.windows}  season_len(metric)={season_length}  "
          f"target_dim={dataset.target_dim}\n")

    cand_paths = [Path(c) for c in (args.candidates or [])]
    results = []
    for path in cand_paths:
        print(f"=== Evaluating {path.name} ===")
        try:
            fn = load_candidate_fn(path)
            r = evaluate_candidate(path.stem, fn, dataset, season_length, G, config_label)
        except Exception as e:
            r = {
                "candidate": path.stem, "dataset": config_label, "valid": False,
                "error": f"LOAD_ERROR: {type(e).__name__}: {e}", "runtime_s": None,
                "MASE": None, "CRPS": None, "RMSE": None, "MSE_mean": None, "MAE": None,
                "reward": float("-inf"),
            }
        results.append(r)
        (log_dir / f"{path.stem}.log").write_text(json.dumps(_json_safe(r), indent=2) + "\n")
        if r["valid"]:
            print(f"  valid=True  MASE={r['MASE']:.6f}  CRPS={r['CRPS']:.6f}  "
                  f"RMSE={r['RMSE']:.4f}  reward={r['reward']:.6f}  ({r['runtime_s']}s)\n")
        else:
            print(f"  valid=False  reward=-inf  error={r['error']}\n")

    payload = {
        "dataset": config_label,
        "prediction_length": dataset.prediction_length,
        "season_length": season_length,
        "reward_definition": "reward = -MASE (lower MASE -> higher reward)",
        "results": [_json_safe(r) for r in results],
    }
    (out_dir / "candidate_results.json").write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = out_dir / "candidate_results.csv"
    fields = ["candidate", "dataset", "valid", "MASE", "CRPS", "RMSE", "MSE_mean", "MAE",
              "reward", "runtime_s", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            if isinstance(row.get("reward"), float) and not math.isfinite(row["reward"]):
                row["reward"] = ""
            w.writerow(row)

    print(f"Wrote {out_dir/'candidate_results.json'}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
