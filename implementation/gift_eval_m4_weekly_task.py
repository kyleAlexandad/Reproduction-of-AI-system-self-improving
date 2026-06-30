"""C2: ERA-ready scorable task wrapper for GIFT-Eval `m4_weekly/W/short`.

This module turns a *simple candidate forecasting function* into a single scalar reward,
computed with the **same** official GIFT-Eval / gluonts evaluation pipeline used in C1.

------------------------------------------------------------------------------------------
CANDIDATE INTERFACE (what an LLM will generate later in C3)
------------------------------------------------------------------------------------------
A candidate is a Python file that defines a function:

    def forecast(context, prediction_length, freq, metadata=None):
        '''
        context           : 1D numpy array of the historical target values (no future labels).
        prediction_length : int, forecast horizon (13 for m4_weekly/W/short).
        freq              : str frequency, e.g. "W-SUN".
        metadata          : optional dict with extra hints
                            (item_id, season_length, context_length, ...).

        Returns: 1D numpy array of length `prediction_length` with POINT forecasts.
        '''

The candidate only ever sees past context (gluonts strips the label window before calling
the predictor), so there is no data leakage by construction.

A point-only forecast is adapted to the probabilistic GIFT-Eval metrics by a
``FunctionPredictor`` that emits a degenerate ``QuantileForecast`` (all quantiles = the point
forecast, plus a "mean" key). Point metrics (MASE, RMSE, MSE[mean]) are therefore exact;
quantile metrics (CRPS=mean_weighted_sum_quantile_loss, MSIS) are computed against this
degenerate distribution -- see README_C2.md for why the naive CRPS differs from C1.

------------------------------------------------------------------------------------------
IMPORTANT: run this with the GIFT-Eval venv, NOT the ERA env:
    /Users/zhangweikun/era/gift-eval/.venv/bin/python gift_eval_m4_weekly_task.py
------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np

# --- Locations (defaults match the C1 setup) --------------------------------------------
GIFT_EVAL_REPO = Path(os.getenv("GIFT_EVAL_REPO", "/Users/zhangweikun/era/gift-eval"))
GIFT_EVAL_DATA = Path(os.getenv("GIFT_EVAL", str(GIFT_EVAL_REPO / "data")))
DATASET_PROPERTIES = GIFT_EVAL_REPO / "notebooks" / "dataset_properties.json"

# Make the dataset path visible to gift_eval.data.Dataset (it reads the GIFT_EVAL env var).
os.environ.setdefault("GIFT_EVAL", str(GIFT_EVAL_DATA))

DATASET_NAME = "m4_weekly"
TERM = "short"

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class CandidateError(Exception):
    """Raised when a candidate violates the interface (bad shape, NaN/inf, crash)."""


# --- gluonts imports (only available in the GIFT-Eval venv) -----------------------------
def _import_gluonts():
    from gluonts.dataset.util import forecast_start
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )
    from gluonts.model import evaluate_model
    from gluonts.model.forecast import QuantileForecast
    from gluonts.model.predictor import RepresentablePredictor
    from gluonts.time_feature import get_seasonality

    return dict(
        forecast_start=forecast_start,
        evaluate_model=evaluate_model,
        QuantileForecast=QuantileForecast,
        RepresentablePredictor=RepresentablePredictor,
        get_seasonality=get_seasonality,
        metrics=[
            MSE(forecast_type="mean"),
            MSE(forecast_type=0.5),
            MAE(),
            MASE(),
            MAPE(),
            SMAPE(),
            MSIS(),
            RMSE(),
            NRMSE(),
            ND(),
            MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
        ],
    )


def build_function_predictor_cls(G):
    RepresentablePredictor = G["RepresentablePredictor"]
    QuantileForecast = G["QuantileForecast"]
    forecast_start = G["forecast_start"]

    class FunctionPredictor(RepresentablePredictor):
        """Adapts a simple ``forecast(context, pl, freq, metadata)`` fn to gluonts."""

        def __init__(
            self,
            prediction_length: int,
            freq: str,
            forecast_fn: Callable,
            season_length: int = 1,
        ):
            super().__init__(prediction_length=prediction_length)
            self.freq = freq
            self.forecast_fn = forecast_fn
            self.season_length = season_length
            self.forecast_keys = ["mean"] + [str(q) for q in QUANTILE_LEVELS]

        def predict(self, dataset, **kwargs) -> Iterator:
            pl = self.prediction_length
            for entry in dataset:
                context = np.asarray(entry["target"], dtype=np.float64)
                if context.ndim != 1:
                    raise CandidateError(
                        f"context must be 1D, got shape {context.shape}"
                    )
                metadata = {
                    "item_id": entry.get("item_id"),
                    "season_length": self.season_length,
                    "context_length": int(context.shape[0]),
                    "prediction_length": pl,
                    "freq": self.freq,
                }
                try:
                    raw = self.forecast_fn(context.copy(), pl, self.freq, metadata)
                except CandidateError:
                    raise
                except Exception as e:  # candidate crashed
                    raise CandidateError(
                        f"candidate raised {type(e).__name__}: {e}"
                    ) from e

                point = np.asarray(raw, dtype=np.float64).reshape(-1)
                if point.shape[0] != pl:
                    raise CandidateError(
                        f"expected forecast of length {pl}, got {point.shape[0]}"
                    )
                if not np.all(np.isfinite(point)):
                    raise CandidateError("forecast contains NaN or inf")

                # Degenerate quantile forecast: every quantile == the point forecast.
                forecast_arrays = np.stack([point] * len(self.forecast_keys), axis=0)
                yield QuantileForecast(
                    forecast_arrays=forecast_arrays,
                    forecast_keys=self.forecast_keys,
                    start_date=forecast_start(entry),
                    item_id=entry.get("item_id"),
                )

    return FunctionPredictor


# --- Task loading -----------------------------------------------------------------------
def load_task(dataset_name: str = DATASET_NAME, term: str = TERM):
    """Load the GIFT-Eval Dataset + metadata (same config as C1)."""
    from gift_eval.data import Dataset

    G = _import_gluonts()
    to_univariate = (
        False
        if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim == 1
        else True
    )
    dataset = Dataset(name=dataset_name, term=term, to_univariate=to_univariate)
    season_length = G["get_seasonality"](dataset.freq)
    return dataset, season_length, G


def _metric(res, key):
    v = res[key]
    return float(v.iloc[0] if hasattr(v, "iloc") else v[0])


def evaluate_candidate(
    name: str,
    forecast_fn: Callable,
    dataset=None,
    season_length: Optional[int] = None,
    G=None,
) -> dict:
    """Evaluate one candidate forecast fn -> dict of metrics + reward.

    Reward = -MASE (GIFT-Eval is lower-is-better; ERA assumes higher-is-better).
    Invalid candidates get valid=False, reward=-inf and an error message.
    """
    if dataset is None or season_length is None or G is None:
        dataset, season_length, G = load_task()

    FunctionPredictor = build_function_predictor_cls(G)
    result = {
        "candidate": name,
        "dataset": f"{DATASET_NAME}/{dataset.freq}/{TERM}",
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
            predictor,
            test_data=dataset.test_data,
            metrics=G["metrics"],
            batch_size=512,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )
        mase = _metric(res, "MASE[0.5]")
        crps = _metric(res, "mean_weighted_sum_quantile_loss")
        rmse = _metric(res, "RMSE[mean]")
        result.update(
            valid=True,
            MASE=mase,
            CRPS=crps,
            RMSE=rmse,
            MSE_mean=_metric(res, "MSE[mean]"),
            MAE=_metric(res, "MAE[0.5]"),
            reward=(-mase if math.isfinite(mase) else float("-inf")),
        )
    except CandidateError as e:
        result["error"] = f"INVALID: {e}"
    except Exception as e:  # pipeline-level failure
        result["error"] = f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"
    result["runtime_s"] = round(time.time() - t0, 3)
    return result


def load_candidate_fn(path: Path) -> Callable:
    """Dynamically import a candidate file and return its `forecast` function."""
    spec = importlib.util.spec_from_file_location(f"candidate_{path.stem}", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "forecast"):
        raise CandidateError(f"{path.name} does not define a `forecast` function")
    return module.forecast


def _json_safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


DEFAULT_CANDIDATE_DIR = Path(
    "/Users/zhangweikun/era/implementation/saved_runs/gift_eval_c2_task_wrapper"
)
DEFAULT_CANDIDATES = [
    "candidate_naive.py",
    "candidate_seasonal_naive.py",
    "candidate_moving_average.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Candidate .py file paths. Default: the 3 baseline candidates in the C2 folder.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_CANDIDATE_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    if args.candidates:
        cand_paths = [Path(c) for c in args.candidates]
    else:
        cand_paths = [DEFAULT_CANDIDATE_DIR / c for c in DEFAULT_CANDIDATES]

    print(f"GIFT_EVAL data = {os.environ.get('GIFT_EVAL')}")
    print("Loading task m4_weekly/short ...")
    dataset, season_length, G = load_task()
    print(
        f"freq={dataset.freq} prediction_length={dataset.prediction_length} "
        f"windows={dataset.windows} season_length={season_length} "
        f"target_dim={dataset.target_dim}\n"
    )

    results = []
    for path in cand_paths:
        print(f"=== Evaluating {path.name} ===")
        try:
            fn = load_candidate_fn(path)
            r = evaluate_candidate(path.stem, fn, dataset, season_length, G)
        except Exception as e:
            r = {
                "candidate": path.stem,
                "dataset": f"{DATASET_NAME}/{dataset.freq}/{TERM}",
                "valid": False,
                "error": f"LOAD_ERROR: {type(e).__name__}: {e}",
                "runtime_s": None,
                "MASE": None,
                "CRPS": None,
                "RMSE": None,
                "MSE_mean": None,
                "MAE": None,
                "reward": float("-inf"),
            }
        results.append(r)
        (log_dir / f"{path.stem}.log").write_text(
            json.dumps(_json_safe(r), indent=2) + "\n"
        )
        if r["valid"]:
            print(
                f"  valid=True  MASE={r['MASE']:.6f}  CRPS={r['CRPS']:.6f}  "
                f"RMSE={r['RMSE']:.4f}  reward={r['reward']:.6f}  "
                f"({r['runtime_s']}s)\n"
            )
        else:
            print(f"  valid=False  reward=-inf  error={r['error']}\n")

    # Write JSON
    json_path = out_dir / "candidate_results.json"
    payload = {
        "dataset": f"{DATASET_NAME}/{dataset.freq}/{TERM}",
        "prediction_length": dataset.prediction_length,
        "season_length": season_length,
        "reward_definition": "reward = -MASE (lower MASE is better -> higher reward)",
        "results": [_json_safe(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    # Write CSV
    csv_path = out_dir / "candidate_results.csv"
    fields = [
        "candidate",
        "dataset",
        "valid",
        "MASE",
        "CRPS",
        "RMSE",
        "MSE_mean",
        "MAE",
        "reward",
        "runtime_s",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            if isinstance(row.get("reward"), float) and not math.isfinite(row["reward"]):
                row["reward"] = ""
            writer.writerow(row)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Per-candidate logs in {log_dir}")


if __name__ == "__main__":
    main()
