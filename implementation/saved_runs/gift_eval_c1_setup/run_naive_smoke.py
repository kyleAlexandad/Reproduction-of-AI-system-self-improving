"""C1 smoke test: run the official GIFT-Eval Naive baseline on ONE small dataset.

This mirrors the official `notebooks/naive.ipynb` wrapper (StatsForecastPredictor /
NaivePredictor) but restricts the run to a single tiny dataset (`m4_weekly`) so we can
verify the evaluation pipeline end-to-end on CPU without downloading the full benchmark.

Run from the gift-eval repo root with the venv active and GIFT_EVAL set in .env:
    python run_naive_smoke.py
"""

import argparse
import csv
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Type

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gluonts.core.component import validated
from gluonts.dataset import Dataset as GluonTSDataset
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
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.time_feature import get_seasonality
from gluonts.transform.feature import LastValueImputation, MissingValueImputation
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive

from gift_eval.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class ModelConfig:
    quantile_levels: Optional[List[float]] = None
    forecast_keys: List[str] = field(init=False)
    statsforecast_keys: List[str] = field(init=False)
    intervals: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return
        intervals = set()
        for quantile_level in self.quantile_levels:
            interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")
        self.intervals = sorted(intervals)


class StatsForecastPredictor(RepresentablePredictor):
    """Wraps a `statsforecast` model to the gluonts predictor interface."""

    ModelType: Type

    @validated()
    def __init__(
        self,
        prediction_length: int,
        season_length: int,
        freq: str,
        quantile_levels: Optional[List[float]] = None,
        imputation_method: MissingValueImputation = LastValueImputation(),
        max_length: Optional[int] = None,
        batch_size: int = 1,
        parallel: bool = False,
        **model_params,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        if "season_length" in inspect.signature(self.ModelType.__init__).parameters:
            model_params["season_length"] = season_length
        self.freq = freq
        self.model = StatsForecast(
            models=[self.ModelType(**model_params)],
            freq=freq,
            fallback_model=SeasonalNaive(season_length=season_length),
            n_jobs=-1 if parallel else 1,
        )
        self.fallback_model = StatsForecast(
            models=[SeasonalNaive(season_length=season_length)],
            freq=freq,
            n_jobs=-1 if parallel else 1,
        )
        self.config = ModelConfig(quantile_levels=quantile_levels)
        self.imputation_method = imputation_method
        self.batch_size = batch_size
        self.max_length = max_length

    def predict(self, dataset: GluonTSDataset, **kwargs) -> Iterator[Forecast]:
        batch = {}
        for idx, entry in enumerate(dataset):
            assert entry["target"].ndim == 1, "only for univariate time series"
            assert len(entry["target"]) >= 1, "series must have >=1 point"
            if self.max_length is not None:
                entry["start"] += len(entry["target"][: -self.max_length])
                entry["target"] = entry["target"][-self.max_length :]
            target = np.asarray(entry["target"], np.float32)
            if np.isnan(target).any():
                target = self.imputation_method(target.copy())
            unique_id = (
                f"{entry['item_id']}_{str(forecast_start(entry))}_{str(len(batch))}"
            )
            start = entry["start"]
            batch[unique_id] = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": pd.date_range(
                        start=start.to_timestamp(),
                        periods=len(target),
                        freq=start.freq,
                    ).to_numpy(),
                    "y": target,
                }
            )
            if len(batch) == self.batch_size:
                results = self.sf_predict(pd.concat(batch.values()))
                yield from self.yield_forecast(batch.keys(), results)
                batch = {}
        if len(batch) > 0:
            results = self.sf_predict(pd.concat(batch.values()))
            yield from self.yield_forecast(batch.keys(), results)

    def sf_predict(self, Y_df: pd.DataFrame) -> pd.DataFrame:
        kwargs = {}
        if self.config.intervals is not None:
            kwargs["level"] = self.config.intervals
        results = self.model.forecast(df=Y_df, h=self.prediction_length, **kwargs)
        row_nan = results.isnull().values.any(axis=-1)
        if row_nan.any():
            nan_ids = results[row_nan].index.values
            nan_df = Y_df[Y_df["unique_id"].isin(nan_ids)]
            fallback_results = self.fallback_model.forecast(
                df=nan_df, h=self.prediction_length, **kwargs
            )
            results = pd.concat(
                [results[~results.index.isin(nan_ids)], fallback_results]
            )
        return results

    def yield_forecast(
        self, item_ids, results: pd.DataFrame
    ) -> Iterator[QuantileForecast]:
        results.set_index("unique_id", inplace=True)
        for idx in item_ids:
            prediction = results.loc[idx]
            forecast_arrays = []
            model_name = self.ModelType.__name__
            for key in self.config.statsforecast_keys:
                if key == "mean":
                    forecast_arrays.append(prediction.loc[:, model_name].to_numpy())
                else:
                    forecast_arrays.append(
                        prediction.loc[:, f"{model_name}-{key}"].to_numpy()
                    )
            yield QuantileForecast(
                forecast_arrays=np.stack(forecast_arrays, axis=0),
                forecast_keys=self.config.forecast_keys,
                start_date=prediction.ds.iloc[0].to_period(freq=self.freq),
                item_id=idx,
            )


class NaivePredictor(StatsForecastPredictor):
    ModelType = Naive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="m4_weekly")
    parser.add_argument("--term", default="short", choices=["short", "medium", "long"])
    parser.add_argument(
        "--out-dir",
        default=str(
            Path.home()
            / "era/implementation/saved_runs/gift_eval_c1_setup"
        ),
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    print(f"GIFT_EVAL = {os.getenv('GIFT_EVAL')}")

    metrics = [
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
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]

    dataset_properties_map = json.load(
        open(REPO_ROOT / "notebooks" / "dataset_properties.json")
    )
    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }

    ds_name = args.dataset
    term = args.term
    if "/" in ds_name:
        ds_key = pretty_names.get(ds_name.split("/")[0].lower(), ds_name.split("/")[0].lower())
        ds_freq = ds_name.split("/")[1]
    else:
        ds_key = pretty_names.get(ds_name.lower(), ds_name.lower())
        ds_freq = dataset_properties_map[ds_key]["frequency"]
    ds_config = f"{ds_key}/{ds_freq}/{term}"

    print(f"Loading dataset: {ds_name} (term={term}) -> config {ds_config}")
    to_univariate = (
        False
        if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
        else True
    )
    dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
    season_length = get_seasonality(dataset.freq)
    print(
        f"freq={dataset.freq}  prediction_length={dataset.prediction_length}  "
        f"windows={dataset.windows}  season_length={season_length}  "
        f"target_dim={dataset.target_dim}"
    )

    predictor = NaivePredictor(
        dataset.prediction_length,
        season_length=season_length,
        freq=dataset.freq,
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        batch_size=512,
    )

    t0 = time.time()
    res = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=metrics,
        batch_size=512,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    elapsed = time.time() - t0
    print(f"Evaluation finished in {elapsed:.2f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_results.csv"
    header = [
        "dataset",
        "model",
        "eval_metrics/MSE[mean]",
        "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]",
        "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS",
        "eval_metrics/RMSE[mean]",
        "eval_metrics/NRMSE[mean]",
        "eval_metrics/ND[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
        "domain",
        "num_variates",
    ]
    def m(key):
        v = res[key]
        return v.iloc[0] if hasattr(v, "iloc") else v[0]

    row = [
        ds_config,
        "naive",
        m("MSE[mean]"),
        m("MSE[0.5]"),
        m("MAE[0.5]"),
        m("MASE[0.5]"),
        m("MAPE[0.5]"),
        m("sMAPE[0.5]"),
        m("MSIS"),
        m("RMSE[mean]"),
        m("NRMSE[mean]"),
        m("ND[0.5]"),
        m("mean_weighted_sum_quantile_loss"),
        dataset_properties_map[ds_key]["domain"],
        dataset_properties_map[ds_key]["num_variates"],
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    print(f"\nWrote results to {csv_path}")
    print("\n=== METRICS ===")
    for k, v in zip(header[2:13], row[2:13]):
        print(f"{k:48s} {v}")


if __name__ == "__main__":
    main()
