# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plot ERA demo search progress from a futs progress JSON.

Robust to several JSON shapes:
  * list of dicts: [{"iteration": 0, "best_score": -0.73}, ...]
  * list of numbers: [-0.73, -0.58, ...]
  * dict of parallel arrays: {"iteration": [...], "best_score": [...]}
  * dict mapping iteration -> score: {"0": -0.73, "1": -0.58, ...}

Usage:
  python plot_progress.py \
      --input results/futs_progress.json \
      --outdir saved_runs/playground_s3e1_iter30
"""

import argparse
import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

# Candidate key names we may encounter for the score / iteration fields.
_SCORE_KEYS = (
    "best_score", "best_scores", "score", "scores", "value", "values",
    "best", "neg_rmse", "fitness",
)
_ITER_KEYS = (
    "iteration", "iterations", "step", "steps", "i", "index", "iter", "x",
)


def _first_key(keys, candidates):
    for c in candidates:
        if c in keys:
            return c
    return None


def extract_trajectory(data: Any) -> tuple[list, list]:
    """Returns (xs, ys) where ys is the best-so-far score trajectory."""
    xs: list = []
    ys: list = []

    if isinstance(data, dict):
        score_key = _first_key(data, _SCORE_KEYS)
        iter_key = _first_key(data, _ITER_KEYS)
        if score_key is not None and isinstance(data[score_key], list):
            # Parallel-arrays form.
            ys = [float(v) for v in data[score_key]]
            if iter_key is not None and isinstance(data[iter_key], list):
                xs = [float(v) for v in data[iter_key]]
            else:
                xs = list(range(len(ys)))
        else:
            # Mapping iteration -> score.
            items = sorted(
                ((float(k), float(v)) for k, v in data.items()),
                key=lambda t: t[0],
            )
            xs = [k for k, _ in items]
            ys = [v for _, v in items]

    elif isinstance(data, list):
        if data and isinstance(data[0], dict):
            score_key = _first_key(data[0], _SCORE_KEYS)
            iter_key = _first_key(data[0], _ITER_KEYS)
            if score_key is None:
                raise ValueError(
                    f"Could not find a score field in dict keys: {list(data[0])}"
                )
            ys = [float(d[score_key]) for d in data]
            if iter_key is not None:
                xs = [float(d[iter_key]) for d in data]
            else:
                xs = list(range(len(ys)))
        else:
            ys = [float(v) for v in data]
            xs = list(range(len(ys)))
    else:
        raise ValueError(f"Unsupported JSON top-level type: {type(data)}")

    # Derive best-so-far (monotonic non-decreasing). Idempotent if the input
    # already stores a best_score series.
    best: list = []
    cur = float("-inf")
    for v in ys:
        cur = max(cur, v)
        best.append(cur)
    return xs, best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/futs_progress.json")
    p.add_argument("--outdir", default="saved_runs/playground_s3e1_iter30")
    p.add_argument("--name", default="progress",
                   help="Output basename (writes <name>.png and <name>.pdf).")
    p.add_argument("--initial", type=float, default=-0.7339448894714724,
                   help="Initial baseline score reference line.")
    p.add_argument("--best10", type=float, default=-0.5785,
                   help="10-iteration best score reference line.")
    p.add_argument("--best30", type=float, default=-0.5775746202235905,
                   help="30-iteration best score reference line.")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    xs, ys = extract_trajectory(data)
    if not xs:
        raise SystemExit(f"No data points parsed from {args.input}")

    os.makedirs(args.outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Main best-so-far trajectory (step makes the "improve at iter k" visible).
    ax.step(xs, ys, where="post", color="#1f77b4", linewidth=2,
            label="ERA best-so-far", zorder=3)
    ax.scatter(xs, ys, s=18, color="#1f77b4", zorder=4)

    # Horizontal reference lines.
    ax.axhline(args.initial, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"Initial baseline ({args.initial:.4f})")
    ax.axhline(args.best10, color="#2ca02c", linestyle=":", linewidth=1.3,
               label=f"10-iter best (~{args.best10:.4f})")
    ax.axhline(args.best30, color="#9467bd", linestyle="-.", linewidth=1.3,
               label=f"30-iter best ({args.best30:.4f})")

    ax.set_title("ERA Demo Search Progress", fontsize=14, fontweight="bold")
    ax.set_xlabel("Search step / iteration")
    ax.set_ylabel("Best score (negative RMSE, higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    png_path = os.path.join(args.outdir, f"{args.name}.png")
    pdf_path = os.path.join(args.outdir, f"{args.name}.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Parsed {len(xs)} points; "
          f"first={ys[0]:.6f}, last={ys[-1]:.6f}")


if __name__ == "__main__":
    main()
