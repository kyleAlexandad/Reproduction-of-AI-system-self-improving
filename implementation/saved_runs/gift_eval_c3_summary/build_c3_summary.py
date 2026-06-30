"""Build the C3 final summary: aggregate the 5/10/20-iteration GIFT-Eval ERA runs.

Reads each run's results.json, writes c3_summary_table.csv, and plots best-generated-so-far MASE
(excluding the naive seed) vs iteration for all runs, with the naive baseline as a reference line.

Run with the ERA env python (only needs numpy + matplotlib):
    python build_c3_summary.py
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE.parent  # saved_runs/

# (label, run_dir, prompt_version_expected)
RUNS = [
    ("smoke (5, baseline)", "gift_eval_c3_era_smoke"),
    ("iter10 (conservative)", "gift_eval_c3_era_iter10_conservative"),
    ("iter20 (conservative)", "gift_eval_c3_era_iter20_conservative"),
]


def load_run(run_dir):
    data = json.loads((RUNS_DIR / run_dir / "results.json").read_text())
    recs = data["records"]
    seed = recs[0]
    generated = recs[1:]
    naive_mase = seed["MASE"]
    valid_gen = [r for r in generated if r["valid"] and r["MASE"] is not None]
    invalid_gen = [r for r in generated if not r["valid"]]
    best_gen = min((r["MASE"] for r in valid_gen), default=None)
    # best-generated-so-far series (skip invalid; carry previous best)
    best_so_far = []
    cur = math.inf
    for r in generated:
        if r["valid"] and r["MASE"] is not None and r["MASE"] < cur:
            cur = r["MASE"]
        best_so_far.append(cur if math.isfinite(cur) else None)
    dist = (best_gen - naive_mase) if best_gen is not None else None
    return {
        "data": data,
        "naive_mase": naive_mase,
        "n_generated": len(generated),
        "valid": len(valid_gen),
        "invalid": len(invalid_gen),
        "best_gen_mase": best_gen,
        "best_incl_seed": data["best_MASE"],
        "dist_to_naive": dist,
        "beat_naive": (best_gen is not None and best_gen < naive_mase),
        "best_so_far": best_so_far,
        "prompt_version": data.get("prompt_version"),
        "iterations": data.get("iterations_requested"),
    }


def main():
    runs = {label: load_run(d) for label, d in RUNS}

    # --- CSV ---
    csv_path = HERE / "c3_summary_table.csv"
    fields = [
        "run", "prompt_version", "iterations", "n_generated", "valid", "invalid",
        "naive_MASE", "best_MASE_incl_seed", "best_generated_MASE_excl_seed",
        "distance_to_naive_excl_seed", "beat_naive",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for label, _ in RUNS:
            r = runs[label]
            w.writerow([
                label, r["prompt_version"], r["iterations"], r["n_generated"],
                r["valid"], r["invalid"], r["naive_mase"], r["best_incl_seed"],
                r["best_gen_mase"], r["dist_to_naive"], r["beat_naive"],
            ])
    print(f"Wrote {csv_path}")

    # --- Plot: best-generated-so-far MASE vs iteration ---
    naive = runs[RUNS[1][0]]["naive_mase"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colors = {"smoke (5, baseline)": "#7f7f7f",
              "iter10 (conservative)": "#1f77b4",
              "iter20 (conservative)": "#2ca02c"}
    markers = {"smoke (5, baseline)": "x",
               "iter10 (conservative)": "o",
               "iter20 (conservative)": "s"}
    for label, _ in RUNS:
        ys = runs[label]["best_so_far"]
        xs = list(range(1, len(ys) + 1))
        ax.step(xs, ys, where="post", color=colors[label], marker=markers[label],
                markersize=4, linewidth=1.8, label=label)
    ax.axhline(naive, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"naive baseline (MASE {naive:.5f})")
    ax.set_title("C3: best generated-candidate MASE so far (excl. naive seed)\n"
                 "GIFT-Eval m4_weekly/W/short  (lower is better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("ERA candidate index (generated, excluding seed)")
    ax.set_ylabel("Best-so-far MASE (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    # Zoom so the near-naive convergence is visible.
    ax.set_ylim(naive - 0.01, 2.83)
    fig.tight_layout()
    png = HERE / "c3_best_so_far_mase.png"
    pdf = HERE / "c3_best_so_far_mase.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

    # --- console summary ---
    print("\n=== C3 SUMMARY ===")
    for label, _ in RUNS:
        r = runs[label]
        print(f"{label:26s} naive={r['naive_mase']:.6f}  best_gen={r['best_gen_mase']:.6f}  "
              f"dist={r['dist_to_naive']:.2e}  valid/invalid={r['valid']}/{r['invalid']}  "
              f"beat_naive={r['beat_naive']}")


if __name__ == "__main__":
    main()
