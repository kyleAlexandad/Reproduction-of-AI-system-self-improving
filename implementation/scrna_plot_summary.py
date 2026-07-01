"""Consolidated scRNA figure: ERA batch-integration reward across stages D1->D3C.

Reads the existing run `results.json` files under saved_runs/ (NO Gemini, NO scanpy) and draws ONE
two-panel figure to saved_runs/scrna_summary/. The scRNA arc spans two reward scales, so each panel
has its own axis:

  * Panel A - SYNTHETIC track (D1 baselines + D2A ERA searches): reward ~1.07-1.30.
  * Panel B - PBMC3k REAL bridge (D3A baselines + D3B ERA + D3C ERA-vs-BoN): reward ~1.03-1.07.

Reward = reduced Python-only proxy (bio_score + batch_mixing_score), HIGHER is better. This is NOT
the official 12-metric scIB score (deferred; needs R/kBET + the Kaggle dataset).

Run (ERA env): /opt/homebrew/bin/python3.12 scrna_plot_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(__file__).resolve().parent / "saved_runs"
OUT = RUNS / "scrna_summary"

BLUE = "#1f77b4"     # ERA / tree search
ORANGE = "#ff7f0e"   # best-of-N
GREEN = "#2ca02c"    # batch-centered reference
GREY = "#7f7f7f"     # seed / plain-PCA baseline


def _load(name):
    return json.loads((RUNS / name / "results.json").read_text())


def _smoke_scores(name):
    """Return {candidate_name: score} for a D1/D3A smoke results.json."""
    d = _load(name)
    res = d["results"] if isinstance(d, dict) and "results" in d else d
    out = {}
    for r in res:
        out[r.get("candidate") or r.get("name")] = r.get("score")
    return out


def _bar(ax, xs, vals, colors, labels, annos):
    bars = ax.bar(xs, vals, color=colors, width=0.62, edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    for b, v, a in zip(bars, vals, annos):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}\n{a}", ha="center", va="bottom",
                fontsize=7.5, zorder=4)
    return bars


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- gather numbers from disk ----
    d1 = _smoke_scores("scrna_d1_synthetic_smoke")          # pca, batch_centered_pca
    d3a = _smoke_scores("scrna_d3a_realdata_smoke")         # pca, batch_centered_pca
    d2a_s = _load("scrna_d2a_synthetic_era_smoke")
    d2a_10 = _load("scrna_d2a_synthetic_era_iter10")
    d3b_s = _load("scrna_d3b_pbmc3k_era_smoke")
    d3b_10 = _load("scrna_d3b_pbmc3k_era_iter10_conservative")
    d3c = _load("scrna_d3c_pbmc3k_era_vs_bon_N10")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150)

    # ================= Panel A: SYNTHETIC =================
    synth_seed = d2a_s["initial_reward"]                     # log-norm PCA seed (~1.0699)
    synth_ref = d1["batch_centered_pca"]                     # ~1.3024 batch-centered reference
    xs = [0, 1]
    vals = [d2a_s["best_reward"], d2a_10["best_reward"]]
    annos = [f"{d2a_s['num_valid']}/{d2a_s['num_invalid']} valid",
             f"{d2a_10['num_valid']}/{d2a_10['num_invalid']} valid"]
    _bar(axA, xs, vals, [BLUE, BLUE], ["D2A ERA\n5-iter", "D2A ERA\n10-iter"], annos)
    axA.axhline(synth_seed, color=GREY, ls=":", lw=1.4, zorder=2,
                label=f"log-norm PCA seed ({synth_seed:.4f})")
    axA.axhline(synth_ref, color=GREEN, ls="--", lw=1.4, zorder=2,
                label=f"batch-centered ref ({synth_ref:.4f})")
    axA.set_title("A. Synthetic track (D1 baselines + D2A ERA)", fontsize=11, fontweight="bold")
    axA.set_ylabel("best reduced-proxy reward (higher is better)")
    axA.set_ylim(1.00, 1.36)
    axA.grid(axis="y", alpha=0.3)
    axA.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ================= Panel B: PBMC3k real bridge =================
    pb_seed = d3a["pca"]                                     # 1.0275 PCA-20 seed
    pb_ref = d3a["batch_centered_pca"]                       # ~1.0701 batch-centered reference
    era = d3c["ERA"]; bon = d3c["best_of_N"]
    xs = [0, 1, 2, 3]
    vals = [d3b_s["best_reward"], d3b_10["best_reward"],
            era["best_generated_reward_excl_seed"], bon["best_generated_reward_excl_seed"]]
    annos = [f"{d3b_s['num_valid']}/{d3b_s['num_invalid']} valid",
             f"{d3b_10['num_valid']}/{d3b_10['num_invalid']} valid",
             f"{era['valid']}/{era['invalid']} valid",
             f"{bon['valid']}/{bon['invalid']} valid"]
    _bar(axB, xs, vals, [BLUE, BLUE, BLUE, ORANGE],
         ["D3B\n5-iter", "D3B\n10-iter", "D3C\nERA", "D3C\nbest-of-N"], annos)
    axB.axhline(pb_seed, color=GREY, ls=":", lw=1.4, zorder=2,
                label=f"PCA-20 seed ({pb_seed:.4f})")
    axB.axhline(pb_ref, color=GREEN, ls="--", lw=1.4, zorder=2,
                label=f"batch-centered ref ({pb_ref:.4f})")
    axB.set_title("B. PBMC3k real bridge (D3A baselines + D3B ERA + D3C ERA-vs-BoN)",
                  fontsize=11, fontweight="bold")
    axB.set_ylim(1.020, 1.095)
    axB.grid(axis="y", alpha=0.3)
    axB.legend(loc="upper left", fontsize=8, framealpha=0.95)

    fig.suptitle("ERA on scRNA-seq batch integration — reduced-proxy reward across stages "
                 "(D1→D3C)\nreduced Python-only proxy (bio + batch-mixing), HIGHER is better; "
                 "NOT the official scIB score",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "scrna_arc_summary.png")
    fig.savefig(OUT / "scrna_arc_summary.pdf")
    print(f"Wrote {OUT/'scrna_arc_summary.png'}")
    print(f"Wrote {OUT/'scrna_arc_summary.pdf'}")

    # tiny machine-readable companion (numbers behind the figure)
    summary = {
        "note": "reduced Python-only proxy reward (bio + batch_mixing), higher better; NOT scIB",
        "synthetic": {
            "log_norm_pca_seed": synth_seed,
            "pca_baseline_D1": d1["pca"],
            "batch_centered_ref_D1": synth_ref,
            "D2A_era_5iter_best": d2a_s["best_reward"],
            "D2A_era_5iter_valid_invalid": [d2a_s["num_valid"], d2a_s["num_invalid"]],
            "D2A_era_10iter_best": d2a_10["best_reward"],
            "D2A_era_10iter_valid_invalid": [d2a_10["num_valid"], d2a_10["num_invalid"]],
        },
        "pbmc3k": {
            "pca20_seed": pb_seed,
            "batch_centered_ref": pb_ref,
            "D3B_5iter_best": d3b_s["best_reward"],
            "D3B_5iter_valid_invalid": [d3b_s["num_valid"], d3b_s["num_invalid"]],
            "D3B_10iter_best": d3b_10["best_reward"],
            "D3B_10iter_valid_invalid": [d3b_10["num_valid"], d3b_10["num_invalid"]],
            "D3C_era_best": era["best_generated_reward_excl_seed"],
            "D3C_era_valid_invalid": [era["valid"], era["invalid"]],
            "D3C_bon_best": bon["best_generated_reward_excl_seed"],
            "D3C_bon_valid_invalid": [bon["valid"], bon["invalid"]],
            "D3C_winner_note": "effective tie (~1.4e-9 gap = float32 vs float64)",
        },
    }
    (OUT / "scrna_arc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {OUT/'scrna_arc_summary.json'}")


if __name__ == "__main__":
    main()
