"""D3A: REAL scRNA data setup + tiny real-data smoke test (Python-only, scRNA env).

Verifies the REAL data path end-to-end on a TINY subset, WITHOUT any of the heavy pieces:
  * loads the real Kaggle `*-train-dataset.h5ad` (raw counts in `layers/counts`),
  * checks the required fields exist (obs['batch'], obs['cell_type']),
  * subsamples to a tiny number of cells (default 100 — same as the notebook's dev test),
  * runs the SAME candidate interface + reduced Python-only score used in D1/D2A
    (`eliminate_batch_effect_fn(adata, config) -> obsm['X_emb']`),
  * writes compact JSON/CSV.

NO real-data DOWNLOAD is performed here, NO R/kBET, NO official scIB score, NO Gemini/ERA/BoN.
If the dataset is absent, the script exits with a clear message telling you exactly which file
is missing, where to place it, and how to download it.

>>> The score here is the D1/D2A reduced PROXY, NOT the official 12-metric scIB score. Real scIB
>>> (kBET/LISI/PCR/cell-cycle + R) is deferred to a later stage.

Run (scRNA env):
    cd /Users/zhangweikun/era/implementation
    /Users/zhangweikun/era/scRNA-env/bin/python -u scrna_realdata_smoke.py --n_cells 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd

# Reuse the EXACT D1/D2A candidates + reduced score + per-candidate harness.
from scrna_synthetic_smoke import (
    candidate_pca,
    candidate_batch_centered_pca,
    reduced_score,
    run_candidate,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "saved_runs", "scrna_d3a_realdata_smoke")

# --- What the upstream notebook (single_cell_batch_integration.ipynb) expects ------------
# INPUT_DIR = './datasets/'  (relative to the notebook in implementation/notebooks/)
# INPUT_FILE_DIR = './datasets/single-cell-batch-integration/'
# Train dataset (raw counts in layers/counts): the file below.
DATASET_SUBDIR = os.path.join("datasets", "single-cell-batch-integration")
REQUIRED_TRAIN_FILE = "ffdaa1f0-b1d1-4135-8774-9fed7bf039ba-train-dataset.h5ad"
KAGGLE_URL = "https://www.kaggle.com/datasets/02bdd1a079f253e04766213cb09e71d79a912200b901cac83fe7ab40bcd7cd48"

# Directories to search for the dataset, most-canonical first.
def candidate_data_dirs() -> list[str]:
    return [
        os.path.join(HERE, "notebooks", DATASET_SUBDIR),  # matches the notebook's CWD
        os.path.join(HERE, DATASET_SUBDIR),               # implementation/datasets/...
        os.path.join(os.getcwd(), DATASET_SUBDIR),        # ./datasets/... from cwd
    ]


def find_data_file(explicit: str | None) -> tuple[str | None, list[str]]:
    """Locate the train-dataset .h5ad. Returns (path_or_None, searched_locations)."""
    searched: list[str] = []
    if explicit:
        searched.append(explicit)
        if os.path.isfile(explicit):
            return explicit, searched
        # If a directory was passed, look inside it.
        if os.path.isdir(explicit):
            cand = os.path.join(explicit, REQUIRED_TRAIN_FILE)
            searched.append(cand)
            if os.path.isfile(cand):
                return cand, searched
    for d in candidate_data_dirs():
        cand = os.path.join(d, REQUIRED_TRAIN_FILE)
        searched.append(cand)
        if os.path.isfile(cand):
            return cand, searched
        # Fallback: any *train-dataset*.h5ad, then any *.h5ad in that dir.
        if os.path.isdir(d):
            hits = sorted(
                [f for f in os.listdir(d) if f.endswith(".h5ad") and "train-dataset" in f]
                or [f for f in os.listdir(d) if f.endswith(".h5ad")]
            )
            if hits:
                return os.path.join(d, hits[0]), searched
    return None, searched


def missing_data_message(searched: list[str]) -> str:
    canonical = os.path.join(HERE, "notebooks", DATASET_SUBDIR)
    lines = [
        "",
        "=" * 78,
        "REAL scRNA DATASET NOT FOUND — D3A cannot run the real-data smoke yet.",
        "=" * 78,
        "",
        f"Required file:  {REQUIRED_TRAIN_FILE}",
        f"Place it in:    {canonical}/",
        "",
        "Searched these locations:",
        *[f"  - {p}" for p in searched],
        "",
        "HOW TO GET IT (Kaggle 'Single-Cell Biology' dataset):",
        f"  Dataset page: {KAGGLE_URL}",
        "",
        "  Option A — Kaggle CLI (needs a Kaggle account + API token at ~/.kaggle/kaggle.json):",
        "    pip install kaggle   # in any env; it is just a downloader",
        f"    mkdir -p '{canonical}'",
        "    # Open the dataset page above while logged in to read its owner/slug, then:",
        "    kaggle datasets download -d <owner>/<dataset-slug> \\",
        f"        -p '{canonical}' --unzip",
        "",
        "  Option B — Manual: download the dataset zip from the page in a browser and unzip it",
        f"    into '{canonical}/' so the .h5ad files sit directly inside that folder.",
        "",
        "For THIS reduced smoke, only the *-train-dataset.h5ad is required",
        "(the *-solution.h5ad and score-bounds CSVs are only needed for the full scIB score, later).",
        "Then re-run:",
        "    /Users/zhangweikun/era/scRNA-env/bin/python -u scrna_realdata_smoke.py --n_cells 100",
        "=" * 78,
    ]
    return "\n".join(lines)


def load_counts_adata(path: str, batch_key: str, label_key: str) -> ad.AnnData:
    """Load the real dataset file and return a clean AnnData with raw counts in .X and
    obs['batch']/obs['cell_type']. Mirrors the notebook, which reads X from layers/counts."""
    full = ad.read_h5ad(path)
    # Raw counts live in layers['counts'] in this benchmark; fall back to .X otherwise.
    if "counts" in getattr(full, "layers", {}):
        X = full.layers["counts"]
    else:
        X = full.X

    obs_cols = list(full.obs.columns)
    if batch_key not in full.obs or label_key not in full.obs:
        raise KeyError(
            f"expected obs columns '{batch_key}' and '{label_key}' not both present. "
            f"Available obs columns: {obs_cols}. "
            f"Re-run with --batch_key/--label_key to map the correct fields."
        )

    obs = pd.DataFrame(
        {
            "batch": full.obs[batch_key].astype("category").values,
            "cell_type": full.obs[label_key].astype("category").values,
        },
        index=full.obs_names.copy(),
    )
    var = full.var.copy()
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def subsample(adata: ad.AnnData, n_cells: int, seed: int) -> ad.AnnData:
    """Random subsample of n_cells (matches the notebook's 100-cell dev test, cell 17)."""
    n = adata.n_obs
    if n_cells >= n:
        return adata.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_cells, replace=False)
    return adata[idx, :].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_cells", type=int, default=100,
                        help="Tiny subsample size (default 100).")
    parser.add_argument("--data_file", default=None,
                        help="Path to the train-dataset .h5ad (or its directory). "
                             "If omitted, standard locations are searched.")
    parser.add_argument("--batch_key", default="batch",
                        help="obs column with batch labels (default 'batch').")
    parser.add_argument("--label_key", default="cell_type",
                        help="obs column with cell-type labels (default 'cell_type').")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    print("=" * 78)
    print("D3A REAL scRNA smoke test  (Python-only; NOT the official scIB score)")
    print("=" * 78)

    data_path, searched = find_data_file(args.data_file)
    if data_path is None:
        print(missing_data_message(searched))
        sys.exit(2)

    print(f"Dataset file: {data_path}")
    t_load = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = load_counts_adata(data_path, args.batch_key, args.label_key)
    except Exception as e:
        print(f"\n[!] Failed to load / parse the dataset: {type(e).__name__}: {e}")
        sys.exit(3)
    print(f"Loaded full AnnData: {full.n_obs} cells x {full.n_vars} genes "
          f"in {time.time()-t_load:.1f}s")
    print(f"  batches: {full.obs['batch'].nunique()} | "
          f"cell_types: {full.obs['cell_type'].nunique()}")

    adata = subsample(full, args.n_cells, args.seed)
    Xd = adata.X
    xmin = float(Xd.min()); xmax = float(Xd.max())
    print(f"Subsampled to {adata.n_obs} cells | batches={adata.obs['batch'].nunique()} "
          f"cell_types={adata.obs['cell_type'].nunique()} | X range [{xmin:.1f}, {xmax:.1f}]\n")

    candidates = {
        "pca": candidate_pca,
        "batch_centered_pca": candidate_batch_centered_pca,
    }
    config = {"n_comps": 20}
    records = []
    for name, fn in candidates.items():
        rec = run_candidate(name, fn, adata, config)
        rec.pop("traceback", None)  # keep JSON compact
        records.append(rec)
        if rec["valid"]:
            print(f"[ OK ] {name:20s} score={rec['score']:.4f} "
                  f"(bio={rec['bio_score']:.3f}, batch_mix={rec['batch_mixing_score']:.3f}) "
                  f"emb={rec['embedding_shape']}")
        else:
            print(f"[FAIL] {name:20s} error: {rec['error']}")

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "task": "scRNA-seq batch integration (D3A real-data tiny smoke)",
        "note": ("Reduced Python-only proxy score; NOT the official scIB score. "
                 "Real scIB (kBET/LISI/PCR/cell-cycle + R) deferred to a later stage."),
        "score_definition": "score = bio_score + batch_mixing_score (higher is better)",
        "dataset_file": data_path,
        "full_shape": [int(full.n_obs), int(full.n_vars)],
        "subsample": {
            "n_cells": int(adata.n_obs),
            "n_batches": int(adata.obs["batch"].nunique()),
            "n_cell_types": int(adata.obs["cell_type"].nunique()),
            "seed": args.seed,
        },
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "anndata": ad.__version__},
    }
    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": records}, f, indent=2)

    csv_cols = ["candidate", "valid", "score", "normalized_score", "bio_score",
                "batch_mixing_score", "bio_silhouette", "batch_silhouette",
                "bio_knn_accuracy", "batch_knn_mixing", "embedding_shape", "error", "runtime_s"]
    df = pd.DataFrame([{k: r.get(k) for k in csv_cols} for r in records])
    csv_path = os.path.join(args.out_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nWrote:\n  {json_path}\n  {csv_path}")
    valid = [r for r in records if r["valid"]]
    print(f"Summary: {len(valid)}/{len(records)} candidates valid on real data.")
    if valid:
        best = max(valid, key=lambda r: r["score"])
        print(f"Best: {best['candidate']} (score={best['score']:.4f})")
    print("\nReminder: D3A reduced PROXY score, not the official scIB score.")


if __name__ == "__main__":
    main()
