"""D3A / D3A-alt: REAL scRNA data smoke test on a TINY subset (Python-only, scRNA env).

Two data sources, selected with --source:

  * `scanpy_pbmc3k` (default) — a SMALL public real dataset (10x PBMC3k, ~5 MB, downloaded by
    scanpy). Real biology is used as a PROXY `cell_type` label (Leiden clusters computed on the
    clean data), and a CONTROLLED ARTIFICIAL batch effect is injected into the counts so the
    batch-mixing objective is meaningful. This is the lightweight real-data bridge between the
    synthetic D2A task and future large-scale scRNA work — NOT official scIB benchmarking.

  * `local_h5ad` — the real Kaggle `*-train-dataset.h5ad` path (raw counts in `layers/counts`,
    real `obs['batch']` + `obs['cell_type']`). Kept for when that dataset is available locally;
    it fails gracefully with download instructions if the file is absent.

In BOTH cases the data flows through the SAME candidate interface + reduced Python-only score used
in D1/D2A:  eliminate_batch_effect_fn(adata, config) -> obsm['X_emb'].

NO R/kBET, NO official scIB score, NO Gemini/ERA/BoN, NO large downloads.

>>> The score here is the D1/D2A reduced PROXY, NOT the official 12-metric scIB score.

Run (scRNA env):
    cd /Users/zhangweikun/era/implementation
    /Users/zhangweikun/era/scRNA-env/bin/python -u scrna_realdata_smoke.py --source scanpy_pbmc3k --n_cells 500
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
import scanpy as sc

# Reuse the EXACT D1/D2A candidates + reduced score + per-candidate harness.
from scrna_synthetic_smoke import (
    candidate_pca,
    candidate_batch_centered_pca,
    run_candidate,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "saved_runs", "scrna_d3a_realdata_smoke")
# scanpy caches downloaded datasets here (gitignored); keeps ~5 MB out of the repo.
SCANPY_CACHE = os.path.join(HERE, "data", "scanpy_cache")

# ---- local_h5ad (Kaggle) expectations, from the upstream notebook ----------------------
DATASET_SUBDIR = os.path.join("datasets", "single-cell-batch-integration")
REQUIRED_TRAIN_FILE = "ffdaa1f0-b1d1-4135-8774-9fed7bf039ba-train-dataset.h5ad"
KAGGLE_URL = "https://www.kaggle.com/datasets/02bdd1a079f253e04766213cb09e71d79a912200b901cac83fe7ab40bcd7cd48"


# ======================================================================================
# Source 1: small public real data (scanpy PBMC3k) + proxy labels + artificial batch
# ======================================================================================
def load_pbmc3k_adata(n_hvg: int = 2000, leiden_resolution: float = 1.0) -> ad.AnnData:
    """Load 10x PBMC3k (real counts), reduce to n_hvg highly-variable genes, and attach a
    PROXY cell_type label from Leiden clustering computed on the CLEAN (pre-batch) data."""
    os.makedirs(SCANPY_CACHE, exist_ok=True)
    sc.settings.datasetdir = SCANPY_CACHE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = sc.datasets.pbmc3k()          # ~2700 cells x 32738 genes, raw counts
        adata.var_names_make_unique()
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        # Proxy cell_type: Leiden on a normalized/scaled copy (real biological structure).
        tmp = adata.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg)
        hvg_genes = tmp.var_names[tmp.var["highly_variable"]].tolist()
        tmp = tmp[:, hvg_genes].copy()
        sc.pp.scale(tmp, max_value=10)
        sc.tl.pca(tmp, n_comps=30)
        sc.pp.neighbors(tmp, n_neighbors=15)
        sc.tl.leiden(tmp, resolution=leiden_resolution, flavor="igraph",
                     n_iterations=2, directed=False)

        # Keep RAW counts on the same HVGs; attach the proxy labels.
        adata = adata[:, hvg_genes].copy()
        adata.obs = pd.DataFrame(
            {"cell_type": pd.Categorical(tmp.obs["leiden"].values)},
            index=adata.obs_names.copy(),
        )
    return adata


def inject_artificial_batch(adata: ad.AnnData, n_batches: int, batch_strength: float,
                            seed: int) -> ad.AnnData:
    """Split cells into n_batches groups and inject a controlled per-(batch,gene) multiplicative
    nuisance effect into the counts. Returns a copy with obs['batch'] added."""
    rng = np.random.default_rng(seed)
    n = adata.n_obs
    batch_ids = rng.integers(0, n_batches, size=n)
    X = adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X.astype(float)
    for b in range(n_batches):
        mask = batch_ids == b
        if not mask.any():
            continue
        factor = np.exp(rng.normal(0.0, batch_strength, size=X.shape[1]))  # log-normal per gene
        X[mask] *= factor
    out = adata.copy()
    out.X = X
    out.obs["batch"] = pd.Categorical([f"batch{b}" for b in batch_ids])
    return out


# ======================================================================================
# Source 2: local Kaggle h5ad (real batch + real cell_type)
# ======================================================================================
def candidate_data_dirs() -> list[str]:
    return [
        os.path.join(HERE, "notebooks", DATASET_SUBDIR),
        os.path.join(HERE, DATASET_SUBDIR),
        os.path.join(os.getcwd(), DATASET_SUBDIR),
    ]


def find_data_file(explicit: str | None) -> tuple[str | None, list[str]]:
    searched: list[str] = []
    if explicit:
        searched.append(explicit)
        if os.path.isfile(explicit):
            return explicit, searched
        if os.path.isdir(explicit):
            cand = os.path.join(explicit, REQUIRED_TRAIN_FILE)
            searched.append(cand)
            if os.path.isfile(cand):
                return cand, searched
    seen = set()
    for d in candidate_data_dirs():
        if d in seen:
            continue
        seen.add(d)
        cand = os.path.join(d, REQUIRED_TRAIN_FILE)
        searched.append(cand)
        if os.path.isfile(cand):
            return cand, searched
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
    return "\n".join([
        "",
        "=" * 78,
        "local_h5ad source: real Kaggle dataset NOT found.",
        "=" * 78,
        f"Required file:  {REQUIRED_TRAIN_FILE}",
        f"Place it in:    {canonical}/",
        "Searched:",
        *[f"  - {p}" for p in searched],
        "",
        f"Dataset page (private, ~3GB): {KAGGLE_URL}",
        "The API/CLI download is blocked (private dataset); download in a browser and drop the",
        "*-train-dataset.h5ad into the folder above, then re-run with --source local_h5ad.",
        "",
        "TIP: for a quick real-data smoke without that download, use the small public source:",
        "  --source scanpy_pbmc3k --n_cells 500",
        "=" * 78,
    ])


def load_counts_adata(path: str, batch_key: str, label_key: str) -> ad.AnnData:
    full = ad.read_h5ad(path)
    X = full.layers["counts"] if "counts" in getattr(full, "layers", {}) else full.X
    obs_cols = list(full.obs.columns)
    if batch_key not in full.obs or label_key not in full.obs:
        raise KeyError(
            f"expected obs columns '{batch_key}' and '{label_key}' not both present. "
            f"Available obs columns: {obs_cols}. Use --batch_key/--label_key to map fields."
        )
    obs = pd.DataFrame(
        {"batch": full.obs[batch_key].astype("category").values,
         "cell_type": full.obs[label_key].astype("category").values},
        index=full.obs_names.copy(),
    )
    return ad.AnnData(X=X, obs=obs, var=full.var.copy())


# ======================================================================================
# Shared
# ======================================================================================
def subsample(adata: ad.AnnData, n_cells: int, seed: int) -> ad.AnnData:
    n = adata.n_obs
    if n_cells >= n:
        return adata.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_cells, replace=False)
    return adata[idx, :].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="scanpy_pbmc3k",
                        choices=["scanpy_pbmc3k", "pbmc3k", "local_h5ad"],
                        help="Data source (default: scanpy_pbmc3k = small public real data).")
    parser.add_argument("--n_cells", type=int, default=500, help="Tiny subsample size.")
    # local_h5ad options
    parser.add_argument("--data_file", default=None,
                        help="[local_h5ad] path to the train-dataset .h5ad (or its directory).")
    parser.add_argument("--batch_key", default="batch",
                        help="[local_h5ad] obs column with batch labels.")
    parser.add_argument("--label_key", default="cell_type",
                        help="[local_h5ad] obs column with cell-type labels.")
    # pbmc3k options
    parser.add_argument("--n_batches", type=int, default=3,
                        help="[pbmc3k] number of artificial batches to inject.")
    parser.add_argument("--batch_strength", type=float, default=0.8,
                        help="[pbmc3k] std of the log-normal per-gene batch effect.")
    parser.add_argument("--n_hvg", type=int, default=2000,
                        help="[pbmc3k] number of highly-variable genes to keep.")
    parser.add_argument("--leiden_resolution", type=float, default=1.0,
                        help="[pbmc3k] Leiden resolution for the proxy cell_type labels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    source = "scanpy_pbmc3k" if args.source == "pbmc3k" else args.source
    print("=" * 78)
    print(f"D3A real-data smoke test  (source={source}; NOT the official scIB score)")
    print("=" * 78)

    batch_kind = label_kind = data_desc = None

    if source == "local_h5ad":
        data_path, searched = find_data_file(args.data_file)
        if data_path is None:
            print(missing_data_message(searched))
            sys.exit(2)
        print(f"Dataset file: {data_path}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full = load_counts_adata(data_path, args.batch_key, args.label_key)
        except Exception as e:
            print(f"\n[!] Failed to load/parse the dataset: {type(e).__name__}: {e}")
            sys.exit(3)
        adata = subsample(full, args.n_cells, args.seed)
        batch_kind, label_kind = "real", "real"
        data_desc = os.path.basename(data_path)
    else:  # scanpy_pbmc3k
        print("Loading small public real data (10x PBMC3k via scanpy)...")
        try:
            full = load_pbmc3k_adata(n_hvg=args.n_hvg, leiden_resolution=args.leiden_resolution)
        except Exception as e:
            print(f"\n[!] Could not load PBMC3k (network/scanpy issue): {type(e).__name__}: {e}"
                  f"\n    Retry, or use --source local_h5ad once the Kaggle file is present.")
            sys.exit(3)
        adata = subsample(full, args.n_cells, args.seed)
        adata = inject_artificial_batch(adata, args.n_batches, args.batch_strength, args.seed)
        batch_kind = f"artificial (injected, n_batches={args.n_batches}, strength={args.batch_strength})"
        label_kind = f"proxy (Leiden res={args.leiden_resolution})"
        data_desc = "10x PBMC3k (scanpy)"

    Xd = adata.X
    Xd = np.asarray(Xd.todense()) if hasattr(Xd, "todense") else np.asarray(Xd)
    print(f"Data: {data_desc} | subsampled to {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  batches: {adata.obs['batch'].nunique()} ({batch_kind})")
    print(f"  cell_types: {adata.obs['cell_type'].nunique()} ({label_kind})")
    print(f"  X range [{Xd.min():.2f}, {Xd.max():.2f}]\n")

    candidates = {"pca": candidate_pca, "batch_centered_pca": candidate_batch_centered_pca}
    config = {"n_comps": 20}
    records = []
    for name, fn in candidates.items():
        rec = run_candidate(name, fn, adata, config)
        rec.pop("traceback", None)
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
                 "Real scIB (kBET/LISI/PCR/cell-cycle + R) deferred."),
        "score_definition": "score = bio_score + batch_mixing_score (higher is better)",
        "source": source,
        "data": data_desc,
        "batch_labels": batch_kind,
        "cell_type_labels": label_kind,
        "subsample": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_batches": int(adata.obs["batch"].nunique()),
            "n_cell_types": int(adata.obs["cell_type"].nunique()),
            "seed": args.seed,
        },
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "anndata": ad.__version__, "scanpy": sc.__version__},
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
    print(f"Summary: {len(valid)}/{len(records)} candidates valid on real data ({source}).")
    if valid:
        best = max(valid, key=lambda r: r["score"])
        print(f"Best: {best['candidate']} (score={best['score']:.4f})")
    print("\nReminder: D3A reduced PROXY score, not the official scIB score.")


if __name__ == "__main__":
    main()
