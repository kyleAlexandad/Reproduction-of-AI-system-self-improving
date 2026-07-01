"""D3B-alt scorer: ERA-scorable PBMC3k real-data scRNA batch-integration scorer (scRNA env).

The real-data analogue of `scrna_synthetic_task.py`. It builds a DETERMINISTIC prepared AnnData
from a small public real dataset (10x PBMC3k) — real biology as a Leiden proxy `cell_type`, plus a
controlled ARTIFICIAL batch effect — caches it, then scores a candidate `.py` file that defines the
upstream ERA interface:

    def eliminate_batch_effect_fn(adata, config):
        ...            # must set adata.obsm["X_emb"]
        return adata

Reward = the same reduced Python-only score used in D1/D2A/D3A (HIGHER is better).

The prepared dataset is cached (keyed by all params) under the gitignored scanpy cache so every
candidate in an ERA run is scored on IDENTICAL data. The ERA controller (`scrna_era_search.py`)
calls this via subprocess with the scRNA-env python, mirroring the GIFT-Eval two-environment pattern.

>>> Still a REDUCED PROXY score, NOT the official scIB score. NO R/kBET, NO official Kaggle dataset,
>>> NO Gemini.

============================ SECURITY WARNING ============================
This scorer executes candidate Python directly (no isolation). Toy use only.
=========================================================================

CLI:
    <scRNA-env python> -u scrna_realdata_task.py --candidate cand.py [--candidate ...] \
        --out-dir /path/to/logs --source scanpy_pbmc3k --n_cells 500 --n_batches 3 \
        --batch_strength 0.8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from typing import Any

import anndata as ad
import numpy as np

from scrna_synthetic_smoke import reduced_score, validate_output
from scrna_synthetic_task import load_candidate_fn
from scrna_realdata_smoke import (
    SCANPY_CACHE,
    inject_artificial_batch,
    load_pbmc3k_adata,
    subsample,
)


def prepared_cache_path(source: str, n_cells: int, n_batches: int, batch_strength: float,
                        n_hvg: int, leiden_resolution: float, seed: int) -> str:
    os.makedirs(SCANPY_CACHE, exist_ok=True)
    key = (f"{source}_n{n_cells}_b{n_batches}_s{batch_strength}"
           f"_hvg{n_hvg}_res{leiden_resolution}_seed{seed}")
    return os.path.join(SCANPY_CACHE, f"prepared_{key}.h5ad")


def build_or_load_prepared(source: str, n_cells: int, n_batches: int, batch_strength: float,
                           n_hvg: int, leiden_resolution: float, seed: int) -> ad.AnnData:
    """Return the deterministic prepared AnnData (built once, then cached + reloaded so every
    candidate is scored on byte-identical data)."""
    if source not in ("scanpy_pbmc3k", "pbmc3k"):
        raise ValueError(f"unsupported --source '{source}' (this scorer handles PBMC3k only)")
    path = prepared_cache_path(source, n_cells, n_batches, batch_strength, n_hvg,
                               leiden_resolution, seed)
    if os.path.isfile(path):
        return ad.read_h5ad(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = load_pbmc3k_adata(n_hvg=n_hvg, leiden_resolution=leiden_resolution)
        adata = subsample(full, n_cells, seed)
        adata = inject_artificial_batch(adata, n_batches, batch_strength, seed)

        # Cache. Cast string indices to plain object str (pandas 3.0 + anndata 0.11 write quirk).
        try:
            ad.settings.allow_write_nullable_strings = True
        except Exception:
            pass
        to_write = adata.copy()
        to_write.obs.index = to_write.obs.index.astype(str).astype(object)
        to_write.var.index = to_write.var.index.astype(str).astype(object)
        try:
            to_write.write_h5ad(path)
            # Reload the cached copy so the FIRST candidate sees the same round-tripped data.
            return ad.read_h5ad(path)
        except Exception:
            # If caching fails, fall back to the in-memory build (still deterministic).
            return adata


def evaluate_candidate(path: str, adata: ad.AnnData,
                       config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score one candidate file on the prepared AnnData. Never raises."""
    config = dict(config or {})
    result: dict[str, Any] = {
        "candidate": os.path.basename(path),
        "candidate_path": os.path.abspath(path),
        "valid": False, "reward": None, "score": None,
        "bio_score": None, "batch_mixing_score": None,
        "bio_silhouette": None, "batch_silhouette": None,
        "bio_knn_accuracy": None, "batch_knn_mixing": None,
        "embedding_shape": None, "error": None, "runtime_s": None,
    }
    n_cells = adata.n_obs
    t0 = time.time()
    try:
        fn = load_candidate_fn(path)
        model_input = adata.copy()
        cell_type_info = model_input.obs.pop("cell_type")   # candidate must NOT use cell_type

        out = fn(model_input, config)

        err = validate_output(out, n_cells)
        if err is not None:
            raise ValueError(err)

        out.obs["cell_type"] = cell_type_info.reindex(out.obs_names).values
        if "batch" not in out.obs:
            out.obs["batch"] = adata.obs["batch"].reindex(out.obs_names).values

        emb_shape = list(np.asarray(out.obsm["X_emb"]).shape)
        sc_res = reduced_score(out)
        result.update(
            valid=True, score=sc_res.score, reward=sc_res.score,
            bio_score=sc_res.bio_score, batch_mixing_score=sc_res.batch_mixing_score,
            bio_silhouette=sc_res.bio_silhouette, batch_silhouette=sc_res.batch_silhouette,
            bio_knn_accuracy=sc_res.bio_knn_accuracy, batch_knn_mixing=sc_res.batch_knn_mixing,
            embedding_shape=emb_shape,
        )
    except Exception as e:  # noqa: BLE001
        result["valid"] = False
        result["reward"] = None
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["runtime_s"] = round(time.time() - t0, 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True,
                        help="Path to a candidate .py (repeatable to score several at once).")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--config-json", default=None)
    # dataset params (must match the controller so all candidates see identical data)
    parser.add_argument("--source", default="scanpy_pbmc3k",
                        choices=["scanpy_pbmc3k", "pbmc3k"])
    parser.add_argument("--n_cells", type=int, default=500)
    parser.add_argument("--n_batches", type=int, default=3)
    parser.add_argument("--batch_strength", type=float, default=0.8)
    parser.add_argument("--n_hvg", type=int, default=2000)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = json.loads(args.config_json) if args.config_json else {}

    adata = build_or_load_prepared(
        args.source, args.n_cells, args.n_batches, args.batch_strength,
        args.n_hvg, args.leiden_resolution, args.seed,
    )

    results = [evaluate_candidate(p, adata, config) for p in args.candidate]

    meta = {
        "task": "scRNA-seq batch integration (D3B-alt PBMC3k ERA-scorable scorer)",
        "note": ("Reduced Python-only proxy; NOT official scIB. Real biology (PBMC3k) + artificial "
                 "batch; NO R/kBET, NO Kaggle HCA dataset."),
        "reward_definition": "reward = reduced score = bio_score + batch_mixing_score (higher better)",
        "dataset": {
            "source": args.source, "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
            "n_batches": args.n_batches, "batch_strength": args.batch_strength,
            "n_hvg": args.n_hvg, "leiden_resolution": args.leiden_resolution, "seed": args.seed,
        },
        "env": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "candidate_results.json"), "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)

    for r in results:
        if r["valid"]:
            print(f"valid=True score={r['score']:.6f} reward={r['reward']:.6f} "
                  f"bio={r['bio_score']:.4f} batch_mix={r['batch_mixing_score']:.4f} "
                  f"emb={r['embedding_shape']} [{r['candidate']}]")
        else:
            print(f"valid=False reward=None error={r['error']} [{r['candidate']}]")


if __name__ == "__main__":
    main()
