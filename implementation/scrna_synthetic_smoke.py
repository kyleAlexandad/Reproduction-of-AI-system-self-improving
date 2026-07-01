"""D1 — scRNA-seq batch-integration SYNTHETIC smoke test (Python-only, no downloads).

Purpose
-------
This is a self-contained smoke test for the upstream ERA single-cell batch-integration
task interface (see `implementation/notebooks/single_cell_batch_integration.ipynb`).
It verifies the plumbing END-TO-END on *synthetic* data, WITHOUT any of the heavy /
external pieces that the real task needs:

  * NO real dataset download (data is generated in-memory).
  * NO R / kBET / anndata2ri / rpy2.
  * NO Gemini, NO ERA search, NO best-of-N.
  * NO official 12-metric scIB score.

What it checks
--------------
  1. The scRNA Python env (anndata / scanpy / scib / sklearn) works.
  2. A small synthetic AnnData with batch + cell-type structure can be built.
  3. A candidate obeying the ERA scRNA interface runs:
         eliminate_batch_effect_fn(adata, config) -> AnnData with obsm['X_emb']
  4. The candidate writes a batch-corrected embedding into obsm['X_emb'].
  5. A REDUCED, Python-only metric suite can score that embedding.
  6. Invalid candidates (missing X_emb / wrong shape) are caught and reported.

>>> IMPORTANT: the score computed here is NOT the official scIB batch-integration
>>> score. It is a lightweight D1 smoke-test proxy only. The real 12-metric scIB
>>> score (ASW/ARI/NMI/kBET/iLISI/cLISI/PCR/cell-cycle/...) is deferred to D2, and
>>> needs R + kBET + the real Kaggle dataset.

Run (from a normal terminal, with the dedicated scRNA env):
    cd /Users/zhangweikun/era/implementation
    /Users/zhangweikun/era/scRNA-env/bin/python -u scrna_synthetic_smoke.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import traceback
from typing import Any, Callable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "saved_runs", "scrna_d1_synthetic_smoke")

# Synthetic-data hyperparameters (kept intentionally small + fast).
N_CELLS = 300
N_GENES = 100
N_BATCHES = 3
N_CELL_TYPES = 3
BATCH_STRENGTH = 1.2   # per-gene log-space batch shift std (nuisance to be removed)
CELLTYPE_STRENGTH = 1.6  # per-gene log-space cell-type signal std (biology to preserve)
SEED = 42


# --------------------------------------------------------------------------------------
# 1. Synthetic data
# --------------------------------------------------------------------------------------
def make_synthetic_adata(
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    n_batches: int = N_BATCHES,
    n_cell_types: int = N_CELL_TYPES,
    batch_strength: float = BATCH_STRENGTH,
    celltype_strength: float = CELLTYPE_STRENGTH,
    seed: int = SEED,
) -> ad.AnnData:
    """Build a small raw-count AnnData with BOTH cell-type signal and batch effect.

    Counts are generated as Poisson draws whose log-rate is
        base + celltype_profile[type] + batch_shift[batch] + per-cell noise.
    So cell type drives genuine biological variation (to be preserved) while batch
    adds a nuisance shift (to be removed) — exactly the tension the real task poses.
    """
    rng = np.random.default_rng(seed)

    cell_types = rng.integers(0, n_cell_types, size=n_cells)
    batches = rng.integers(0, n_batches, size=n_cells)

    base = rng.normal(1.0, 0.3, size=n_genes)                       # per-gene baseline
    celltype_profiles = rng.normal(0.0, celltype_strength,
                                   size=(n_cell_types, n_genes))    # biology
    batch_shifts = rng.normal(0.0, batch_strength,
                              size=(n_batches, n_genes))            # nuisance
    cell_noise = rng.normal(0.0, 0.25, size=(n_cells, n_genes))     # per-cell jitter

    log_rate = (
        base[None, :]
        + celltype_profiles[cell_types]
        + batch_shifts[batches]
        + cell_noise
    )
    rate = np.exp(np.clip(log_rate, None, 8.0))   # clip to avoid huge counts / overflow
    counts = rng.poisson(rate).astype(np.float32)

    obs = pd.DataFrame(
        {
            "batch": pd.Categorical([f"batch{b}" for b in batches]),
            "cell_type": pd.Categorical([f"type{t}" for t in cell_types]),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)])
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    return adata


# --------------------------------------------------------------------------------------
# 2. Candidate interface (same as the upstream ERA scRNA task)
#    def eliminate_batch_effect_fn(adata, config) -> AnnData with obsm['X_emb']
#    NOTE: candidates must NOT use cell_type (the harness pops it before calling).
# --------------------------------------------------------------------------------------
def _lognorm_pca(adata: ad.AnnData, n_comps: int) -> np.ndarray:
    """Standard normalize_total -> log1p -> PCA. Returns the embedding matrix."""
    work = adata.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    X = work.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    n_comps = int(min(n_comps, X.shape[1] - 1, X.shape[0] - 1))
    emb = PCA(n_components=n_comps, random_state=0).fit_transform(X)
    return emb


def candidate_pca(adata: ad.AnnData, config: dict[str, Any]) -> ad.AnnData:
    """Baseline 1: log-normalized PCA (no batch correction at all)."""
    emb = _lognorm_pca(adata, n_comps=config.get("n_comps", 20))
    return ad.AnnData(obs=adata.obs.copy(), var=adata.var.copy(), obsm={"X_emb": emb})


def candidate_batch_centered_pca(adata: ad.AnnData, config: dict[str, Any]) -> ad.AnnData:
    """Baseline 2: log-normalized PCA, then subtract the per-batch mean in embedding
    space (a simple linear batch correction). Uses obs['batch'] only — never cell_type.
    """
    emb = _lognorm_pca(adata, n_comps=config.get("n_comps", 20))
    batch = adata.obs["batch"].to_numpy()
    corrected = emb.copy()
    for b in np.unique(batch):
        mask = batch == b
        corrected[mask] -= corrected[mask].mean(axis=0, keepdims=True)
    return ad.AnnData(obs=adata.obs.copy(), var=adata.var.copy(),
                      obsm={"X_emb": corrected})


def candidate_invalid_missing(adata: ad.AnnData, config: dict[str, Any]) -> ad.AnnData:
    """Invalid: returns an AnnData with NO X_emb in obsm (should be caught)."""
    return ad.AnnData(obs=adata.obs.copy(), var=adata.var.copy())


def candidate_invalid_shape(adata: ad.AnnData, config: dict[str, Any]) -> ad.AnnData:
    """Invalid: X_emb has the wrong number of rows (should be caught)."""
    emb = _lognorm_pca(adata, n_comps=config.get("n_comps", 20))
    bad = emb[:10]  # wrong: only 10 rows instead of n_cells
    return ad.AnnData(obs=adata.obs.copy(), var=adata.var.copy(), obsm={"X_emb": bad})


CANDIDATES: dict[str, Callable[[ad.AnnData, dict[str, Any]], ad.AnnData]] = {
    "pca": candidate_pca,
    "batch_centered_pca": candidate_batch_centered_pca,
    "invalid_missing_x_emb": candidate_invalid_missing,
    "invalid_wrong_shape": candidate_invalid_shape,
}


# --------------------------------------------------------------------------------------
# 3. Reduced, Python-only score (NOT the official scIB score — D1 proxy only)
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class ReducedScore:
    # headline (higher is better): bio_score + batch_mixing_score, range ~[0, 2]
    score: float
    normalized_score: float        # score / 2, range ~[0, 1]
    # biology preservation (higher = cell types stay separable = good)
    bio_score: float               # (silhouette(cell_type) + 1) / 2  in [0, 1]
    bio_silhouette: float          # raw silhouette on cell_type in [-1, 1]
    bio_knn_accuracy: float | None  # kNN cell-type accuracy (cross-val); None if too few per class
    # batch mixing (higher = batches well mixed = good)
    batch_mixing_score: float      # 1 - (silhouette(batch) + 1)/2  in [0, 1]
    batch_silhouette: float        # raw silhouette on batch in [-1, 1]
    batch_knn_mixing: float | None  # 1 - normalized kNN batch bal-acc; None if too few per class


def _knn_cv_predict(X, labels):
    """Robust kNN cross-val prediction. Returns None if the label set is too small for
    stratified CV (e.g. a rare cell type with <2 members in a tiny real-data subset), so
    the secondary kNN proxy degrades gracefully instead of raising. On large synthetic
    classes this uses cv=3 exactly as before (numbers unchanged)."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min()) if counts.size else 0
    if classes.size < 2 or min_count < 2:
        return None
    cv = int(min(3, min_count))
    try:
        return cross_val_predict(KNeighborsClassifier(n_neighbors=5), X, labels, cv=cv)
    except Exception:
        return None


def reduced_score(emb_adata: ad.AnnData, seed: int = SEED) -> ReducedScore:
    """Cheap proxy for the two goals of batch integration.

    Biology preservation  -> cell types SHOULD remain separable in X_emb.
    Batch mixing          -> batches SHOULD be well mixed (NOT separable) in X_emb.

    Both are mapped so that HIGHER is better, and the headline `score` sums them
    (mirroring `score = biological_preservation_score + batch_mixing_score`).

    Proxies used:
      * silhouette_score on cell_type / batch labels (primary).
      * kNN classification cross-check (cell_type accuracy; batch balanced accuracy).
    This is deliberately lightweight and is NOT the official scIB metric.
    """
    X = np.asarray(emb_adata.obsm["X_emb"])
    cell_type = emb_adata.obs["cell_type"].to_numpy()
    batch = emb_adata.obs["batch"].to_numpy()

    # --- silhouette-based proxies ---
    bio_sil = float(silhouette_score(X, cell_type))
    batch_sil = float(silhouette_score(X, batch))
    bio_score = (bio_sil + 1.0) / 2.0                 # higher = types separable
    batch_mixing_score = 1.0 - (batch_sil + 1.0) / 2.0  # higher = batches mixed

    # --- kNN cross-checks (cheap 5-NN cross_val_predict; robust to tiny classes) ---
    bio_pred = _knn_cv_predict(X, cell_type)
    bio_knn_acc = float((bio_pred == cell_type).mean()) if bio_pred is not None else None

    batch_pred = _knn_cv_predict(X, batch)
    if batch_pred is not None:
        batch_bal_acc = float(balanced_accuracy_score(batch, batch_pred))
        chance = 1.0 / len(np.unique(batch))
        # normalize balanced-acc above chance to [0,1], then invert so higher = mixed
        batch_sep = np.clip((batch_bal_acc - chance) / (1.0 - chance), 0.0, 1.0)
        batch_knn_mixing = float(1.0 - batch_sep)
    else:
        batch_knn_mixing = None

    score = bio_score + batch_mixing_score
    return ReducedScore(
        score=float(score),
        normalized_score=float(score / 2.0),
        bio_score=float(bio_score),
        bio_silhouette=bio_sil,
        bio_knn_accuracy=bio_knn_acc,
        batch_mixing_score=float(batch_mixing_score),
        batch_silhouette=batch_sil,
        batch_knn_mixing=batch_knn_mixing,
    )


# --------------------------------------------------------------------------------------
# 4. Harness: run + validate + score each candidate
# --------------------------------------------------------------------------------------
def validate_output(out: Any, n_cells: int) -> str | None:
    """Return an error string if the candidate output is invalid, else None."""
    if not isinstance(out, ad.AnnData):
        return f"output is not AnnData (got {type(out).__name__})"
    if "X_emb" not in out.obsm:
        return "output.obsm is missing required key 'X_emb'"
    emb = np.asarray(out.obsm["X_emb"])
    if emb.ndim != 2:
        return f"X_emb must be 2D, got ndim={emb.ndim}"
    if emb.shape[0] != n_cells:
        return f"X_emb has {emb.shape[0]} rows, expected {n_cells} (one per cell)"
    if emb.shape[1] < 1:
        return "X_emb has 0 columns"
    if not np.all(np.isfinite(emb)):
        return "X_emb contains NaN or inf"
    return None


def run_candidate(
    name: str,
    fn: Callable[[ad.AnnData, dict[str, Any]], ad.AnnData],
    adata: ad.AnnData,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run one candidate the way the ERA harness would: pop cell_type so the candidate
    cannot use it, score with cell_type restored. Any failure -> valid=False."""
    n_cells = adata.n_obs
    t0 = time.time()
    rec: dict[str, Any] = {
        "candidate": name,
        "valid": False,
        "score": None,
        "bio_score": None,
        "batch_mixing_score": None,
        "bio_silhouette": None,
        "batch_silhouette": None,
        "bio_knn_accuracy": None,
        "batch_knn_mixing": None,
        "normalized_score": None,
        "embedding_shape": None,
        "error": None,
        "runtime_s": None,
    }
    try:
        # Enforce "candidate must not use cell_type": remove it from the input.
        model_input = adata.copy()
        cell_type_info = model_input.obs.pop("cell_type")

        out = fn(model_input, config)

        err = validate_output(out, n_cells)
        if err is not None:
            raise ValueError(err)

        # Restore cell_type (aligned by cell name) and ensure batch is present.
        out.obs["cell_type"] = cell_type_info.reindex(out.obs_names).values
        if "batch" not in out.obs:
            out.obs["batch"] = adata.obs["batch"].reindex(out.obs_names).values

        rec["embedding_shape"] = list(np.asarray(out.obsm["X_emb"]).shape)

        sc_res = reduced_score(out)
        rec.update(
            valid=True,
            score=sc_res.score,
            normalized_score=sc_res.normalized_score,
            bio_score=sc_res.bio_score,
            batch_mixing_score=sc_res.batch_mixing_score,
            bio_silhouette=sc_res.bio_silhouette,
            batch_silhouette=sc_res.batch_silhouette,
            bio_knn_accuracy=sc_res.bio_knn_accuracy,
            batch_knn_mixing=sc_res.batch_knn_mixing,
        )
    except Exception as e:  # noqa: BLE001 — smoke test intentionally catches everything
        rec["valid"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    finally:
        rec["runtime_s"] = round(time.time() - t0, 4)
    return rec


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    config: dict[str, Any] = {"n_comps": 20}

    print("=" * 78)
    print("D1 scRNA SYNTHETIC smoke test  (Python-only; NOT the official scIB score)")
    print("=" * 78)

    adata = make_synthetic_adata()
    print(
        f"Synthetic AnnData: {adata.n_obs} cells x {adata.n_vars} genes | "
        f"batches={list(adata.obs['batch'].cat.categories)} | "
        f"cell_types={list(adata.obs['cell_type'].cat.categories)}"
    )
    print(f"X dtype={adata.X.dtype} min={adata.X.min():.1f} max={adata.X.max():.1f} "
          f"(raw counts)\n")

    records = []
    for name, fn in CANDIDATES.items():
        rec = run_candidate(name, fn, adata, config)
        records.append(rec)
        if rec["valid"]:
            print(
                f"[ OK ] {name:24s} score={rec['score']:.4f} "
                f"(bio={rec['bio_score']:.3f}, batch_mix={rec['batch_mixing_score']:.3f}) "
                f"emb={rec['embedding_shape']} | bioKNN={rec['bio_knn_accuracy']:.3f} "
                f"batchMixKNN={rec['batch_knn_mixing']:.3f}"
            )
        else:
            print(f"[FAIL] {name:24s} valid=False  error: {rec['error']}")

    # --- persist results (JSON + CSV). NO markdown files. ---
    meta = {
        "task": "scRNA-seq batch integration (D1 synthetic smoke test)",
        "note": (
            "Reduced Python-only proxy score; NOT the official 12-metric scIB score. "
            "Real scIB scoring (kBET/LISI/PCR/cell-cycle/... + R) is deferred to D2."
        ),
        "score_definition": "score = bio_score + batch_mixing_score  (higher is better)",
        "bio_score": "(silhouette(cell_type)+1)/2, higher = cell types preserved",
        "batch_mixing_score": "1-(silhouette(batch)+1)/2, higher = batches well mixed",
        "dataset": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_batches": int(adata.obs["batch"].nunique()),
            "n_cell_types": int(adata.obs["cell_type"].nunique()),
            "batch_strength": BATCH_STRENGTH,
            "celltype_strength": CELLTYPE_STRENGTH,
            "seed": SEED,
        },
        "env": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "anndata": ad.__version__,
            "scanpy": sc.__version__,
        },
    }
    json_path = os.path.join(OUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": records}, f, indent=2)

    # CSV (drop the long traceback field for readability).
    csv_cols = [
        "candidate", "valid", "score", "normalized_score", "bio_score",
        "batch_mixing_score", "bio_silhouette", "batch_silhouette",
        "bio_knn_accuracy", "batch_knn_mixing", "embedding_shape", "error", "runtime_s",
    ]
    df = pd.DataFrame([{k: r.get(k) for k in csv_cols} for r in records])
    csv_path = os.path.join(OUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)

    print("\nWrote:")
    print(f"  {json_path}")
    print(f"  {csv_path}")

    valid = [r for r in records if r["valid"]]
    n_valid, n_total = len(valid), len(records)
    print(f"\nSummary: {n_valid}/{n_total} candidates valid; "
          f"{n_total - n_valid} correctly flagged invalid.")
    if valid:
        best = max(valid, key=lambda r: r["score"])
        print(f"Best valid candidate: {best['candidate']} (score={best['score']:.4f})")
    print("\nReminder: this is a D1 SMOKE-TEST proxy, not the official scIB score.")


if __name__ == "__main__":
    main()
