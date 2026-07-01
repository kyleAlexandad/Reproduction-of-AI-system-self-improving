"""D2A: ERA-scorable SYNTHETIC scRNA batch-integration scorer (runs in the scRNA env).

This is the scRNA analogue of the GIFT-Eval C2/C6A scorer (`gift_eval_task.py`). It takes a
candidate `.py` file that defines the upstream ERA scRNA interface

    def eliminate_batch_effect_fn(adata, config):
        ...            # must set adata.obsm["X_emb"]
        return adata

runs it on the SAME deterministic synthetic AnnData used in D1
(`scrna_synthetic_smoke.make_synthetic_adata`), validates the embedding, and scores it with the
SAME reduced Python-only proxy (`scrna_synthetic_smoke.reduced_score`).

  reward = reduced score  (HIGHER is better; ERA maximises)

Still SYNTHETIC ONLY — NO real dataset, NO R/kBET, NO Gemini. This must be run with the scRNA env
python (`/Users/zhangweikun/era/scRNA-env/bin/python`); the ERA controller
(`scrna_era_search.py`) calls it via subprocess, mirroring the GIFT-Eval two-environment pattern.

>>> The score here is NOT the official 12-metric scIB score. It is a D2A synthetic smoke-test
>>> proxy only. Real scIB scoring (kBET/LISI/PCR/cell-cycle + R) and real data are deferred to D2B.

============================ SECURITY WARNING ============================
This scorer executes candidate Python directly (no isolation). Toy use only.
=========================================================================

CLI:
    <scRNA-env python> -u scrna_synthetic_task.py --candidate cand.py [--candidate cand2.py ...] \
        --out-dir /path/to/logs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from typing import Any

import numpy as np

# Reuse the EXACT D1 synthetic data + reduced score (same directory, same env).
# make_synthetic_adata is deterministic (seed=42) so D1 and D2A see identical data.
from scrna_synthetic_smoke import make_synthetic_adata, reduced_score, validate_output


def load_candidate_fn(path: str):
    """Import `eliminate_batch_effect_fn` from a candidate .py file."""
    spec = importlib.util.spec_from_file_location("scrna_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load candidate module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "eliminate_batch_effect_fn", None)
    if fn is None or not callable(fn):
        raise AttributeError(
            "candidate file must define a callable "
            "eliminate_batch_effect_fn(adata, config)"
        )
    return fn


def evaluate_candidate(path: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score one candidate file. Never raises: any failure -> valid=False, reward=None."""
    config = dict(config or {})
    result: dict[str, Any] = {
        "candidate": os.path.basename(path),
        "candidate_path": os.path.abspath(path),
        "valid": False,
        "reward": None,          # = score when valid; None when invalid
        "score": None,
        "bio_score": None,
        "batch_mixing_score": None,
        "bio_silhouette": None,
        "batch_silhouette": None,
        "bio_knn_accuracy": None,
        "batch_knn_mixing": None,
        "embedding_shape": None,
        "error": None,
        "runtime_s": None,
    }

    adata = make_synthetic_adata()   # deterministic; identical to D1
    n_cells = adata.n_obs
    t0 = time.time()
    try:
        fn = load_candidate_fn(path)

        # Enforce "candidate must NOT use cell_type": pop it before calling.
        model_input = adata.copy()
        cell_type_info = model_input.obs.pop("cell_type")

        out = fn(model_input, config)

        err = validate_output(out, n_cells)
        if err is not None:
            raise ValueError(err)

        # Restore cell_type (by cell name) + ensure batch present, then score.
        out.obs["cell_type"] = cell_type_info.reindex(out.obs_names).values
        if "batch" not in out.obs:
            out.obs["batch"] = adata.obs["batch"].reindex(out.obs_names).values

        emb_shape = list(np.asarray(out.obsm["X_emb"]).shape)
        sc_res = reduced_score(out)

        result.update(
            valid=True,
            score=sc_res.score,
            reward=sc_res.score,          # reward = reduced score, higher is better
            bio_score=sc_res.bio_score,
            batch_mixing_score=sc_res.batch_mixing_score,
            bio_silhouette=sc_res.bio_silhouette,
            batch_silhouette=sc_res.batch_silhouette,
            bio_knn_accuracy=sc_res.bio_knn_accuracy,
            batch_knn_mixing=sc_res.batch_knn_mixing,
            embedding_shape=emb_shape,
        )
    except Exception as e:  # noqa: BLE001 — scorer intentionally catches everything
        result["valid"] = False
        result["reward"] = None
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["runtime_s"] = round(time.time() - t0, 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="Path to a candidate .py file (repeatable to score several at once).",
    )
    parser.add_argument("--out-dir", default=None,
                        help="Directory to write candidate_results.json (and .csv).")
    parser.add_argument("--config-json", default=None,
                        help="Optional JSON string passed as `config` to the candidate.")
    args = parser.parse_args()

    config = json.loads(args.config_json) if args.config_json else {}

    results = [evaluate_candidate(p, config) for p in args.candidate]

    meta = {
        "task": "scRNA-seq batch integration (D2A synthetic ERA-scorable scorer)",
        "note": (
            "Reduced Python-only proxy; NOT the official scIB score. Real scIB (kBET/LISI/PCR/"
            "cell-cycle + R) and real data are deferred to D2B."
        ),
        "reward_definition": "reward = reduced score = bio_score + batch_mixing_score (higher better)",
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }

    payload = {"meta": meta, "results": results}

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        json_path = os.path.join(args.out_dir, "candidate_results.json")
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

    # Always echo a compact, parseable summary to stdout (used as a fallback by the controller).
    for r in results:
        if r["valid"]:
            print(
                f"valid=True score={r['score']:.6f} reward={r['reward']:.6f} "
                f"bio={r['bio_score']:.4f} batch_mix={r['batch_mixing_score']:.4f} "
                f"emb={r['embedding_shape']} [{r['candidate']}]"
            )
        else:
            print(f"valid=False reward=None error={r['error']} [{r['candidate']}]")


if __name__ == "__main__":
    main()
