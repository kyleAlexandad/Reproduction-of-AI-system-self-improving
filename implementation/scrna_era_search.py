"""D2A: ERA / FUTS tree search over SYNTHETIC scRNA batch-integration candidates.

This is the scRNA analogue of `gift_eval_era_search.py`. It connects the synthetic scRNA scorer
(`scrna_synthetic_task.py`) to the existing ERA Flat-UCB Tree Search (`futs.search`) and a Gemini
generator (`llm.GeminiLLM`).

ARCHITECTURE (two Python environments, on purpose) — mirrors the GIFT-Eval pattern
-----------------------------------------------------------------------------------
  * THIS script runs in the ERA env (the one with `google-genai`, `futs.py`, `llm.py`). It NEVER
    imports scanpy / anndata / scib.
  * Each candidate is scored by launching the scorer in the scRNA env via subprocess:
        <scrna_python> -u scrna_synthetic_task.py --candidate <cand.py> --out-dir <scratch>
    and the reward is read back from the scorer's `candidate_results.json`
    (with a stdout-regex fallback).

ERA loop: Gemini writes/edits an `eliminate_batch_effect_fn(adata, config)` function -> the scorer
evaluates the produced embedding on synthetic data -> reward = reduced score (HIGHER is better)
-> FUTS selects a parent and iterates.

Reward convention: reward = reduced score (bio_score + batch_mixing_score). Unlike GIFT-Eval
(reward = -MASE), here HIGHER raw score is already better, so NO negation. Invalid candidates
(crash / no X_emb / wrong shape / NaN) get reward = -inf and are recorded cleanly so the search
continues.

>>> SYNTHETIC ONLY (D2A): NO real dataset, NO R/kBET, NO official scIB score. The score is a
>>> smoke-test proxy. Real data + real scIB scoring are deferred to D2B.

============================ SECURITY WARNING ============================
The scorer executes LLM-generated Python directly (no isolation). Toy use only.
=========================================================================

NOTE: do NOT run this without a GEMINI_API_KEY; it makes real Gemini calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import futs
from llm import GeminiLLM, DEFAULT_MODEL

# --- Defaults (two-env split) ------------------------------------------------------------
DEFAULT_SCRNA_PYTHON = "/Users/zhangweikun/era/scRNA-env/bin/python"
DEFAULT_SCORER_SCRIPT = str(Path(__file__).resolve().parent / "scrna_synthetic_task.py")
DEFAULT_OUT_DIR = "saved_runs/scrna_d2a_synthetic_era_smoke"


# ------------------------------- initial (seed) candidate --------------------------------
# Simple log-normalized PCA baseline (no batch correction). This is the seed the search
# starts from; ERA is asked to improve batch mixing while preserving biology.
PCA_SEED_CODE = '''\
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA


def eliminate_batch_effect_fn(adata, config):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    n_comps = int(min(10, X.shape[1] - 1, X.shape[0] - 1))
    emb = PCA(n_components=n_comps, random_state=0).fit_transform(X)
    adata.obsm["X_emb"] = emb
    return adata
'''


PROBLEM_DESCRIPTION = (
    "Improve a single-cell RNA-seq BATCH INTEGRATION function on a small synthetic dataset. "
    "The candidate maps raw counts to a low-dimensional embedding (obsm['X_emb']) that removes "
    "batch effects while preserving biological (cell-type) structure. Reward = a reduced proxy "
    "score (higher is better)."
)


def _interface_spec() -> str:
    return '''\
TASK: single-cell RNA-seq BATCH INTEGRATION on a small SYNTHETIC dataset.

You are given an AnnData object `adata` with:
  - adata.X               : raw gene-expression counts, shape (n_cells, n_genes).
  - adata.obs["batch"]    : categorical batch label (a NUISANCE effect to remove/mix).
  - adata.obs["cell_type"]: HIDDEN at integration time -- it is REMOVED before your function is
                            called, and you MUST NOT rely on it.

The synthetic data contains BOTH genuine biological signal (cell types) AND nuisance batch effects.
GOAL: produce a low-dimensional embedding that MIXES the batches (removes batch differences) while
PRESERVING biological structure (keeps cell types separable).

Write a SINGLE Python function with EXACTLY this signature:

    def eliminate_batch_effect_fn(adata, config):
        ...
        return adata   # an AnnData with adata.obsm["X_emb"] set

HARD REQUIREMENTS (a violation makes the candidate invalid, reward = -inf):
  1. Define a function named exactly `eliminate_batch_effect_fn(adata, config)`.
  2. Return an AnnData object with adata.obsm["X_emb"] = a 2D float array of shape (n_cells, d),
     where n_cells == adata.n_obs (one row per input cell) and 1 <= d <= n_genes.
  3. Do NOT use adata.obs["cell_type"] in any way.
  4. You MAY use adata.obs["batch"] to remove batch effects.
  5. The embedding must be finite (NO NaN/inf).
  6. CPU-only, lightweight, fast. AVOID heavy neural networks (no torch/tensorflow/jax training).
  7. Do NOT download data or call any network/external service.
  8. Use ONLY numpy, scipy, scanpy, and scikit-learn (all available).

RECOMMENDED LIGHTWEIGHT TECHNIQUES (combine/improve; be creative but robust):
  - normalize_total + log1p, then PCA.
  - batch-centering in embedding space: subtract each batch's mean embedding vector.
  - per-batch mean correction / robust scaling of genes before PCA.
  - simple linear batch-effect removal (e.g. regress out batch indicator variables per gene).
  - z-scoring genes; clipping outliers; blends of the above.

REWARD (higher is better): reduced score = bio_score + batch_mixing_score, where
  - bio_score rewards keeping cell types SEPARABLE (biology preserved),
  - batch_mixing_score rewards MIXING the batches (batch effect removed).
  (This is a synthetic smoke-test proxy, NOT the official scIB metric.)

Return ONLY the Python code for `eliminate_batch_effect_fn` (plus any imports it needs).
No explanations, no tests, no __main__ block.
'''


def build_generation_prompt(parent_program: str, parent_score: float | None) -> str:
    if parent_score is not None and math.isfinite(parent_score):
        parent_reward = f"{parent_score:.6f}"
    else:
        parent_reward = "-inf (parent was invalid)"
    spec = _interface_spec()
    return (
        f"{spec}\n"
        f"--- CURRENT (PARENT) CANDIDATE ---\n"
        f"# parent reward (reduced score, higher is better): {parent_reward}\n"
        f"{parent_program}\n"
        f"--- END PARENT CANDIDATE ---\n\n"
        f"Write an IMPROVED `eliminate_batch_effect_fn` that should achieve a HIGHER reward than "
        f"the parent above (better batch mixing AND preserved biology), following the guidance. "
        f"Return ONLY the Python code."
    )


class CandidateRecord:
    __slots__ = (
        "iteration", "candidate_id", "parent_id", "valid", "reward", "score",
        "bio_score", "batch_mixing_score", "embedding_shape", "error",
        "candidate_path", "runtime_s",
    )

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    def as_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}


def _finite_or_neg_inf(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("-inf")
    return v if math.isfinite(v) else float("-inf")


def score_program(
    program: str,
    candidate_id: int,
    out_dir: Path,
    scrna_python: str,
    scorer_script: str,
    timeout_s: int = 300,
) -> dict:
    """Write a candidate program to disk and score it via the scRNA env subprocess.

    Returns a dict with keys: valid, reward, score, bio_score, batch_mixing_score,
    embedding_shape, error, candidate_path, runtime_s. Never raises; any failure ->
    valid=False, reward=-inf.
    """
    candidates_dir = out_dir / "candidates"
    logs_dir = out_dir / "candidate_logs" / f"cand_{candidate_id:03d}"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cand_path = candidates_dir / f"cand_{candidate_id:03d}.py"
    cand_path.write_text(program)

    result = {
        "valid": False,
        "reward": float("-inf"),
        "score": None,
        "bio_score": None,
        "batch_mixing_score": None,
        "embedding_shape": None,
        "error": None,
        "candidate_path": str(cand_path),
        "runtime_s": None,
    }

    cmd = [
        scrna_python, "-u", scorer_script,
        "--candidate", str(cand_path),
        "--out-dir", str(logs_dir),
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        result["error"] = f"TIMEOUT: scorer exceeded {timeout_s}s"
        result["runtime_s"] = round(time.time() - t0, 3)
        (logs_dir / "stdout.txt").write_text("TIMEOUT\n")
        return result
    except Exception as e:  # subprocess could not start
        result["error"] = f"SUBPROCESS_ERROR: {type(e).__name__}: {e}"
        result["runtime_s"] = round(time.time() - t0, 3)
        return result

    result["runtime_s"] = round(time.time() - t0, 3)
    (logs_dir / "stdout.txt").write_text(
        (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    )

    # --- Primary: parse the scorer's JSON file ---
    json_path = logs_dir / "candidate_results.json"
    rec = None
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            results = data.get("results") or []
            if results:
                rec = results[0]
        except Exception as e:
            result["error"] = f"JSON_PARSE_ERROR: {e}"

    # --- Fallback: parse from stdout ---
    if rec is None and proc.stdout:
        m_valid = re.search(r"valid=(True|False)", proc.stdout)
        m_score = re.search(r"score=([-+0-9.einf]+)", proc.stdout)
        if m_valid:
            rec = {
                "valid": (m_valid.group(1) == "True"),
                "score": float(m_score.group(1)) if m_score else None,
                "reward": float(m_score.group(1)) if m_score else None,
                "bio_score": None, "batch_mixing_score": None,
                "embedding_shape": None, "error": "parsed-from-stdout (json missing)",
            }

    if rec is None:
        if not result["error"]:
            tail = (proc.stderr or proc.stdout or "")[-500:]
            result["error"] = f"NO_RESULT (returncode={proc.returncode}): {tail}"
        return result

    valid = bool(rec.get("valid"))
    reward = rec.get("reward")
    if reward is None:
        reward = rec.get("score")
    result.update(
        valid=valid,
        reward=_finite_or_neg_inf(reward) if valid else float("-inf"),
        score=rec.get("score"),
        bio_score=rec.get("bio_score"),
        batch_mixing_score=rec.get("batch_mixing_score"),
        embedding_shape=rec.get("embedding_shape"),
        error=rec.get("error"),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of ERA expansions (Gemini calls). Default 5.")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
                        help="Gemini model. Default: $GEMINI_MODEL or gemini-2.5-flash.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--initial_candidate", default=None,
                        help="Path to a candidate .py to SEED the search (default: PCA baseline).")
    parser.add_argument("--scrna_python", default=DEFAULT_SCRNA_PYTHON)
    parser.add_argument("--scorer_script", default=DEFAULT_SCORER_SCRIPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c_puct", type=float, default=1.0)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    # --- Preconditions ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.\n"
              "  unset GOOGLE_API_KEY\n"
              '  export GEMINI_API_KEY="my_key"\n'
              "then re-run.")
        sys.exit(1)
    if os.environ.get("GOOGLE_API_KEY"):
        print("[!] GOOGLE_API_KEY is also set; the SDK may prefer it. "
              "For a clean run, `unset GOOGLE_API_KEY` first.")
    scrna_py = Path(args.scrna_python)
    if not scrna_py.exists():
        print(f"[!] scRNA env python not found: {scrna_py}")
        sys.exit(1)
    if not Path(args.scorer_script).exists():
        print(f"[!] Scorer script not found: {args.scorer_script}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates").mkdir(exist_ok=True)
    (out_dir / "candidate_logs").mkdir(exist_ok=True)

    # Resolve the initial (seed) candidate.
    if args.initial_candidate:
        initial_code = Path(args.initial_candidate).read_text()
        initial_name = Path(args.initial_candidate).name
    else:
        initial_code = PCA_SEED_CODE
        initial_name = "pca_baseline (built-in)"

    print(f"Out dir:        {out_dir}")
    print(f"Model:          {args.model}")
    print(f"Iterations:     {args.iterations}")
    print(f"Initial seed:   {initial_name}")
    print(f"scRNA python:   {args.scrna_python}")
    print(f"Scorer script:  {args.scorer_script}")
    print(f"Reward:         reduced score (higher is better)\n")

    llm = GeminiLLM(api_key, model_name=args.model)

    records: list[CandidateRecord] = []
    sol_to_id: dict[int, int] = {}
    next_id = {"v": 0}

    # --- 1) Score the initial (seed) candidate ---
    (out_dir / "initial_candidate.py").write_text(initial_code)
    print(f"=== Scoring initial candidate ({initial_name}) ===")
    init_id = next_id["v"]; next_id["v"] += 1
    init_res = score_program(initial_code, init_id, out_dir, args.scrna_python, args.scorer_script)
    initial_solution = futs.Solution(initial_code)
    sol_to_id[id(initial_solution)] = init_id
    initial_score = init_res["reward"]
    records.append(CandidateRecord(
        iteration=0, candidate_id=init_id, parent_id=None,
        valid=init_res["valid"], reward=init_res["reward"], score=init_res["score"],
        bio_score=init_res["bio_score"], batch_mixing_score=init_res["batch_mixing_score"],
        embedding_shape=init_res["embedding_shape"], error=init_res["error"],
        candidate_path=init_res["candidate_path"], runtime_s=init_res["runtime_s"],
    ))
    print(f"  initial: valid={init_res['valid']} reward={initial_score} "
          f"(bio={init_res['bio_score']}, batch_mix={init_res['batch_mixing_score']})\n")
    if not init_res["valid"]:
        print("[!] Initial candidate did not score as valid. Check the scRNA env / scorer. "
              "Aborting before spending Gemini calls.")
        _write_outputs(out_dir, records, args, initial_score, initial_name, aborted=True)
        sys.exit(1)

    # --- 2) Generator ---
    def generate_fn(problem, parent_solution, parent_score):
        parent_id = sol_to_id.get(id(parent_solution))
        prompt = build_generation_prompt(parent_solution.program, parent_score)
        try:
            code = llm.draw_sample(prompt)
        except Exception as e:  # SafeGenerator behaviour: never crash the whole run
            print(f"  [!] Generation failed after retries; recording -inf candidate. "
                  f"({str(e)[:100]})")
            code = "# generation failed - intentionally invalid candidate (no eliminate_batch_effect_fn)\n"
        sol = futs.Solution(code)
        cand_id = next_id["v"]; next_id["v"] += 1
        sol_to_id[id(sol)] = cand_id
        generate_fn._pending = (cand_id, parent_id)  # type: ignore[attr-defined]
        return sol

    # --- 3) Executor ---
    def execute_fn(problem, solution):
        cand_id = sol_to_id.get(id(solution))
        parent_id = None
        pending = getattr(generate_fn, "_pending", None)
        if pending and pending[0] == cand_id:
            parent_id = pending[1]
        res = score_program(solution.program, cand_id, out_dir, args.scrna_python, args.scorer_script)
        records.append(CandidateRecord(
            iteration=cand_id, candidate_id=cand_id, parent_id=parent_id,
            valid=res["valid"], reward=res["reward"], score=res["score"],
            bio_score=res["bio_score"], batch_mixing_score=res["batch_mixing_score"],
            embedding_shape=res["embedding_shape"], error=res["error"],
            candidate_path=res["candidate_path"], runtime_s=res["runtime_s"],
        ))
        status = "valid" if res["valid"] else f"INVALID ({str(res['error'])[:60]})"
        print(f"  [ERA] iter {cand_id}/{args.iterations}: parent={parent_id} "
              f"reward={res['reward']} score={res['score']} [{status}]")
        return res["reward"]

    # --- 4) Run FUTS tree search ---
    problem = futs.Problem(PROBLEM_DESCRIPTION)
    print(f"=== ERA / FUTS tree search ({args.iterations} iterations) ===")
    best_solution, best_score = futs.search(
        problem=problem,
        initial_solution=initial_solution,
        initial_score=initial_score,
        generate_fn=generate_fn,
        execute_fn=execute_fn,
        num_iterations=args.iterations,
        c_puct=args.c_puct,
    )

    (out_dir / "best_candidate.py").write_text(best_solution.program)
    _write_outputs(out_dir, records, args, initial_score, initial_name, best_score=best_score)


def _write_outputs(out_dir, records, args, initial_score, initial_name, best_score=None,
                   aborted=False):
    """Write results.json and progress.csv (higher reward = better)."""
    valid_records = [r for r in records if r.valid]
    invalid_records = [r for r in records if not r.valid]
    rewards = [r.reward for r in records if r.reward is not None and math.isfinite(r.reward)]
    best_reward = max(rewards) if rewards else float("-inf")
    best_rec = None
    for r in records:
        if r.reward is not None and math.isfinite(r.reward) and r.reward == best_reward:
            best_rec = r
            break

    def _safe(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    # progress.csv
    csv_path = out_dir / "progress.csv"
    fields = ["iteration", "candidate_id", "parent_id", "valid", "reward", "score",
              "bio_score", "batch_mixing_score", "embedding_shape", "runtime_s",
              "candidate_path", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = r.as_dict()
            row["reward"] = "" if (row["reward"] is None or not math.isfinite(row["reward"])) else row["reward"]
            w.writerow(row)

    results = {
        "task": "scRNA synthetic batch integration (D2A ERA smoke test)",
        "model": args.model,
        "initial_candidate": initial_name,
        "iterations_requested": args.iterations,
        "reward_definition": "reward = reduced score = bio_score + batch_mixing_score (higher better)",
        "note": "Synthetic smoke test; NOT the official scIB score. Real data/scIB deferred to D2B.",
        "aborted": aborted,
        "initial_reward": _safe(initial_score),
        "best_reward": _safe(best_reward),
        "best_score": (best_rec.score if best_rec else None),
        "best_candidate_id": (best_rec.candidate_id if best_rec else None),
        "num_candidates_total": len(records),
        "num_valid": len(valid_records),
        "num_invalid": len(invalid_records),
        "improvement_over_initial": (
            _safe(best_reward - initial_score)
            if (math.isfinite(best_reward) and math.isfinite(initial_score)) else None
        ),
        "records": [{**r.as_dict(), "reward": _safe(r.reward)} for r in records],
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print("\n=========== scRNA ERA Search Summary (synthetic) ===========")
    print(f"Model:              {args.model}")
    print(f"Initial seed:       {initial_name}")
    print(f"Initial reward:     {initial_score}")
    print(f"Best reward:        {best_reward}")
    print(f"Best candidate id:  {results['best_candidate_id']}")
    print(f"Valid / invalid:    {len(valid_records)} / {len(invalid_records)}")
    print(f"Wrote: {out_dir/'results.json'}")
    print(f"Wrote: {out_dir/'progress.csv'}")
    if not aborted:
        print(f"Wrote: {out_dir/'best_candidate.py'}")
    print("============================================================")


if __name__ == "__main__":
    main()
