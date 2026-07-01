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
DEFAULT_SYNTH_SCORER = str(Path(__file__).resolve().parent / "scrna_synthetic_task.py")
DEFAULT_REALDATA_SCORER = str(Path(__file__).resolve().parent / "scrna_realdata_task.py")
DEFAULT_SCORER_SCRIPT = DEFAULT_SYNTH_SCORER  # synthetic default (backward compatible)
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

# 20-component PCA seed used for the pbmc3k task (matches the D3A-alt PCA baseline, ~1.0275).
PCA_SEED_CODE_20 = '''\
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
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    emb = PCA(n_components=n_comps, random_state=0).fit_transform(X)
    adata.obsm["X_emb"] = emb
    return adata
'''

# Optional stronger seed (opt-in via --initial_seed batch_centered): PCA20 + per-batch mean centering
# in embedding space (the ~1.0701 reference). Default seed stays plain PCA so ERA must find this itself.
BATCH_CENTERED_SEED_CODE_20 = '''\
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
    n_comps = int(min(20, X.shape[1] - 1, X.shape[0] - 1))
    emb = PCA(n_components=n_comps, random_state=0).fit_transform(X)
    batch = adata.obs["batch"].to_numpy()
    for b in np.unique(batch):
        mask = batch == b
        emb[mask] -= emb[mask].mean(axis=0, keepdims=True)
    adata.obsm["X_emb"] = emb
    return adata
'''


PROBLEM_DESCRIPTION = (
    "Improve a single-cell RNA-seq BATCH INTEGRATION function on a small synthetic dataset. "
    "The candidate maps raw counts to a low-dimensional embedding (obsm['X_emb']) that removes "
    "batch effects while preserving biological (cell-type) structure. Reward = a reduced proxy "
    "score (higher is better)."
)


PROMPT_VERSIONS = ["baseline", "pbmc3k_conservative_v2"]


def default_prompt_version(task: str) -> str:
    return "pbmc3k_conservative_v2" if task == "pbmc3k" else "baseline"


# v1 ("baseline"): open-ended technique list (kept for A/B comparison).
BASELINE_STRATEGY = '''\
RECOMMENDED LIGHTWEIGHT TECHNIQUES (combine/improve; be creative but robust):
  - normalize_total + log1p, then PCA.
  - batch-centering in embedding space: subtract each batch's mean embedding vector.
  - per-batch mean correction / robust scaling of genes before PCA.
  - simple linear batch-effect removal (e.g. regress out batch indicator variables per gene).
  - z-scoring genes; clipping outliers; blends of the above.

REWARD (higher is better): reduced score = bio_score + batch_mixing_score, where
  - bio_score rewards keeping cell types SEPARABLE (biology preserved),
  - batch_mixing_score rewards MIXING the batches (batch effect removed).'''


# v2 ("pbmc3k_conservative_v2"): hardened after the 5-iter smoke (4/6 invalid from bad ComBat APIs).
PBMC3K_CONSERVATIVE_V2_STRATEGY = '''\
ENVIRONMENT CONSTRAINTS -- read carefully. These EXACT mistakes made most candidates INVALID before:
  * Do NOT use `scanpy.external.pp` or `scanpy.external.pp.combat` -- that module/attribute is NOT
    available here (AttributeError: module 'scanpy.external.pp' has no attribute 'combat').
  * Do NOT use `sc.tl.combat` -- it does NOT exist here (AttributeError: combat).
  * Do NOT use ComBat at all UNLESS you FIRST verify it with `hasattr(sc.pp, "combat")`. Strongly
    prefer to AVOID ComBat entirely.
  * AVOID advanced / external integration APIs (harmony, bbknn, scanorama, mnnpy, combat) -- they are
    not reliably importable in this environment and will crash the candidate.
  * Do NOT pass `n_jobs=-1` to scanpy preprocessing (e.g. `sc.pp.regress_out(..., n_jobs=-1)`); it can
    raise "number sections must be larger than 0". PREFER plain NUMPY per-batch mean subtraction over
    `sc.pp.regress_out`.

USE ONLY these simple, reliable building blocks (all verified to work in this environment):
  1. sc.pp.normalize_total(adata, target_sum=1e4)
  2. sc.pp.log1p(adata)
  3. PCA (sklearn PCA or sc.pp.pca) to build the embedding
  4. subtract each BATCH's MEAN vector in EMBEDDING space, AFTER PCA        <-- most helpful
  5. subtract each BATCH's MEAN per gene in EXPRESSION space, BEFORE PCA
  6. robust scaling / z-scoring of genes (sc.pp.scale(max_value=10) or sklearn StandardScaler)
  7. simple sklearn preprocessing

MEASURED REFERENCE SCORES on THIS exact task (trust them):
  - PCA baseline (normalize_total -> log1p -> PCA):              reward ~= 1.0275
  - PCA THEN subtract per-batch mean in EMBEDDING space:         reward ~= 1.0701  (BEST simple method)
STRONG, RELIABLE DIRECTION: normalize_total -> log1p -> PCA -> then subtract each batch's mean embedding
vector (batch-centering in embedding space). Aim to MATCH or BEAT ~1.0701 with ROBUST code. Small safe
add-ons (also centering per-gene BEFORE PCA, or light scaling) are welcome, but keep EVERYTHING to
numpy / sklearn / basic scanpy so the candidate stays VALID.

REWARD (higher is better): reduced score = bio_score + batch_mixing_score, where
  - bio_score rewards keeping cell types SEPARABLE (biology preserved),
  - batch_mixing_score rewards MIXING the batches (batch effect removed).
  (Reduced proxy score, NOT the official scIB metric.)'''


def _strategy_block(prompt_version: str) -> str:
    if prompt_version == "pbmc3k_conservative_v2":
        return PBMC3K_CONSERVATIVE_V2_STRATEGY
    return BASELINE_STRATEGY


def _interface_spec(task: str = "synthetic", prompt_version: str = "baseline") -> str:
    if task == "pbmc3k":
        intro = (
            "TASK: single-cell RNA-seq BATCH INTEGRATION on a SMALL REAL dataset (10x PBMC3k) with\n"
            "ARTIFICIAL batch effects injected into the counts. The cell-type structure is REAL\n"
            "(proxy labels from clustering the clean data); the batches are synthetic groups with a\n"
            "controlled per-gene nuisance shift added to expression."
        )
        data_line = ("The data has REAL biological signal (cell clusters) AND an injected nuisance "
                     "batch effect.")
    else:
        intro = "TASK: single-cell RNA-seq BATCH INTEGRATION on a small SYNTHETIC dataset."
        data_line = ("The synthetic data contains BOTH genuine biological signal (cell types) AND "
                     "nuisance batch effects.")
    return f'''\
{intro}

You are given an AnnData object `adata` with:
  - adata.X               : raw gene-expression counts, shape (n_cells, n_genes).
  - adata.obs["batch"]    : categorical batch label (a NUISANCE effect to remove/mix).
  - adata.obs["cell_type"]: HIDDEN at integration time -- it is REMOVED before your function is
                            called, and you MUST NOT rely on it.

{data_line}
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

{_strategy_block(prompt_version)}

Return ONLY the Python code for `eliminate_batch_effect_fn` (plus any imports it needs).
No explanations, no tests, no __main__ block.
'''


def build_generation_prompt(parent_program: str, parent_score: float | None,
                            task: str = "synthetic",
                            prompt_version: str = "baseline") -> str:
    if parent_score is not None and math.isfinite(parent_score):
        parent_reward = f"{parent_score:.6f}"
    else:
        parent_reward = "-inf (parent was invalid)"
    spec = _interface_spec(task, prompt_version)
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
    scorer_extra_args=None,
    candidates_dir=None,
    logs_dir=None,
) -> dict:
    """Write a candidate program to disk and score it via the scRNA env subprocess.

    Returns a dict with keys: valid, reward, score, bio_score, batch_mixing_score,
    embedding_shape, error, candidate_path, runtime_s. Never raises; any failure ->
    valid=False, reward=-inf.

    `candidates_dir` / `logs_dir` override where the candidate .py and its scorer logs are
    written (default: <out_dir>/candidates and <out_dir>/candidate_logs/cand_NNN). The compare
    harness uses this to keep ERA and best-of-N candidates in separate folders.
    """
    candidates_dir = Path(candidates_dir) if candidates_dir is not None else (out_dir / "candidates")
    logs_dir = Path(logs_dir) if logs_dir is not None else (out_dir / "candidate_logs" / f"cand_{candidate_id:03d}")
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
    if scorer_extra_args:
        cmd.extend(str(a) for a in scorer_extra_args)
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
    parser.add_argument("--task", default="synthetic", choices=["synthetic", "pbmc3k"],
                        help="Which scRNA task to search. 'synthetic' (default) or 'pbmc3k' (real).")
    parser.add_argument("--initial_candidate", default=None,
                        help="Path to a candidate .py to SEED the search (default: PCA baseline).")
    parser.add_argument("--scrna_python", default=DEFAULT_SCRNA_PYTHON)
    parser.add_argument("--scorer_script", default=None,
                        help="Scorer script. Defaults by --task (synthetic vs pbmc3k scorer).")
    parser.add_argument("--seed", type=int, default=0, help="FUTS / random seed.")
    parser.add_argument("--c_puct", type=float, default=1.0)
    # --- pbmc3k dataset params (passed through to the real-data scorer) ---
    parser.add_argument("--source", default="scanpy_pbmc3k", choices=["scanpy_pbmc3k", "pbmc3k"])
    parser.add_argument("--n_cells", type=int, default=500)
    parser.add_argument("--n_batches", type=int, default=3)
    parser.add_argument("--batch_strength", type=float, default=0.8)
    parser.add_argument("--n_hvg", type=int, default=2000)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)
    parser.add_argument("--data_seed", type=int, default=42,
                        help="[pbmc3k] seed for the prepared dataset (subsample + batch injection).")
    parser.add_argument("--prompt_version", default=None, choices=PROMPT_VERSIONS,
                        help="Generation prompt. Default: 'pbmc3k_conservative_v2' for --task pbmc3k "
                             "(hardened: bans broken ComBat/external APIs, adds reference scores), "
                             "'baseline' otherwise.")
    parser.add_argument("--initial_seed", default="pca", choices=["pca", "batch_centered"],
                        help="[pbmc3k] built-in seed candidate: 'pca' (default) or 'batch_centered' "
                             "(PCA + per-batch mean centering, the ~1.0701 reference).")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    task = args.task
    prompt_version = args.prompt_version or default_prompt_version(task)
    args.prompt_version = prompt_version  # persist the resolved value for results.json
    # Resolve scorer script + per-candidate dataset args by task.
    scorer_script = args.scorer_script or (
        DEFAULT_REALDATA_SCORER if task == "pbmc3k" else DEFAULT_SYNTH_SCORER
    )
    scorer_extra_args = None
    if task == "pbmc3k":
        scorer_extra_args = [
            "--source", args.source, "--n_cells", args.n_cells, "--n_batches", args.n_batches,
            "--batch_strength", args.batch_strength, "--n_hvg", args.n_hvg,
            "--leiden_resolution", args.leiden_resolution, "--seed", args.data_seed,
        ]

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
    if not Path(scorer_script).exists():
        print(f"[!] Scorer script not found: {scorer_script}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates").mkdir(exist_ok=True)
    (out_dir / "candidate_logs").mkdir(exist_ok=True)

    # Resolve the initial (seed) candidate. Default is plain PCA for EVERY task; the pbmc3k main
    # experiment must start from PCA so ERA has to discover batch centering itself.
    if args.initial_candidate:
        initial_code = Path(args.initial_candidate).read_text()
        initial_name = Path(args.initial_candidate).name
    elif task == "pbmc3k":
        if args.initial_seed == "batch_centered":
            initial_code = BATCH_CENTERED_SEED_CODE_20
            initial_name = "batch_centered_pca20 (built-in, opt-in)"
        else:
            initial_code = PCA_SEED_CODE_20
            initial_name = "pca20_baseline (built-in)"
    else:
        initial_code = PCA_SEED_CODE
        initial_name = "pca_baseline (built-in)"

    print(f"Task:           {task}")
    print(f"Out dir:        {out_dir}")
    print(f"Model:          {args.model}")
    print(f"Iterations:     {args.iterations}")
    print(f"Prompt version: {prompt_version}")
    print(f"Initial seed:   {initial_name}")
    print(f"scRNA python:   {args.scrna_python}")
    print(f"Scorer script:  {scorer_script}")
    if task == "pbmc3k":
        print(f"Dataset:        {args.source} n_cells={args.n_cells} n_batches={args.n_batches} "
              f"strength={args.batch_strength} data_seed={args.data_seed}")
    print(f"Reward:         reduced score (higher is better)\n")

    llm = GeminiLLM(api_key, model_name=args.model)

    records: list[CandidateRecord] = []
    sol_to_id: dict[int, int] = {}
    next_id = {"v": 0}

    # --- 1) Score the initial (seed) candidate ---
    (out_dir / "initial_candidate.py").write_text(initial_code)
    print(f"=== Scoring initial candidate ({initial_name}) ===")
    init_id = next_id["v"]; next_id["v"] += 1
    init_res = score_program(initial_code, init_id, out_dir, args.scrna_python, scorer_script,
                             scorer_extra_args=scorer_extra_args)
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
        prompt = build_generation_prompt(parent_solution.program, parent_score, task, prompt_version)
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
        res = score_program(solution.program, cand_id, out_dir, args.scrna_python, scorer_script,
                            scorer_extra_args=scorer_extra_args)
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

    task = getattr(args, "task", "synthetic")
    task_label = ("scRNA PBMC3k real-data batch integration (D3B-alt ERA smoke)"
                  if task == "pbmc3k"
                  else "scRNA synthetic batch integration (D2A ERA smoke test)")
    dataset_params = None
    if task == "pbmc3k":
        dataset_params = {
            "source": args.source, "n_cells": args.n_cells, "n_batches": args.n_batches,
            "batch_strength": args.batch_strength, "n_hvg": args.n_hvg,
            "leiden_resolution": args.leiden_resolution, "data_seed": args.data_seed,
        }
    results = {
        "task": task,
        "task_label": task_label,
        "dataset_params": dataset_params,
        "prompt_version": getattr(args, "prompt_version", None),
        "model": args.model,
        "initial_candidate": initial_name,
        "iterations_requested": args.iterations,
        "reward_definition": "reward = reduced score = bio_score + batch_mixing_score (higher better)",
        "note": ("PBMC3k real biology + artificial batch; reduced proxy score, NOT official scIB."
                 if task == "pbmc3k"
                 else "Synthetic smoke test; NOT the official scIB score."),
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

    print(f"\n=========== scRNA ERA Search Summary ({task}) ===========")
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
