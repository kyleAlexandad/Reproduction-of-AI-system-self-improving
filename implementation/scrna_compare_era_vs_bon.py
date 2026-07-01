"""D3C-alt: fair scRNA ERA tree search vs. best-of-N independent sampling (PBMC3k bridge).

This is the scRNA analogue of `gift_eval_compare_era_vs_bon.py`. Both methods use the SAME model,
prompt (`pbmc3k_conservative_v2`), initial PCA-20 seed, scRNA scorer, and reward (= reduced proxy
score, HIGHER is better). They make the SAME number of Gemini calls (N each). The ONLY difference:

  * Method A - ERA tree search: each candidate is generated conditioned on a TREE-SELECTED parent
    (FUTS / PUCT), so it can iteratively refine previous candidates (reuses `futs.search`).
  * Method B - best-of-N: each candidate is generated independently from the SAME initial seed
    prompt, never conditioning on any previously generated candidate.

Framing (see D3B-alt, saved_runs/scrna_d3b_pbmc3k_era_iter10_conservative/): on this PBMC3k bridge
the batch-centered-PCA reference (~1.0701) is the strong simple target and the PCA-20 seed is 1.0275.
D3C-alt is a fair PROCESS comparison under equal budget: which method reaches a higher / closer /
more-stable reward. We report best reward (incl/excl seed), delta vs seed, valid/invalid rate,
duplicate rate, and the best-so-far curve.

ARCHITECTURE: runs in the ERA env; scores every candidate by subprocess into the scRNA env
(reusing `scrna_era_search.score_program`). It NEVER imports scanpy / anndata / scib.

>>> PBMC3k BRIDGE (NOT the paper benchmark): real 10x PBMC3k biology + an ARTIFICIAL injected batch,
>>> scored by a REDUCED Python-only proxy (bio_score + batch_mixing_score). NO official Kaggle HCA
>>> dataset, NO R/kBET, NO official 12-metric scIB, NO 20k-cell run.

============================ SECURITY WARNING ============================
The scorer executes LLM-generated Python directly (no isolation). Toy use only.
=========================================================================

NOTE: do NOT run this without a GEMINI_API_KEY; it makes real Gemini calls (2*N total).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import futs
from llm import GeminiLLM, DEFAULT_MODEL

# Reuse the D3B building blocks (single source of truth).
from scrna_era_search import (
    DEFAULT_SCRNA_PYTHON,
    DEFAULT_SYNTH_SCORER,
    DEFAULT_REALDATA_SCORER,
    PROMPT_VERSIONS,
    default_prompt_version,
    PROBLEM_DESCRIPTION,
    PCA_SEED_CODE,
    PCA_SEED_CODE_20,
    BATCH_CENTERED_SEED_CODE_20,
    build_generation_prompt,
    score_program,
)

# The batch-centered-PCA reference reward on this exact PBMC3k task (the strong simple target).
BATCH_CENTERED_REFERENCE = 1.070069915149361

SENTINEL_INVALID = "# generation failed - intentionally invalid candidate (no eliminate_batch_effect_fn)\n"


def generate_candidate(llm, task, prompt_version, parent_program, parent_score):
    """Draw one candidate from Gemini; on hard failure return a sentinel invalid program."""
    prompt = build_generation_prompt(parent_program, parent_score, task, prompt_version)
    try:
        return llm.draw_sample(prompt)
    except Exception as e:  # SafeGenerator behaviour: never crash the whole run
        print(f"  [!] Generation failed after retries; recording -inf candidate. "
              f"({str(e)[:100]})")
        return SENTINEL_INVALID


def _rec(iteration, cid, parent_id, res):
    return {
        "iteration": iteration,
        "candidate_id": cid,
        "parent_id": parent_id,
        "valid": res["valid"],
        "reward": res["reward"],
        "score": res["score"],
        "bio_score": res["bio_score"],
        "batch_mixing_score": res["batch_mixing_score"],
        "embedding_shape": res["embedding_shape"],
        "error": res["error"],
        "candidate_path": res["candidate_path"],
        "runtime_s": res["runtime_s"],
    }


def run_era(llm, args, out_dir, seed_code, seed_res, scorer_extra_args):
    """Method A: FUTS tree search for N expansions, conditioning on tree-selected parents."""
    scrna_py, scorer = args.scrna_python, args.scorer_script
    cand_dir = out_dir / "era_candidates"
    records = [_rec(0, 0, None, seed_res)]
    sol_to_id = {}
    nid = {"v": 1}

    seed_sol = futs.Solution(seed_code)
    sol_to_id[id(seed_sol)] = 0
    seed_score = seed_res["reward"]

    def generate_fn(problem, parent_solution, parent_score):
        parent_id = sol_to_id.get(id(parent_solution))
        code = generate_candidate(llm, args.task, args.prompt_version,
                                  parent_solution.program, parent_score)
        sol = futs.Solution(code)
        cid = nid["v"]; nid["v"] += 1
        sol_to_id[id(sol)] = cid
        generate_fn._pending = (cid, parent_id)  # type: ignore[attr-defined]
        return sol

    def execute_fn(problem, solution):
        cid = sol_to_id.get(id(solution))
        parent_id = None
        pending = getattr(generate_fn, "_pending", None)
        if pending and pending[0] == cid:
            parent_id = pending[1]
        res = score_program(
            solution.program, cid, out_dir, scrna_py, scorer,
            scorer_extra_args=scorer_extra_args,
            candidates_dir=cand_dir,
            logs_dir=out_dir / "candidate_logs" / f"era_{cid:03d}",
        )
        records.append(_rec(cid, cid, parent_id, res))
        status = "valid" if res["valid"] else f"INVALID ({str(res['error'])[:50]})"
        print(f"  [ERA] {cid}/{args.N}: parent={parent_id} reward={res['reward']} [{status}]")
        return res["reward"]

    print(f"\n=== Method A: ERA tree search (N={args.N}) ===")
    futs.search(
        problem=futs.Problem(PROBLEM_DESCRIPTION),
        initial_solution=seed_sol,
        initial_score=seed_score,
        generate_fn=generate_fn,
        execute_fn=execute_fn,
        num_iterations=args.N,
        c_puct=args.c_puct,
    )
    return records


def run_bon(llm, args, out_dir, seed_code, seed_res, scorer_extra_args):
    """Method B: best-of-N independent sampling, always from the SAME initial seed."""
    scrna_py, scorer = args.scrna_python, args.scorer_script
    cand_dir = out_dir / "bon_candidates"
    records = [_rec(0, 0, None, seed_res)]
    seed_score = seed_res["reward"]

    print(f"\n=== Method B: best-of-N independent sampling (N={args.N}) ===")
    for i in range(1, args.N + 1):
        # Always condition on the identical initial seed (never on prior candidates).
        code = generate_candidate(llm, args.task, args.prompt_version, seed_code, seed_score)
        res = score_program(
            code, i, out_dir, scrna_py, scorer,
            scorer_extra_args=scorer_extra_args,
            candidates_dir=cand_dir,
            logs_dir=out_dir / "candidate_logs" / f"bon_{i:03d}",
        )
        records.append(_rec(i, i, 0, res))
        status = "valid" if res["valid"] else f"INVALID ({str(res['error'])[:50]})"
        print(f"  [BoN] {i}/{args.N}: reward={res['reward']} [{status}]")
    return records


def _norm(program: str) -> str:
    return "\n".join(line.rstrip() for line in (program or "").strip().splitlines())


def method_stats(records, seed_reward, cand_dir):
    """Compute fair-comparison metrics for one method (records[0] is the shared seed).

    HIGHER reward is better here (reduced proxy score), so 'best' = MAX and delta_vs_seed
    POSITIVE means improvement over the seed.
    """
    generated = records[1:]
    valid = [r for r in generated
             if r["valid"] and r["reward"] is not None and math.isfinite(r["reward"])]
    invalid = [r for r in generated if not r["valid"]]
    best_gen = max((r["reward"] for r in valid), default=None)
    best_incl_seed = max([seed_reward] + [r["reward"] for r in valid])
    # best-so-far over generated candidates (skip invalid; carry previous best); higher is better
    best_so_far, cur = [], -math.inf
    for r in generated:
        if r["valid"] and r["reward"] is not None and math.isfinite(r["reward"]) and r["reward"] > cur:
            cur = r["reward"]
        best_so_far.append(cur if math.isfinite(cur) else None)
    # duplicate detection over generated candidate programs (exact, whitespace-normalised)
    programs = []
    for r in generated:
        p = Path(r["candidate_path"]) if r["candidate_path"] else None
        programs.append(_norm(p.read_text()) if (p and p.exists()) else "")
    n_unique_programs = len(set(programs)) if programs else 0
    n_dup_programs = len(programs) - n_unique_programs
    n_unique_reward = len({round(r["reward"], 9) for r in valid}) if valid else 0
    return {
        "n_generated": len(generated),
        "valid": len(valid),
        "invalid": len(invalid),
        "best_reward_incl_seed": best_incl_seed,
        "best_generated_reward_excl_seed": best_gen,
        # delta_vs_seed = best generated reward - seed reward. POSITIVE means the method IMPROVED
        # on the seed (higher reward is better).
        "delta_vs_seed": (best_gen - seed_reward) if best_gen is not None else None,
        "beat_seed": (best_gen is not None and best_gen > seed_reward),
        "reached_reference": (best_gen is not None and best_gen >= BATCH_CENTERED_REFERENCE - 1e-9),
        "n_unique_programs": n_unique_programs,
        "n_duplicate_programs": n_dup_programs,
        "n_unique_valid_reward": n_unique_reward,
        "best_so_far": best_so_far,
    }


def _safe(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=10,
                        help="Candidates per method (Gemini calls per method). Total = 2*N.")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--out_dir", default="saved_runs/scrna_d3c_pbmc3k_era_vs_bon_N10")
    parser.add_argument("--task", default="pbmc3k", choices=["pbmc3k", "synthetic"],
                        help="Which scRNA task to compare on. Default 'pbmc3k' (real bridge).")
    parser.add_argument("--initial_seed", default="pca", choices=["pca", "batch_centered"],
                        help="[pbmc3k] shared built-in seed: 'pca' (default, ~1.0275) or "
                             "'batch_centered' (PCA + per-batch mean centering, the ~1.0701 ref).")
    parser.add_argument("--initial_candidate", default=None,
                        help="Path to a .py seed candidate (overrides --initial_seed).")
    parser.add_argument("--prompt_version", default=None, choices=PROMPT_VERSIONS,
                        help="Generation prompt. Default: per-task "
                             "(pbmc3k -> pbmc3k_conservative_v2, synthetic -> baseline).")
    parser.add_argument("--scrna_python", default=DEFAULT_SCRNA_PYTHON)
    parser.add_argument("--scorer_script", default=None,
                        help="Scorer script. Defaults by --task (pbmc3k vs synthetic scorer).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c_puct", type=float, default=1.0)
    # --- pbmc3k dataset params (forwarded to the real-data scorer; identical for both methods) ---
    parser.add_argument("--source", default="scanpy_pbmc3k", choices=["scanpy_pbmc3k", "pbmc3k"])
    parser.add_argument("--n_cells", type=int, default=500)
    parser.add_argument("--n_batches", type=int, default=3)
    parser.add_argument("--batch_strength", type=float, default=0.8)
    parser.add_argument("--n_hvg", type=int, default=2000)
    parser.add_argument("--leiden_resolution", type=float, default=1.0)
    parser.add_argument("--data_seed", type=int, default=42,
                        help="[pbmc3k] seed for the prepared dataset (subsample + batch injection).")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    task = args.task
    args.prompt_version = args.prompt_version or default_prompt_version(task)
    args.scorer_script = args.scorer_script or (
        DEFAULT_REALDATA_SCORER if task == "pbmc3k" else DEFAULT_SYNTH_SCORER
    )
    scorer_extra_args = None
    dataset_params = None
    if task == "pbmc3k":
        scorer_extra_args = [
            "--source", args.source, "--n_cells", args.n_cells, "--n_batches", args.n_batches,
            "--batch_strength", args.batch_strength, "--n_hvg", args.n_hvg,
            "--leiden_resolution", args.leiden_resolution, "--seed", args.data_seed,
        ]
        dataset_params = {
            "source": args.source, "n_cells": args.n_cells, "n_batches": args.n_batches,
            "batch_strength": args.batch_strength, "n_hvg": args.n_hvg,
            "leiden_resolution": args.leiden_resolution, "data_seed": args.data_seed,
        }

    # --- Preconditions ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.\n  unset GOOGLE_API_KEY\n"
              '  export GEMINI_API_KEY="my_key"\nthen re-run.')
        sys.exit(1)
    if os.environ.get("GOOGLE_API_KEY"):
        print("[!] GOOGLE_API_KEY is also set; the SDK may prefer it. `unset GOOGLE_API_KEY` first.")
    if not Path(args.scrna_python).exists():
        print(f"[!] scRNA env python not found: {args.scrna_python}"); sys.exit(1)
    if not Path(args.scorer_script).exists():
        print(f"[!] Scorer script not found: {args.scorer_script}"); sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("era_candidates", "bon_candidates", "candidate_logs"):
        (out_dir / sub).mkdir(exist_ok=True)

    # Resolve the shared seed candidate (default PCA baseline for EVERY method).
    if args.initial_candidate:
        seed_code = Path(args.initial_candidate).read_text()
        seed_name = Path(args.initial_candidate).name
    elif task == "pbmc3k":
        if args.initial_seed == "batch_centered":
            seed_code, seed_name = BATCH_CENTERED_SEED_CODE_20, "batch_centered_pca20 (built-in, opt-in)"
        else:
            seed_code, seed_name = PCA_SEED_CODE_20, "pca20_baseline (built-in)"
    else:
        seed_code, seed_name = PCA_SEED_CODE, "pca_baseline (built-in)"

    print(f"Task:           {task}")
    print(f"Out dir:        {out_dir}")
    print(f"Model:          {args.model}")
    print(f"N per method:   {args.N}   (total Gemini calls = {2*args.N})")
    print(f"Prompt version: {args.prompt_version}")
    print(f"Shared seed:    {seed_name}")
    print(f"scRNA python:   {args.scrna_python}")
    print(f"Scorer script:  {args.scorer_script}")
    if task == "pbmc3k":
        print(f"Dataset:        {args.source} n_cells={args.n_cells} n_batches={args.n_batches} "
              f"strength={args.batch_strength} data_seed={args.data_seed}")
    print(f"Reward:         reduced score (higher is better)\n")

    llm = GeminiLLM(api_key, model_name=args.model)

    # --- Score the shared seed ONCE (deterministic; reused as the floor for both methods) ---
    (out_dir / "initial_candidate.py").write_text(seed_code)
    print(f"=== Scoring shared seed ({seed_name}) ===")
    seed_res = score_program(
        seed_code, 0, out_dir, args.scrna_python, args.scorer_script,
        scorer_extra_args=scorer_extra_args,
        candidates_dir=out_dir / "era_candidates",
        logs_dir=out_dir / "candidate_logs" / "seed",
    )
    if not seed_res["valid"]:
        print(f"[!] Seed did not score valid: {seed_res['error']}. Aborting before Gemini calls.")
        sys.exit(1)
    seed_reward = seed_res["reward"]
    print(f"  seed reward = {seed_reward} (bio={seed_res['bio_score']}, "
          f"batch_mix={seed_res['batch_mixing_score']})\n")
    # mirror the seed file into bon_candidates for a self-contained layout
    (out_dir / "bon_candidates" / "cand_000.py").write_text(seed_code)

    # --- Run both methods (equal budget) ---
    era_records = run_era(llm, args, out_dir, seed_code, seed_res, scorer_extra_args)
    bon_records = run_bon(llm, args, out_dir, seed_code, seed_res, scorer_extra_args)

    era = method_stats(era_records, seed_reward, out_dir / "era_candidates")
    bon = method_stats(bon_records, seed_reward, out_dir / "bon_candidates")

    # --- Decide the (process) winner: HIGHER best-generated reward wins; tie if within 1e-9 ---
    def cmp_best(a, b):
        av = a["best_generated_reward_excl_seed"]
        bv = b["best_generated_reward_excl_seed"]
        if av is None and bv is None:
            return "tie"
        if av is None:
            return "best-of-N"
        if bv is None:
            return "ERA"
        if abs(av - bv) < 1e-9:
            return "tie"
        return "ERA" if av > bv else "best-of-N"

    winner_excl_seed = cmp_best(era, bon)

    # --- Write per-candidate progress CSVs ---
    def write_progress(path, records):
        fields = ["iteration", "candidate_id", "parent_id", "valid", "reward", "score",
                  "bio_score", "batch_mixing_score", "embedding_shape", "runtime_s",
                  "candidate_path", "error"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in records:
                row = dict(r)
                rw = row.get("reward")
                row["reward"] = "" if (rw is None or (isinstance(rw, float) and not math.isfinite(rw))) else rw
                w.writerow(row)

    write_progress(out_dir / "progress_era.csv", era_records)
    write_progress(out_dir / "progress_bon.csv", bon_records)

    # --- summary.csv (one row per method + a seed row) ---
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_generated", "valid", "invalid",
                    "best_reward_incl_seed", "best_generated_reward_excl_seed",
                    "delta_vs_seed(pos=better)", "beat_seed", "reached_reference_1.0701",
                    "n_unique_programs", "n_duplicate_programs", "n_unique_valid_reward"])
        w.writerow(["seed_baseline", 0, "", "", seed_reward, "", 0.0, "", "", "", "", ""])
        for name, s in (("ERA", era), ("best_of_N", bon)):
            w.writerow([name, s["n_generated"], s["valid"], s["invalid"],
                        s["best_reward_incl_seed"], s["best_generated_reward_excl_seed"],
                        s["delta_vs_seed"], s["beat_seed"], s["reached_reference"],
                        s["n_unique_programs"], s["n_duplicate_programs"],
                        s["n_unique_valid_reward"]])

    # --- results.json ---
    def clean(s):
        return {k: (_safe(v) if not isinstance(v, list) else [_safe(x) for x in v])
                for k, v in s.items()}

    results = {
        "task": task,
        "task_label": ("scRNA PBMC3k real-data batch integration (D3C-alt ERA vs best-of-N)"
                       if task == "pbmc3k"
                       else "scRNA synthetic batch integration (ERA vs best-of-N)"),
        "dataset_params": dataset_params,
        "seed_candidate": seed_name,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "N_per_method": args.N,
        "total_gemini_calls": 2 * args.N,
        "reward_definition": "reward = reduced score = bio_score + batch_mixing_score (higher better)",
        "note": ("PBMC3k real biology + artificial batch; reduced proxy score, NOT official scIB."
                 if task == "pbmc3k"
                 else "Synthetic smoke test; NOT the official scIB score."),
        "seed_baseline_reward": seed_reward,
        "batch_centered_reference_reward": BATCH_CENTERED_REFERENCE,
        "winner_by_best_generated_reward": winner_excl_seed,
        "either_beat_seed": bool(era["beat_seed"] or bon["beat_seed"]),
        "delta_vs_seed_note": "delta_vs_seed = best_generated_reward - seed_reward; POSITIVE = improvement",
        "ERA": clean(era),
        "best_of_N": clean(bon),
        "era_records": [{**r, "reward": _safe(r["reward"])} for r in era_records],
        "bon_records": [{**r, "reward": _safe(r["reward"])} for r in bon_records],
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    # --- Plot best-so-far reward (generated, excl. seed) for both methods ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.step(range(1, len(era["best_so_far"]) + 1), era["best_so_far"], where="post",
            color="#1f77b4", marker="o", markersize=4, linewidth=1.9, label="ERA tree search")
    ax.step(range(1, len(bon["best_so_far"]) + 1), bon["best_so_far"], where="post",
            color="#ff7f0e", marker="s", markersize=4, linewidth=1.9,
            label="Best-of-N independent")
    ax.axhline(seed_reward, color="#7f7f7f", linestyle=":", linewidth=1.3,
               label=f"PCA seed ({seed_reward:.5f})")
    ax.axhline(BATCH_CENTERED_REFERENCE, color="#2ca02c", linestyle="--", linewidth=1.3,
               label=f"batch-centered ref ({BATCH_CENTERED_REFERENCE:.5f})")
    ax.set_title(f"D3C-alt: ERA vs Best-of-N on scRNA PBMC3k ({task})\n"
                 f"best generated-candidate reward so far (N={args.N}, higher is better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Candidate index (generated, excluding seed)")
    ax.set_ylabel("Best-so-far reward (higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "era_vs_bon_reward.png")
    fig.savefig(out_dir / "era_vs_bon_reward.pdf")

    # --- Console summary ---
    print("\n======= scRNA ERA vs Best-of-N Summary =======")
    print(f"Task:               {task}   Model: {args.model}   N/method: {args.N}")
    print(f"Seed baseline reward: {seed_reward:.8f}   batch-centered ref: {BATCH_CENTERED_REFERENCE:.8f}")
    for name, s in (("ERA", era), ("Best-of-N", bon)):
        bg = s["best_generated_reward_excl_seed"]
        d = s["delta_vs_seed"]
        print(f"{name:10s} best(incl seed)={s['best_reward_incl_seed']:.8f}  "
              f"best gen(excl seed)={bg if bg is None else round(bg,8)}  "
              f"delta_vs_seed={d if d is None else format(d,'+.4f')} (pos=better)  "
              f"valid/invalid={s['valid']}/{s['invalid']}  "
              f"reached_ref={s['reached_reference']}  dup_programs={s['n_duplicate_programs']}")
    print(f"Winner (higher best generated reward): {winner_excl_seed}")
    print(f"Either beat seed: {results['either_beat_seed']}")
    print(f"Wrote: {out_dir/'results.json'}, summary.csv, progress_era.csv, progress_bon.csv, "
          f"era_vs_bon_reward.png")
    print("==============================================")


if __name__ == "__main__":
    main()
