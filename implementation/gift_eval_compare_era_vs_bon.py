# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""C4: fair GIFT-Eval ERA tree search vs. best-of-N independent sampling (m4_weekly/W/short).

Both methods use the SAME model, prompt (conservative_v2), initial naive seed, GIFT-Eval scorer,
and reward (= -MASE). They make the SAME number of Gemini calls (N each). The ONLY difference:

  * Method A - ERA tree search: each candidate is generated conditioned on a TREE-SELECTED parent
    (FUTS/PUCT), so it can iteratively refine previous candidates (reuses `futs.search`).
  * Method B - best-of-N: each candidate is generated independently from the SAME initial naive
    seed prompt, never conditioning on any previously generated candidate.

Framing (see saved_runs/gift_eval_c3_summary/README_C3_FINAL.md): naive is near-optimal on this
subset, so C4 is NOT a "beat naive" contest. It is a fair PROCESS comparison: which method gets a
better/closer/more-stable result under equal budget. We report best MASE (incl/excl seed), distance
to naive, valid/invalid rate, duplicate rate, and the best-so-far curve.

ARCHITECTURE: runs in the ERA env; scores every candidate by subprocess to the GIFT-Eval venv
(reusing `gift_eval_era_search.score_program`). Never imports gluonts/gift_eval into the ERA env.

============================ SECURITY WARNING ============================
The C2 scorer executes LLM-generated Python directly (no isolation). Toy use only.
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
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import futs
from llm import GeminiLLM, DEFAULT_MODEL

# Reuse the C3 building blocks (single source of truth).
from gift_eval_era_search import (
    DATASET_CONFIG,
    PREDICTION_LENGTH,
    DEFAULT_GIFT_EVAL_PYTHON,
    DEFAULT_SCORER_SCRIPT,
    INITIAL_CANDIDATE_CODE,
    PROBLEM_DESCRIPTION,
    PROMPT_SPECS,
    build_generation_prompt,
    score_program,
)

SENTINEL_INVALID = "# generation failed - intentionally invalid candidate (no forecast)\n"


def generate_candidate(llm, prompt_version, parent_program, parent_score):
    """Draw one candidate from Gemini; on hard failure return a sentinel invalid program."""
    prompt = build_generation_prompt(prompt_version, parent_program, parent_score)
    try:
        return llm.draw_sample(prompt)
    except Exception as e:  # SafeGenerator behaviour
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
        "MASE": res["MASE"],
        "CRPS": res["CRPS"],
        "RMSE": res["RMSE"],
        "error": res["error"],
        "candidate_path": res["candidate_path"],
        "runtime_s": res["runtime_s"],
    }


def run_era(llm, args, out_dir, seed_code, seed_res):
    """Method A: FUTS tree search for N expansions, conditioning on tree-selected parents."""
    gift_py, scorer = args.gift_eval_python, args.scorer_script
    cand_dir = out_dir / "era_candidates"
    records = [_rec(0, 0, None, seed_res)]
    sol_to_id = {}
    nid = {"v": 1}

    seed_sol = futs.Solution(seed_code)
    sol_to_id[id(seed_sol)] = 0
    seed_score = seed_res["reward"]

    def generate_fn(problem, parent_solution, parent_score):
        parent_id = sol_to_id.get(id(parent_solution))
        code = generate_candidate(llm, args.prompt_version, parent_solution.program, parent_score)
        sol = futs.Solution(code)
        cid = nid["v"]; nid["v"] += 1
        sol_to_id[id(sol)] = cid
        generate_fn._pending = (cid, parent_id)
        return sol

    def execute_fn(problem, solution):
        cid = sol_to_id.get(id(solution))
        parent_id = None
        pending = getattr(generate_fn, "_pending", None)
        if pending and pending[0] == cid:
            parent_id = pending[1]
        res = score_program(
            solution.program, cid, out_dir, gift_py, scorer,
            candidates_dir=cand_dir,
            logs_dir=out_dir / "candidate_logs" / f"era_{cid:03d}",
        )
        records.append(_rec(cid, cid, parent_id, res))
        status = "valid" if res["valid"] else f"INVALID ({str(res['error'])[:50]})"
        print(f"  [ERA] {cid}/{args.N}: parent={parent_id} MASE={res['MASE']} [{status}]")
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


def run_bon(llm, args, out_dir, seed_code, seed_res):
    """Method B: best-of-N independent sampling, always from the SAME initial naive seed."""
    gift_py, scorer = args.gift_eval_python, args.scorer_script
    cand_dir = out_dir / "bon_candidates"
    records = [_rec(0, 0, None, seed_res)]
    seed_score = seed_res["reward"]

    print(f"\n=== Method B: best-of-N independent sampling (N={args.N}) ===")
    for i in range(1, args.N + 1):
        # Always condition on the identical initial naive seed (never on prior candidates).
        code = generate_candidate(llm, args.prompt_version, seed_code, seed_score)
        res = score_program(
            code, i, out_dir, gift_py, scorer,
            candidates_dir=cand_dir,
            logs_dir=out_dir / "candidate_logs" / f"bon_{i:03d}",
        )
        records.append(_rec(i, i, 0, res))
        status = "valid" if res["valid"] else f"INVALID ({str(res['error'])[:50]})"
        print(f"  [BoN] {i}/{args.N}: MASE={res['MASE']} [{status}]")
    return records


def _norm(program: str) -> str:
    return "\n".join(line.rstrip() for line in (program or "").strip().splitlines())


def method_stats(records, naive_mase, cand_dir):
    """Compute fair-comparison metrics for one method's records (records[0] is the shared seed)."""
    generated = records[1:]
    valid = [r for r in generated if r["valid"] and r["MASE"] is not None]
    invalid = [r for r in generated if not r["valid"]]
    best_gen = min((r["MASE"] for r in valid), default=None)
    best_incl_seed = min([naive_mase] + [r["MASE"] for r in valid])
    # best-so-far over generated candidates (skip invalid; carry previous best)
    best_so_far, cur = [], math.inf
    for r in generated:
        if r["valid"] and r["MASE"] is not None and r["MASE"] < cur:
            cur = r["MASE"]
        best_so_far.append(cur if math.isfinite(cur) else None)
    # duplicate detection over generated candidate programs (exact, whitespace-normalised)
    programs = []
    for r in generated:
        p = Path(r["candidate_path"])
        programs.append(_norm(p.read_text()) if p.exists() else "")
    n_unique_programs = len(set(programs)) if programs else 0
    n_dup_programs = len(programs) - n_unique_programs
    n_unique_mase = len({round(r["MASE"], 9) for r in valid}) if valid else 0
    return {
        "n_generated": len(generated),
        "valid": len(valid),
        "invalid": len(invalid),
        "best_MASE_incl_seed": best_incl_seed,
        "best_generated_MASE_excl_seed": best_gen,
        "distance_to_naive_excl_seed": (best_gen - naive_mase) if best_gen is not None else None,
        "beat_naive": (best_gen is not None and best_gen < naive_mase),
        "n_unique_programs": n_unique_programs,
        "n_duplicate_programs": n_dup_programs,
        "n_unique_valid_MASE": n_unique_mase,
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
    parser.add_argument("--out_dir", default="saved_runs/gift_eval_c4_era_vs_bon_N10")
    parser.add_argument("--prompt_version", default="conservative_v2",
                        choices=sorted(PROMPT_SPECS.keys()))
    parser.add_argument("--gift_eval_python", default=DEFAULT_GIFT_EVAL_PYTHON)
    parser.add_argument("--scorer_script", default=DEFAULT_SCORER_SCRIPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c_puct", type=float, default=1.0)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.\n  unset GOOGLE_API_KEY\n"
              '  export GEMINI_API_KEY="my_key"\nthen re-run.')
        sys.exit(1)
    if os.environ.get("GOOGLE_API_KEY"):
        print("[!] GOOGLE_API_KEY is also set; the SDK may prefer it. `unset GOOGLE_API_KEY` first.")
    if not Path(args.gift_eval_python).exists():
        print(f"[!] GIFT-Eval python not found: {args.gift_eval_python}"); sys.exit(1)
    if not Path(args.scorer_script).exists():
        print(f"[!] Scorer script not found: {args.scorer_script}"); sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("era_candidates", "bon_candidates", "candidate_logs"):
        (out_dir / sub).mkdir(exist_ok=True)

    print(f"Out dir:        {out_dir}")
    print(f"Model:          {args.model}")
    print(f"N per method:   {args.N}   (total Gemini calls = {2*args.N})")
    print(f"Prompt version: {args.prompt_version}")
    print(f"Reward:         -MASE (higher is better)\n")

    llm = GeminiLLM(api_key, model_name=args.model)

    # --- Score the shared naive seed ONCE (deterministic; reused as the floor for both methods) ---
    seed_code = INITIAL_CANDIDATE_CODE
    (out_dir / "initial_candidate.py").write_text(seed_code)
    print("=== Scoring shared naive seed ===")
    seed_res = score_program(
        seed_code, 0, out_dir, args.gift_eval_python, args.scorer_script,
        candidates_dir=out_dir / "era_candidates",
        logs_dir=out_dir / "candidate_logs" / "seed",
    )
    if not seed_res["valid"]:
        print(f"[!] Seed did not score valid: {seed_res['error']}. Aborting before Gemini calls.")
        sys.exit(1)
    naive_mase = seed_res["MASE"]
    print(f"  naive seed MASE = {naive_mase}\n")
    # mirror the seed file into bon_candidates for a self-contained layout
    (out_dir / "bon_candidates" / "cand_000.py").write_text(seed_code)

    # --- Run both methods (equal budget) ---
    era_records = run_era(llm, args, out_dir, seed_code, seed_res)
    bon_records = run_bon(llm, args, out_dir, seed_code, seed_res)

    era = method_stats(era_records, naive_mase, out_dir / "era_candidates")
    bon = method_stats(bon_records, naive_mase, out_dir / "bon_candidates")

    # --- Decide the (process) winner: lower best-generated MASE wins; tie if within 1e-9 ---
    def cmp_best(a, b):
        av = a["best_generated_MASE_excl_seed"]
        bv = b["best_generated_MASE_excl_seed"]
        if av is None and bv is None:
            return "tie"
        if av is None:
            return "best-of-N"
        if bv is None:
            return "ERA"
        if abs(av - bv) < 1e-9:
            return "tie"
        return "ERA" if av < bv else "best-of-N"

    winner_excl_seed = cmp_best(era, bon)

    # --- Write per-candidate progress CSVs ---
    def write_progress(path, records):
        fields = ["iteration", "candidate_id", "parent_id", "valid", "reward", "MASE",
                  "CRPS", "RMSE", "runtime_s", "candidate_path", "error"]
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

    # --- summary.csv (one row per method + a naive row) ---
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_generated", "valid", "invalid",
                    "best_MASE_incl_seed", "best_generated_MASE_excl_seed",
                    "distance_to_naive_excl_seed", "beat_naive",
                    "n_unique_programs", "n_duplicate_programs", "n_unique_valid_MASE"])
        w.writerow(["naive_baseline", 0, "", "", naive_mase, "", 0.0, "", "", "", ""])
        for name, s in (("ERA", era), ("best_of_N", bon)):
            w.writerow([name, s["n_generated"], s["valid"], s["invalid"],
                        s["best_MASE_incl_seed"], s["best_generated_MASE_excl_seed"],
                        s["distance_to_naive_excl_seed"], s["beat_naive"],
                        s["n_unique_programs"], s["n_duplicate_programs"],
                        s["n_unique_valid_MASE"]])

    # --- results.json ---
    def clean(s):
        return {k: (_safe(v) if not isinstance(v, list) else [_safe(x) for x in v])
                for k, v in s.items()}

    results = {
        "dataset": DATASET_CONFIG,
        "prediction_length": PREDICTION_LENGTH,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "N_per_method": args.N,
        "total_gemini_calls": 2 * args.N,
        "reward_definition": "reward = -MASE (lower MASE -> higher reward)",
        "naive_baseline_MASE": naive_mase,
        "winner_by_best_generated_MASE": winner_excl_seed,
        "either_beat_naive": bool(era["beat_naive"] or bon["beat_naive"]),
        "ERA": clean(era),
        "best_of_N": clean(bon),
        "era_records": [{**r, "reward": _safe(r["reward"])} for r in era_records],
        "bon_records": [{**r, "reward": _safe(r["reward"])} for r in bon_records],
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    # --- Plot best-so-far MASE (generated, excl. seed) for both methods ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.step(range(1, len(era["best_so_far"]) + 1), era["best_so_far"], where="post",
            color="#1f77b4", marker="o", markersize=4, linewidth=1.9, label="ERA tree search")
    ax.step(range(1, len(bon["best_so_far"]) + 1), bon["best_so_far"], where="post",
            color="#ff7f0e", marker="s", markersize=4, linewidth=1.9,
            label="Best-of-N independent")
    ax.axhline(naive_mase, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"naive baseline (MASE {naive_mase:.5f})")
    ax.set_title(f"C4: ERA vs Best-of-N on GIFT-Eval {DATASET_CONFIG}\n"
                 f"best generated-candidate MASE so far (N={args.N}, lower is better)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Candidate index (generated, excluding seed)")
    ax.set_ylabel("Best-so-far MASE (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "era_vs_bon_mase.png")
    fig.savefig(out_dir / "era_vs_bon_mase.pdf")

    # --- Console summary ---
    print("\n================ C4 SUMMARY ================")
    print(f"Dataset:            {DATASET_CONFIG}   Model: {args.model}   N/method: {args.N}")
    print(f"Naive baseline MASE: {naive_mase:.8f}")
    for name, s in (("ERA", era), ("Best-of-N", bon)):
        bg = s["best_generated_MASE_excl_seed"]
        d = s["distance_to_naive_excl_seed"]
        print(f"{name:10s} best(incl seed)={s['best_MASE_incl_seed']:.8f}  "
              f"best gen(excl seed)={bg if bg is None else round(bg,8)}  "
              f"dist_to_naive={d if d is None else format(d,'.2e')}  "
              f"valid/invalid={s['valid']}/{s['invalid']}  "
              f"dup_programs={s['n_duplicate_programs']}")
    print(f"Winner (lower best generated MASE): {winner_excl_seed}")
    print(f"Either beat naive: {results['either_beat_naive']}")
    print(f"Wrote: {out_dir/'results.json'}, summary.csv, progress_era.csv, progress_bon.csv, "
          f"era_vs_bon_mase.png")
    print("============================================")


if __name__ == "__main__":
    main()
