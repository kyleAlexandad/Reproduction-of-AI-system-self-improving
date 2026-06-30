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
"""Second ERA benchmark: scikit-learn Breast Cancer binary classification.

A self-contained sibling of ``playground_s3e1.py`` used to test whether the ERA
loop generalizes from the California-Housing *regression* demo to a *binary
classification* task. Same structure: an LLM writes/edits a `train_and_predict`
function, the local sandbox executes it, and we score it.

Key differences from playground_s3e1.py:
  * Data comes from ``sklearn.datasets.load_breast_cancer`` (NO external download).
  * Task is binary classification; the target column is ``target`` (0/1).
  * Metric is ROC-AUC. It is already "higher is better" (range 0..1), so the
    scorer returns the AUC directly (no negation, unlike the RMSE demo).

============================ SECURITY WARNING ============================
This uses the local ``sandbox.py``, which is NOT a secure sandbox: it runs
LLM-generated Python code directly on your machine with no isolation. Keep this
task small, local, and toy-level only.
=========================================================================
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

import futs
from sandbox import Sandbox
from llm import GeminiLLM

# All outputs (data split + results) live under one self-contained directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "saved_runs", "breast_cancer_era")
TRAIN_PATH = os.path.join(OUTPUT_DIR, "local_train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "local_test.csv")
TARGET_COL = "target"


def prepare_data():
    """Loads the built-in breast-cancer dataset and writes a local 80/20 split.

    Returns the true test labels (for ROC-AUC scoring). Deterministic.
    """
    print("Preparing local validation split (sklearn breast_cancer)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    df = data.frame  # 30 feature columns + a 'target' column (0/1)

    # Deterministic shuffle + 80/20 split.
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    train_df.to_csv(TRAIN_PATH, index=False)
    # The test CSV given to the model holds features only (no target).
    test_df.drop(TARGET_COL, axis=1).to_csv(TEST_PATH, index=False)

    return test_df[TARGET_COL].values


def get_data_head():
    """A small data preview for the prompt."""
    df = pd.read_csv(TRAIN_PATH)
    return df.head().to_markdown()


# Intentionally weak baseline so ERA has clear room to improve: a shallow
# decision tree, returning probabilities of the positive class for ROC-AUC.
INITIAL_CODE = """
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def train_and_predict(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X = train.drop('target', axis=1)
    y = train['target']

    model = DecisionTreeClassifier(max_depth=1, random_state=0)
    model.fit(X, y)

    # Return probability of the positive class (class 1) for ROC-AUC.
    return model.predict_proba(test)[:, 1]
"""


class BreastCancerProblem(futs.Problem):
    pass


class BreastCancerGenerator:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def __call__(self, problem: futs.Problem, parent_solution: futs.Solution,
                 parent_score: float) -> futs.Solution:
        data_preview = get_data_head()

        prompt = f"""
{problem.description}

Here is a preview of the TRAINING data (train_path):
{data_preview}

=== DATA SCHEMA (READ CAREFULLY) ===
- `train_path` is a CSV that DOES contain the binary label column '{TARGET_COL}' (values 0 or 1).
- `test_path` is a CSV that DOES NOT contain '{TARGET_COL}'. It has the feature columns ONLY.
- The true test labels are held externally by the scorer; you never see them.
- Therefore you MUST NOT call `test.drop('{TARGET_COL}', axis=1)` -- the column is not in the
  test CSV and that raises `KeyError: "['{TARGET_COL}'] not found in axis"`. If you want to be
  safe, only drop with `errors='ignore'`: `test.drop('{TARGET_COL}', axis=1, errors='ignore')`.

The metric is ROC-AUC (Area Under the ROC Curve). Higher is better.

The previous solution had a score (ROC-AUC) of: {parent_score:.5f}
Previous Solution Code:
```python
{parent_solution.program}
```

Please generate a NEW, IMPROVED Python function named `train_and_predict` that:
1. Accepts `train_path` and `test_path` as strings.
2. Trains a binary classification model on the TRAIN data (which has '{TARGET_COL}').
3. Returns the PREDICTED PROBABILITIES for the positive class (class 1) on the TEST data,
   as a numpy array or list -- one probability between 0 and 1 per row, same length as the test CSV.
4. You can use pandas, numpy, scikit-learn.

IMPORTANT: DO NOT use `xgboost` or `lightgbm`.

Use exactly this column handling (the test CSV has NO '{TARGET_COL}' column):
```python
import pandas as pd
import numpy as np
# ... other imports

def train_and_predict(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop('{TARGET_COL}', axis=1)   # train HAS the label column
    y_train = train['{TARGET_COL}']
    X_test = test.copy()                           # test has features ONLY -- do NOT drop '{TARGET_COL}'

    # ... optional feature engineering / scaling: fit on X_train, apply the SAME transform to X_test ...
    # ... build and train your model ...
    model.fit(X_train, y_train)

    # Return probability of class 1
    return model.predict_proba(X_test)[:, 1]
```
Provide the full, runnable code including imports.

IMPORTANT CONSTRAINTS FOR SPEED:
1. DO NOT use GridSearchCV or RandomizedSearchCV.
2. If using RandomForest or Boosting, set `n_estimators` to maximum 50.
3. Keep the model lightweight (execution time limit is 60 seconds).
"""
        code = self.llm.draw_sample(prompt)
        return futs.Solution(code)


class BreastCancerExecutor:
    def __init__(self, sandbox: Sandbox, y_true: np.ndarray):
        self.sandbox = sandbox
        self.y_true = y_true

    def __call__(self, problem: futs.Problem, solution: futs.Solution) -> float:
        with open(TRAIN_PATH, 'r') as f:
            train_csv = f.read()
        with open(TEST_PATH, 'r') as f:
            test_csv = f.read()

        # Inject the data as files into a writable temp dir inside the sandbox.
        setup_code = f"""
import os
import tempfile
tmp_data_dir = os.path.join(tempfile.gettempdir(), 'futs_bc_data')
os.makedirs(tmp_data_dir, exist_ok=True)
train_path = os.path.join(tmp_data_dir, 'train.csv')
test_path = os.path.join(tmp_data_dir, 'test.csv')

with open(train_path, 'w') as f:
    f.write({repr(train_csv)})
with open(test_path, 'w') as f:
    f.write({repr(test_csv)})
"""
        full_program = setup_code + "\n" + solution.program

        wrapper_code = """
def wrapper(unused_arg):
    return train_and_predict(train_path, test_path)
"""
        final_program = full_program + wrapper_code

        result, success = self.sandbox.run(final_program, "wrapper", None)

        if not success or result is None:
            print(f"Execution failed: {result}")
            return float('-inf')  # Invalid solution

        try:
            predictions = np.asarray(result, dtype=float)
            y_true = self.y_true

            if len(predictions) != len(y_true):
                print(f"Shape mismatch: {len(predictions)} vs {len(y_true)}")
                return float('-inf')

            # ROC-AUC: higher is better, so return it directly (no negation).
            score = roc_auc_score(y_true, predictions)
            return float(score)
        except Exception as e:
            print(f"Scoring error: {e}")
            return float('-inf')


def _save_progress_plot(history, path):
    """Optional progress plot (AUC vs iteration). Best-effort."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [h["iteration"] for h in history]
    ys = [h["best_score"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.step(xs, ys, where="post", color="#1f77b4", linewidth=2, marker="o",
            markersize=4, label="ERA best-so-far")
    ax.set_title("ERA Breast-Cancer Search Progress", fontsize=14, fontweight="bold")
    ax.set_xlabel("Search step / iteration")
    ax.set_ylabel("Best score (ROC-AUC, higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    print(f"Progress plot saved to {path}")


def run_experiment(iterations=10):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY")
        return

    y_val = prepare_data()

    llm = GeminiLLM(api_key)
    print(f"Using Gemini model: {llm.model_name}")
    sandbox = Sandbox(timeout_seconds=60)

    problem = BreastCancerProblem(
        "Improve the binary classification model for the sklearn Breast Cancer dataset.")
    initial_solution = futs.Solution(INITIAL_CODE)

    executor = BreastCancerExecutor(sandbox, y_val)
    print("Evaluating initial solution...")
    initial_score = executor(problem, initial_solution)
    print(f"Initial Score (ROC-AUC): {initial_score}")

    generator = BreastCancerGenerator(llm)

    print("Starting Search...")
    history = [{"iteration": 0, "best_score": initial_score}]

    class TrackingExecutor:
        def __init__(self, inner_executor):
            self.inner = inner_executor
            self.best_so_far = initial_score
            self.iteration = 0

        def __call__(self, problem, solution):
            score = self.inner(problem, solution)
            self.iteration += 1
            if score > self.best_so_far:
                self.best_so_far = score
            history.append({"iteration": self.iteration,
                            "best_score": self.best_so_far})
            return score

    tracking_executor = TrackingExecutor(executor)

    best_sol, best_score = initial_solution, initial_score
    try:
        best_sol, best_score = futs.search(
            problem=problem,
            initial_solution=initial_solution,
            initial_score=initial_score,
            generate_fn=generator,
            execute_fn=tracking_executor,
            num_iterations=iterations,
            c_puct=1.0,
        )
    except Exception as e:
        # Save whatever progress we have rather than losing the API spend.
        print(f"[!] Search stopped early: {e}")

    # --- Save outputs under saved_runs/breast_cancer_era/ --------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "progress.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "best_code.py"), "w") as f:
        f.write(best_sol.program)
    with open(os.path.join(OUTPUT_DIR, "best_score.txt"), "w") as f:
        f.write(f"initial_roc_auc: {initial_score}\nbest_roc_auc: {best_score}\n")
    _save_progress_plot(history, os.path.join(OUTPUT_DIR, "progress.png"))

    print(f"\nProgress/best code/score saved to {OUTPUT_DIR}")
    print(f"Initial ROC-AUC: {initial_score:.6f}")
    print(f"Best ROC-AUC:    {best_score:.6f}")
    print("\nBest Code:\n" + best_sol.program)


if __name__ == "__main__":
    run_experiment()
