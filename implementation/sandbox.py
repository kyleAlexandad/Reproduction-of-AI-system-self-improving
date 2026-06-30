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
import base64
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Any


class Sandbox:
    """Minimal local executor for the ERA playground demo.

    WARNING: This is not a secure sandbox. It runs generated Python code locally.
    Use only for trusted toy experiments. For serious reproduction, use Docker,
    firejail, a VM, or another isolated execution environment.
    """

    def __init__(self, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        program: str,
        function_to_run: str,
        test_input: Any = None,
        timeout_seconds: int | None = None,
    ) -> tuple[Any, bool]:
        timeout = timeout_seconds or self.timeout_seconds

        encoded_input = base64.b64encode(pickle.dumps(test_input)).decode("utf-8")

        runner = f"""
{program}

if __name__ == "__main__":
    import base64
    import pickle
    import sys
    import traceback

    try:
        _input = pickle.loads(base64.b64decode("{encoded_input}"))
        _result = {function_to_run}(_input)
        _payload = base64.b64encode(pickle.dumps(_result)).decode("utf-8")
        print("RESULT_B64:" + _payload)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "candidate.py")

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(runner)

            try:
                proc = subprocess.run(
                    [sys.executable, script_path],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return "Execution timed out", False

            if proc.returncode != 0:
                return proc.stderr or proc.stdout, False

            for line in proc.stdout.splitlines():
                if line.startswith("RESULT_B64:"):
                    payload = line[len("RESULT_B64:") :]
                    result = pickle.loads(base64.b64decode(payload))
                    return result, True

            return "No RESULT_B64 found in stdout:\\n" + proc.stdout, False
