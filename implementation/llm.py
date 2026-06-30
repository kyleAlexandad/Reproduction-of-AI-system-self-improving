from google import genai
import os
import time
import random
import re
from typing import Protocol

# Default model. Can be overridden without editing code via the GEMINI_MODEL
# environment variable, e.g. when gemini-2.5-flash is overloaded (503):
#   export GEMINI_MODEL=gemini-2.5-flash-lite
DEFAULT_MODEL = "gemini-2.5-flash"

class LLM(Protocol):
    def draw_sample(self, prompt: str) -> str:
        ...

class GeminiLLM:
    def __init__(self, api_key: str, model_name: str | None = None):
        self.client = genai.Client(api_key=api_key, vertexai=False)
        # Explicit arg wins; otherwise honour GEMINI_MODEL env; otherwise default.
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    def draw_sample(self, prompt: str) -> str:
        full_prompt = f"""
You are an expert Data Scientist and Python programmer.
Your task is to write Python code to solve a machine learning problem.
Return ONLY the python code.

--- BEGIN PROMPT ---
{prompt}
--- END PROMPT ---
"""
        max_retries = 5
        base_delay = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt
                )
                content = response.text
                
                # Clean up markdown code blocks if present
                content = re.sub(r'^```python\n', '', content, flags=re.MULTILINE)
                content = re.sub(r'^```\n', '', content, flags=re.MULTILINE)
                content = re.sub(r'\n```$', '', content, flags=re.MULTILINE)
                
                return content
            except Exception as e:
                # Retry on transient errors: rate limits (429) AND temporary
                # server-side issues (500/502/503, UNAVAILABLE, overloaded,
                # internal, deadline). A momentary 503 should not crash a whole
                # multi-call experiment.
                msg = str(e)
                transient = any(
                    tok in msg
                    for tok in (
                        "429", "500", "502", "503",
                        "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED",
                        "overloaded", "high demand",
                    )
                )
                if transient and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"  [!] Transient API error, retrying in {delay:.1f}s... "
                          f"({msg[:80]})")
                    time.sleep(delay)
                else:
                    print(f"Gemini API Error: {e}")
                    raise e