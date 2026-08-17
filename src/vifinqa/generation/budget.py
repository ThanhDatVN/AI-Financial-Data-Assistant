"""Ask the server how long a prompt is, instead of inferring it.

Recovering from a context refusal never converged. The refusal reports the prompt as "at least"
so many tokens, the corrected retry came back over the limit again, and questions 213 and 442 both
died having asked 5,245 tokens of an 11,140-token prompt in a 16,384 context -- identical figures
for two unrelated questions, which no reading of the arithmetic explained.

Guessing was the mistake. The tokenizer is right there, it costs a round trip with no decoding
behind it, and it answers exactly the question the retry loop was trying to infer.

Shared rather than copied because two callers now need it: the generator sizes each question's
budget against a served model, and the synthetic filter has to drop distractors until the answer
fits. A character-count approximation in the second one would measure a different prompt than the
first, which is the whole failure this module exists to end.
"""

from __future__ import annotations

import httpx


def tokenize_url(base_url: str) -> str:
    """vLLM serves /tokenize beside the OpenAI-compatible routes, not inside them."""
    return base_url.rstrip("/").removesuffix("/v1") + "/tokenize"


def measure_prompt(url: str, model: str, messages: list[dict[str, str]], timeout: float) -> int:
    """How many tokens this prompt costs, or 0 when the server has no tokenizer route."""
    try:
        response = httpx.post(
            url, json={"model": model, "messages": messages}, timeout=min(timeout, 30.0)
        )
        response.raise_for_status()
        return int(response.json()["count"])
    except Exception:  # noqa: BLE001 - measuring is an optimisation, not a requirement
        return 0


def room_for_output(context_limit: int, prompt_tokens: int, *, margin: int = 64) -> int:
    """How many output tokens this prompt can still afford, with room to spare."""
    return context_limit - prompt_tokens - margin
