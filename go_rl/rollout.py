from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .board import GtpParseError


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 4


class VllmHttpClient:
    def __init__(self, base_url: str, model: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompts: Iterable[str], sampling: SamplingParams) -> List[str]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("requests is required for HTTP inference.") from exc

        payload = {
            "model": self.model,
            "prompt": list(prompts),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "stop": ["\n"],
        }
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        outputs = []
        for choice in data.get("choices", []):
            outputs.append(choice.get("text", "").strip())
        return outputs


def normalize_move(output: str) -> str:
    stripped = output.strip()
    if not stripped:
        raise GtpParseError("Empty model output")
    move = stripped.split()[0]
    return move.upper()
