"""Small direct HTTP client and strict output parser for local Ollama."""

from __future__ import annotations

import http.client
import json
import sys
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from llm_fact_checker.prompts import SYSTEM_PROMPT, build_user_prompt, parse_prediction


class OllamaResponseError(Exception):
    """Raised when Ollama answers, but the reply is not usable (no message, or not JSON)."""


class OllamaRequestError(ValueError):
    """Raised when Ollama rejects a request (HTTP 4xx). Retrying will not help."""


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            if 400 <= error.code < 500:
                raise OllamaRequestError(
                    f"Ollama rejected the request ({path}): HTTP {error.code}"
                ) from error
            raise RuntimeError(f"Ollama request failed ({path}): {error}") from error
        except json.JSONDecodeError as error:
            raise OllamaResponseError(f"Ollama returned invalid JSON ({path})") from error
        except (OSError, http.client.HTTPException) as error:
            # URLError, timeouts, and connections dropped mid-response are all transient
            raise RuntimeError(f"Ollama request failed ({path}): {error}") from error

    def check(self, model: str) -> dict[str, Any]:
        try:
            version = self._request("/api/version")
            tags = self._request("/api/tags")
        except RuntimeError as error:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. Start Ollama and retry. {error}"
            ) from error
        model_info = next(
            (item for item in tags.get("models", []) if item.get("name") == model), None
        )
        if model_info is None:
            available = sorted(item.get("name", "") for item in tags.get("models", []))
            raise RuntimeError(
                f"Required Ollama model {model!r} is not installed. Available models: {available}. "
                "This pipeline will not pull models automatically."
            )
        return {"version": version.get("version"), "model": model_info}

    def chat(
        self,
        model: str,
        claim: str,
        evidence: list[dict[str, Any]],
        seed: int = 42,
        temperature: float = 0,
        retries: int = 2,
    ) -> tuple[str, float, dict[str, Any]]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(claim, evidence)},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": temperature, "seed": seed},
        }
        started = time.perf_counter()
        for attempt in range(retries + 1):
            try:
                response = self._request("/api/chat", payload)
                content = response.get("message", {}).get("content")
                if not isinstance(content, str):
                    raise OllamaResponseError("Ollama response did not contain message.content")
                metadata = {
                    key: response[key]
                    for key in (
                        "done_reason",
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "eval_count",
                    )
                    if key in response
                }
                return content, time.perf_counter() - started, metadata
            except RuntimeError:
                if attempt == retries:
                    raise
                time.sleep(1 << attempt)
        raise AssertionError("unreachable")


def infer_cases(
    client: OllamaClient,
    model: str,
    cases: list[dict[str, Any]],
    seed: int,
    temperature: float,
    progress_prefix: str = "Inference",
    id_field: str = "fever_id",
    id_label: str = "FEVER",
) -> list[dict[str, Any]]:
    """Run cases sequentially and preserve every response or failure."""
    results = []
    for index, case in enumerate(cases, 1):
        print(
            f"{progress_prefix} {index}/{len(cases)}: {id_label} {case[id_field]}",
            file=sys.stderr,
        )
        raw_output = None
        parsed = None
        latency = None
        response_metadata = {}
        error = None
        try:
            raw_output, latency, response_metadata = client.chat(
                model,
                case["claim"],
                case["evidence"],
                seed=seed,
                temperature=temperature,
            )
            parsed = parse_prediction(raw_output)
        except (RuntimeError, OllamaResponseError, OllamaRequestError) as caught:
            error = str(caught)
        results.append(
            {
                id_field: case[id_field],
                "claim": case["claim"],
                "evidence": case["evidence"],
                "gold_label": case["gold_label"],
                "raw_model_output": raw_output,
                "parsed_prediction": parsed,
                "correct": parsed == case["gold_label"],
                "valid_format": parsed is not None,
                "inference_latency_seconds": latency,
                "inference_error": error,
                "ollama_response_metadata": response_metadata,
            }
        )
    return results
