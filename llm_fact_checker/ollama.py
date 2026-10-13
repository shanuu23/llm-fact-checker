"""Small direct HTTP client and strict output parser for local Ollama."""

from __future__ import annotations

from collections.abc import Callable
import http.client
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from llm_fact_checker.common import write_jsonl
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
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run cases sequentially and preserve every response or failure.

    ``on_result`` is called with each row as soon as it exists, so a caller can save progress
    instead of waiting for the whole run.
    """
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
        row = {
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
        if "flags" in case:  # audit flags travel with the case so results can be split by them
            row["flags"] = case["flags"]
        results.append(row)
        if on_result is not None:
            on_result(row)
    return results


def infer_cases_resumable(
    client: OllamaClient,
    model: str,
    cases: list[dict[str, Any]],
    seed: int,
    temperature: float,
    partial_path: Path,
    *,
    resume: bool,
    progress_prefix: str = "Inference",
    id_field: str = "fever_id",
    id_label: str = "FEVER",
) -> list[dict[str, Any]]:
    """Like ``infer_cases``, but every row is appended to ``partial_path`` as it arrives.

    With ``resume`` the rows already saved are kept, cases that failed are asked again, and only
    the missing cases are sent to the model. Without it an earlier partial file is discarded.
    Rows come back in the order of ``cases``. The caller deletes ``partial_path`` once it has
    written its final file.
    """
    saved: dict[Any, dict[str, Any]] = {}
    if partial_path.exists():
        if resume:
            lines = partial_path.read_text(encoding="utf-8").splitlines()
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if number == len(lines):
                        break  # the write was cut off by the crash; that case is asked again
                    raise ValueError(f"Invalid JSON in {partial_path} at line {number}") from None
                saved[row[id_field]] = row  # a later row for the same case wins
            write_jsonl(partial_path, saved.values())  # drop a cut-off line before appending
        else:
            partial_path.unlink()
    todo = [
        case
        for case in cases
        if case[id_field] not in saved or saved[case[id_field]]["inference_error"]
    ]
    if saved:
        print(
            f"{progress_prefix}: resuming, {len(cases) - len(todo)} of {len(cases)} "
            "cases already answered",
            file=sys.stderr,
        )
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as sink:

        def save(row: dict[str, Any]) -> None:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()

        new_rows = infer_cases(
            client,
            model,
            todo,
            seed,
            temperature,
            progress_prefix=progress_prefix,
            id_field=id_field,
            id_label=id_label,
            on_result=save,
        )
    saved.update({row[id_field]: row for row in new_rows})
    return [saved[case[id_field]] for case in cases]
