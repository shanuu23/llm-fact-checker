"""Local model access and deterministic workloads. No network calls on import."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
from pathlib import Path
import time

from llm_fact_checker.common import file_sha256, read_jsonl
from llm_fact_checker.data_prep.vitaminc import (
    identify_contrastive_pairs,
    load_vitaminc_test,
    validate_pairs,
)
from llm_fact_checker.experiments.consistency import RUN_SEEDS, TEMPERATURE
from llm_fact_checker.experiments.robustness import build_perturbations
from llm_fact_checker.experiments.sensitivity import _inference_cases
from llm_fact_checker.ollama import OllamaClient, OllamaRequestError

ROOT = Path(__file__).resolve().parents[1]
TESTS = {
    "baseline": "Correctness",
    "consistency": "Consistency",
    "robustness": "Robustness",
    "sensitivity": "Sensitivity",
}
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
FEVER_SAMPLE = ROOT / "data/fever/processed/fever_baseline_150.jsonl"
FEVER_FULL = ROOT / "data/fever/full/fever_dev_full.jsonl"
PAIRS_SAMPLE = ROOT / "data/vitaminc/processed/vitaminc_sensitivity_100_pairs.jsonl"
VITAMINC_TEST = ROOT / "data/vitaminc/raw/vitaminc/test.jsonl"
INPUT_HASHES = {
    FEVER_SAMPLE: "6bc7cf910cb0b31186e5fc397ebc7cefab2de1ee65d29ae601e369d8758344cb",
    FEVER_FULL: "46d03c136a5774811e00a92c67c9ad84f78b1daad981bf17c7a07a74f5474882",
    PAIRS_SAMPLE: "731a0ec8510a4e7c22be86afc0378279d644fec5aa4c6d041eb0b787c0155800",
}


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def protocol_fingerprint():
    # Commit IDs alone miss uncommitted scoring changes. Refuse mixed-code resumes.
    paths = [ROOT / "app/evaluator.py", ROOT / "app/jobs.py"]
    paths += sorted((ROOT / "llm_fact_checker").rglob("*.py"))
    return fingerprint({str(p.relative_to(ROOT)): file_sha256(p) for p in paths})


class LocalOllamaClient(OllamaClient):
    """Loopback only: no proxies, redirects, credentials, downloads or cloud routing."""

    def __init__(self, before_call=None):
        super().__init__("http://127.0.0.1:11434")
        self.before_call = before_call

    def _request(self, path, payload=None):
        if path not in {"/api/tags", "/api/show", "/api/version", "/api/chat"}:
            raise OllamaRequestError("Unsupported local Ollama operation.")
        if path == "/api/chat" and self.before_call:
            self.before_call()
        connection = http.client.HTTPConnection(
            "127.0.0.1", 11434, timeout=120 if path == "/api/chat" else 10
        )
        try:
            connection.request(
                "GET" if payload is None else "POST",
                path,
                body=None if payload is None else json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                raise OllamaRequestError(f"Local Ollama returned HTTP {response.status}.")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise OllamaRequestError("Ollama response exceeded the 2 MiB limit.")
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("error"):
                raise OllamaRequestError("Ollama returned an unexpected response.")
            if path == "/api/chat" and (
                data.get("done") is not True
                or not isinstance(data.get("message"), dict)
                or not isinstance(data["message"].get("content"), str)
            ):
                raise OllamaRequestError("Ollama did not return a completed text response.")
            return data
        except (OSError, http.client.HTTPException):
            # A timed-out call may still be running. Do not silently retry it.
            raise OllamaRequestError(
                "Local Ollama is unavailable or timed out. Start Ollama and keep the laptop awake."
            ) from None
        except (ValueError, UnicodeError) as error:
            if isinstance(error, OllamaRequestError):
                raise
            raise OllamaRequestError("Local Ollama returned invalid JSON.") from None
        finally:
            connection.close()

    def list_models(self):
        models = self._request("/api/tags").get("models")
        if not isinstance(models, list) or any(
            not isinstance(m, dict) or not isinstance(m.get("name"), str) for m in models
        ):
            raise OllamaRequestError("Ollama returned an invalid model list.")
        names = sorted(
            {
                m["name"]
                for m in models
                if m["name"]
                and not m.get("remote_host")
                and not m["name"].endswith((":cloud", "-cloud"))
            }
        )
        if not names:
            raise OllamaRequestError(
                "No local models found. Install a text model in Ollama first."
            )
        return names

    def check(self, model):
        metadata = super().check(model)
        details = self._request("/api/show", {"model": model})
        if (
            details.get("remote_host")
            or details.get("remote_model")
            or metadata["model"].get("remote_host")
            or model.endswith((":cloud", "-cloud"))
        ):
            raise OllamaRequestError("Choose a local model, not a cloud-routed model.")
        capabilities = details.get("capabilities")
        if not isinstance(capabilities, list) or "completion" not in capabilities:
            raise OllamaRequestError("This model does not support text generation.")
        if not metadata["model"].get("digest"):
            raise OllamaRequestError("Missing model digest; cannot identify this model safely.")
        metadata["details_sha256"] = fingerprint(details)
        return metadata


def model_identity(metadata):
    return {
        "digest": metadata["model"]["digest"],
        "version": metadata["version"],
        "details": metadata["details_sha256"],
    }


def checked_rows(path):
    if file_sha256(path) != INPUT_HASHES[path]:
        raise ValueError(f"Benchmark changed: {path.name}. Restore the frozen input first.")
    return list(read_jsonl(path))


def selection_order(rows, id_field):
    return sorted(rows, key=lambda r: fingerprint([42, str(r[id_field])]))


def make_plan(mode, percentage, tests):
    if mode not in {"Quick demo", "Custom"} or not 1 <= percentage <= 100:
        raise ValueError("Choose Quick demo or a percentage between 1 and 100.")
    if not tests or set(tests) - TESTS.keys():
        raise ValueError("Select at least one supported test.")
    tests = [key for key in TESTS if key in tests]
    fever, pairs, sources = [], [], {}
    if set(tests) - {"sensitivity"}:
        path = FEVER_SAMPLE if mode == "Quick demo" else FEVER_FULL
        population = checked_rows(path)
        sources["fever"] = {"sha256": file_sha256(path), "population": len(population)}
        # Interleave label-specific seeded orders; prefixes remain nested and balanced.
        groups = [
            selection_order([r for r in population if r["gold_label"] == label], "fever_id")
            for label in ("SUPPORTED", "REFUTED")
        ]
        ordered = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
        if mode == "Quick demo":
            fever = ordered[:2]
            fever.append(next((r for r in ordered[2:] if len(r["evidence"]) > 1), ordered[2]))
        else:
            fever = ordered[: max(2, math.ceil(len(ordered) * percentage / 100))]
    if "sensitivity" in tests:
        if mode == "Quick demo":
            population = checked_rows(PAIRS_SAMPLE)
            source_hash = file_sha256(PAIRS_SAMPLE)
        else:
            rows, summary = load_vitaminc_test(VITAMINC_TEST)
            population = identify_contrastive_pairs(rows)
            source_hash = summary["sha256"]
        validate_pairs(population, unique_claims=mode == "Quick demo")
        sources["vitaminc"] = {"sha256": source_hash, "population": len(population)}
        count = 3 if mode == "Quick demo" else math.ceil(len(population) * percentage / 100)
        pairs = selection_order(population, "pair_id")[:count]
    inputs = {"fever": fever, "pairs": pairs, "reorder": [], "distractor": []}
    if "robustness" in tests:
        _, reordered, inputs["distractor"] = build_perturbations(fever)
        inputs["reorder"] = [r for r in reordered if r["reorder_eligible"]]
    config = {
        "mode": mode,
        "percentage": None if mode == "Quick demo" else percentage,
        "tests": tests,
        "sources": sources,
        "selection": "seed42-label-interleaved-v1",
        "protocol": protocol_fingerprint(),
        "inputs_sha256": fingerprint(inputs),
    }
    plan = {"config": config, "inputs": inputs}
    config["expected_calls"] = sum(len(cases) for _, cases, _, _, _ in stages(plan))
    config["workload_id"] = fingerprint(config)
    return plan


def stages(plan):
    data = plan["inputs"]
    for test in plan["config"]["tests"]:
        if test == "baseline":
            yield "baseline", data["fever"], 42, 0, "fever_id"
        elif test == "consistency":
            for i, seed in enumerate(RUN_SEEDS, 1):
                yield f"consistency_{i}", data["fever"], seed, TEMPERATURE, "fever_id"
        elif test == "robustness":
            for name, key in (
                ("control", "fever"),
                ("reorder", "reorder"),
                ("distractor", "distractor"),
            ):
                yield f"robustness_{name}", data[key], 42, 0, "fever_id"
        else:
            yield "sensitivity", _inference_cases(data["pairs"]), 42, 0, "prediction_id"


def calibrate(client, model, plan):
    """Measure three input lengths per distinct workload; never save scored results."""
    metadata = client.check(model)
    measurements = {}
    for name, cases, seed, temperature, _ in stages(plan):
        family = "consistency" if name.startswith("consistency_") else name
        if family in measurements or not cases:
            continue
        ordered = sorted(cases, key=lambda c: len(json.dumps(c["evidence"])))
        seconds = []
        for i in sorted({0, len(ordered) // 2, len(ordered) - 1}):
            case = ordered[i]
            _, elapsed, _ = client.chat(model, case["claim"], case["evidence"], seed, temperature)
            seconds.append(elapsed)
        measurements[family] = seconds
    return {
        "model": model,
        "identity": model_identity(metadata),
        "workload_id": plan["config"]["workload_id"],
        "time": time.time(),
        "measurements": measurements,
    }


def estimated_seconds(plan, timing):
    if (
        timing["workload_id"] != plan["config"]["workload_id"]
        or time.time() - timing["time"] > 3600
    ):
        return None
    lower = upper = 0
    for name, cases, _, _, _ in stages(plan):
        if not cases:
            continue
        values = timing["measurements"]["consistency" if name.startswith("consistency_") else name]
        lower += len(cases) * min(values)
        upper += len(cases) * max(values)
    return lower, upper
