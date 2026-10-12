"""The model-calling path under failure: every case must be preserved, never lost with the run."""

from __future__ import annotations

import contextlib
import io
import socket
import unittest
from unittest import mock

try:
    from tests.fake_ollama import DROP, FakeOllama
except ModuleNotFoundError:  # run with tests/ itself on sys.path
    from fake_ollama import DROP, FakeOllama

from llm_fact_checker.ollama import OllamaClient, infer_cases

MODEL = "fake-model:1b"


def make_cases(count: int) -> list[dict]:
    return [
        {
            "fever_id": index,
            "claim": f"Claim {index}.",
            "evidence": [{"page": "Page", "sentence_id": 0, "text": "Some evidence."}],
            "gold_label": "SUPPORTED",
        }
        for index in range(1, count + 1)
    ]


def run(server: FakeOllama, count: int, **kwargs):
    client = OllamaClient(server.url, timeout=5)
    with contextlib.redirect_stderr(io.StringIO()):
        return infer_cases(client, MODEL, make_cases(count), seed=7, temperature=0, **kwargs)


def reply(label: str):
    return 200, {"message": {"content": label}}


@mock.patch("llm_fact_checker.ollama.time.sleep")  # keep retry back-off instant
class FailureHandlingTests(unittest.TestCase):
    def test_http_400_on_one_case_is_preserved_and_the_run_continues(self, _sleep):
        def handler(n, _payload):
            return (400, None) if n == 3 else reply("SUPPORTED")

        with FakeOllama(installed=(MODEL,), chat_handler=handler) as server:
            results = run(server, 5)
        self.assertEqual(len(results), 5)
        failed = results[2]
        self.assertIsNone(failed["parsed_prediction"])
        self.assertFalse(failed["valid_format"])
        self.assertIn("HTTP 400", failed["inference_error"])
        self.assertEqual(sum(bool(r["inference_error"]) for r in results), 1)
        self.assertEqual(len(server.chat_requests), 5)  # a rejected request is not retried

    def test_transient_server_error_is_retried_then_succeeds(self, _sleep):
        def handler(n, _payload):
            return (500, None) if n == 1 else reply("REFUTED")

        with FakeOllama(installed=(MODEL,), chat_handler=handler) as server:
            (result,) = run(server, 1)
        self.assertIsNone(result["inference_error"])
        self.assertEqual(result["parsed_prediction"], "REFUTED")
        self.assertEqual(len(server.chat_requests), 2)

    def test_persistent_server_error_is_preserved_after_retries(self, _sleep):
        with FakeOllama(installed=(MODEL,), chat_handler=lambda n, p: (503, None)) as server:
            (result,) = run(server, 1)
        self.assertIn("failed", result["inference_error"])
        self.assertIsNone(result["raw_model_output"])
        self.assertEqual(len(server.chat_requests), 3)  # first try + two retries

    def test_connection_dropped_mid_request_is_retried(self, _sleep):
        def handler(n, _payload):
            return DROP if n == 1 else reply("SUPPORTED")

        with FakeOllama(installed=(MODEL,), chat_handler=handler) as server:
            (result,) = run(server, 1)
        self.assertIsNone(result["inference_error"])
        self.assertEqual(len(server.chat_requests), 2)

    def test_reply_that_is_not_json_is_preserved_without_retry(self, _sleep):
        with FakeOllama(installed=(MODEL,), chat_handler=lambda n, p: (200, b"<html>")) as server:
            (result,) = run(server, 1)
        self.assertIn("invalid JSON", result["inference_error"])
        self.assertEqual(len(server.chat_requests), 1)

    def test_reply_without_a_message_is_preserved_without_retry(self, _sleep):
        with FakeOllama(installed=(MODEL,), chat_handler=lambda n, p: (200, {"done": True})) as s:
            (result,) = run(s, 1)
        self.assertIn("message.content", result["inference_error"])
        self.assertEqual(len(s.chat_requests), 1)

    def test_one_failure_does_not_disturb_neighbouring_cases(self, _sleep):
        def handler(n, _payload):
            return (400, None) if n == 2 else reply("REFUTED")

        with FakeOllama(installed=(MODEL,), chat_handler=handler) as server:
            results = run(server, 3)
        self.assertEqual([r["fever_id"] for r in results], [1, 2, 3])
        self.assertEqual([r["parsed_prediction"] for r in results], ["REFUTED", None, "REFUTED"])


class OutputHandlingTests(unittest.TestCase):
    def test_a_near_miss_label_is_recorded_as_invalid_not_repaired(self):
        with FakeOllama(installed=(MODEL,), chat_handler=lambda n, p: reply("Supported.")) as s:
            (result,) = run(s, 1)
        self.assertEqual(result["raw_model_output"], "Supported.")
        self.assertIsNone(result["parsed_prediction"])
        self.assertFalse(result["valid_format"])
        self.assertFalse(result["correct"])
        self.assertIsNone(result["inference_error"])

    def test_request_carries_the_seed_temperature_and_no_streaming(self):
        with FakeOllama(installed=(MODEL,)) as server:
            run(server, 1)
        (payload,) = server.chat_requests
        self.assertEqual(payload["model"], MODEL)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["options"], {"temperature": 0, "seed": 7})

    def test_gold_label_is_never_sent_to_the_model(self):
        with FakeOllama(installed=(MODEL,)) as server:
            run(server, 1)
        sent = " ".join(m["content"] for m in server.chat_requests[0]["messages"])
        self.assertNotIn("gold", sent.lower())


class ConnectionTests(unittest.TestCase):
    def test_unreachable_server_is_reported_as_not_reachable(self):
        with socket.socket() as probe:  # grab a port that nothing is listening on
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        client = OllamaClient(f"http://127.0.0.1:{port}", timeout=2)
        with self.assertRaisesRegex(RuntimeError, "not reachable"):
            client.check(MODEL)

    def test_missing_model_lists_what_is_installed(self):
        with FakeOllama(installed=("other:1b",)) as server:
            with self.assertRaisesRegex(RuntimeError, r"not installed.*other:1b"):
                OllamaClient(server.url).check(MODEL)


if __name__ == "__main__":
    unittest.main()
