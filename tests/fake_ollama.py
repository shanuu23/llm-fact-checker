"""A tiny scriptable fake Ollama server for tests (standard library only).

It answers ``/api/version``, ``/api/tags`` and ``/api/chat``. By default it returns one of the
three labels, chosen from the request text so runs are deterministic. Pass ``chat_handler`` to
script failures: it receives the 1-based chat request number and the request payload, and returns
``(status, body)`` where ``body`` is a dict (sent as JSON), ``bytes`` (sent as-is), or ``None``;
return ``DROP`` to close the connection without answering.
"""

from __future__ import annotations

from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any

LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")
DROP = object()

ChatHandler = Callable[[int, dict[str, Any]], Any]


class FakeOllama:
    def __init__(
        self,
        installed: tuple[str, ...] = ("fake-model:1b",),
        chat_handler: ChatHandler | None = None,
    ):
        self.installed = installed
        self.chat_handler = chat_handler
        self.chat_requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, status: int, body: bytes | None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _send_json(self, payload: dict[str, Any]) -> None:
                self._send(200, json.dumps(payload).encode())

            def do_GET(self):
                if self.path == "/api/version":
                    self._send_json({"version": "0.0.0-fake"})
                elif self.path == "/api/tags":
                    models = [{"name": n, "digest": f"digest-of-{n}"} for n in outer.installed]
                    self._send_json({"models": models})
                else:
                    self._send(404, None)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(length))
                outer.chat_requests.append(payload)
                if outer.chat_handler is not None:
                    outcome = outer.chat_handler(len(outer.chat_requests), payload)
                    if outcome is DROP:
                        self.close_connection = True
                        return
                    if outcome is not None:
                        status, body = outcome
                        if isinstance(body, dict):
                            body = json.dumps(body).encode()
                        self._send(status, body)
                        return
                user = payload["messages"][-1]["content"]
                label = LABELS[sum(map(ord, user)) % 3]
                self._send_json({"message": {"content": label}, "done_reason": "stop"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> FakeOllama:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
