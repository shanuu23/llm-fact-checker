"""One local Streamlit server per packaged workspace; reopen its own browser page."""

import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def server_url(state):
    value = json.loads(state.read_text())
    port, token = value["port"], value["token"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Invalid local server port.")
    if (
        not isinstance(token, str)
        or len(token) != 32
        or any(c not in "0123456789abcdef" for c in token)
    ):
        raise ValueError("Invalid local server identifier.")
    return f"http://127.0.0.1:{port}/modellens-{token}"


def healthy(url):
    # A unique base path avoids opening a different app even if a port is reused.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with client.open(url + "/_stcore/health", timeout=1) as response:
            return response.status == 200 and response.read(16) == b"ok"
    except (OSError, urllib.error.URLError):
        return False


def main(root=ROOT, open_browser=webbrowser.open):
    desktop = root / ".modellens/desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    state = desktop / "server.json"
    with (desktop / "server.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            for _ in range(30):
                try:
                    url = server_url(state)
                    if healthy(url):
                        print(f"ModelLens is already running: {url}", flush=True)
                        open_browser(url)
                        return 0
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                time.sleep(1)
            print("ModelLens is still starting. Check its existing setup/server window.")
            return 1

        state.unlink(missing_ok=True)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        token = uuid.uuid4().hex
        url = f"http://127.0.0.1:{port}/modellens-{token}"
        env = {**os.environ, "MODELLENS_LOCAL": "1"}
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/main.py",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            f"--server.baseUrlPath=modellens-{token}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
        print(
            "Starting ModelLens. Keep this window open; pause evaluations before closing it.",
            flush=True,
        )
        process = subprocess.Popen(command, cwd=root, env=env, pass_fds=(lock.fileno(),))
        try:
            for _ in range(60):
                if process.poll() is not None:
                    return process.returncode or 1
                if healthy(url):
                    state.write_text(json.dumps({"port": port, "token": token}))
                    print(f"ModelLens is ready: {url}", flush=True)
                    open_browser(url)
                    return process.wait()
                time.sleep(0.5)
            print("ModelLens did not become ready. Review the server error above.")
            return 1
        except KeyboardInterrupt:
            return 0
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            state.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
