"""Offline desktop packaging tests; never start browsers, models or installers."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts import desktop_server, package_app

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "darwin", "Workspace extraction uses macOS ditto")
class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ModelLens ' workspace ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.resources = self.root / "resources with ' quotes"
        self.resources.mkdir()
        self.payload = self.resources / "payload.zip"
        with zipfile.ZipFile(self.payload, "w") as archive:
            archive.writestr("ModelLens/Launch ModelLens.command", "#!/bin/bash\nexit 0\n")
            archive.writestr("ModelLens/README.md", "fixture\n")
        self.release = hashlib.sha256(self.payload.read_bytes()).hexdigest()
        self.marker = self.resources / "release-id.txt"
        self.marker.write_text(self.release + "\n")
        self.support = self.root / "Application Support" / "ModelLens"
        self.workspace = self.support / self.release

    def prepare(self):
        return subprocess.run(
            [
                "/bin/bash",
                str(ROOT / "scripts/macos_workspace.sh"),
                str(self.marker),
                str(self.support),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )

    def test_real_extraction_quotes_and_repeat_preserves_local_data(self):
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = self.workspace / "ModelLens"
        self.assertEqual(result.stdout.strip(), str(expected))
        self.assertEqual((self.workspace / ".payload-id").read_text().strip(), self.release)
        saved = expected / ".modellens/runs/keep.json"
        saved.parent.mkdir(parents=True)
        saved.write_text("saved user result")
        (expected / "README.md").write_text("local modification")
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(saved.read_text(), "saved user result")
        self.assertEqual((expected / "README.md").read_text(), "local modification")

    def test_bad_hash_stops_before_creating_workspace(self):
        self.payload.write_bytes(self.payload.read_bytes() + b"altered")
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("damaged", result.stderr)
        self.assertFalse(self.support.exists())

    def test_invalid_build_identifier_stops_early(self):
        self.marker.write_text("../outside\n")
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid", result.stderr)
        self.assertFalse(self.support.exists())

    def test_existing_unowned_or_partial_workspace_is_not_overwritten(self):
        self.workspace.mkdir(parents=True)
        user_file = self.workspace / "keep.txt"
        user_file.write_text("keep")
        for marker in (None, self.release):
            with self.subTest(marker=marker):
                if marker:
                    (self.workspace / ".payload-id").write_text(marker + "\n")
                result = self.prepare()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(user_file.read_text(), "keep")
                self.assertFalse((self.workspace / "ModelLens").exists())

    def test_support_and_workspace_symlinks_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.support.parent.mkdir(parents=True)
        self.support.symlink_to(outside, target_is_directory=True)
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(outside.iterdir()), [])
        self.support.unlink()
        self.support.mkdir()
        self.workspace.symlink_to(outside, target_is_directory=True)
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(outside.iterdir()), [])

    def test_symlinked_ownership_marker_and_launcher_are_rejected(self):
        self.assertEqual(self.prepare().returncode, 0)
        marker = self.workspace / ".payload-id"
        marker.unlink()
        marker.symlink_to(self.marker)
        self.assertNotEqual(self.prepare().returncode, 0)
        marker.unlink()
        marker.write_text(self.release + "\n")
        launcher = self.workspace / "ModelLens/Launch ModelLens.command"
        launcher.unlink()
        other = self.root / "other.command"
        other.write_text("#!/bin/bash\nexit 0\n")
        launcher.symlink_to(other)
        self.assertNotEqual(self.prepare().returncode, 0)


class FakeProcess:
    def __init__(self, exit_code=None):
        self.returncode = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.returncode or 0
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class DesktopServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.desktop = self.root / ".modellens/desktop"
        self.desktop.mkdir(parents=True)
        self.state = self.desktop / "server.json"
        self.browser = Mock()
        # No real network listener, HTTP request, process, browser or delay.
        self.socket = self.enterContext(patch.object(desktop_server.socket, "socket"))
        self.socket.return_value.__enter__.return_value.getsockname.return_value = (
            "127.0.0.1",
            63123,
        )
        self.sleep = self.enterContext(patch.object(desktop_server.time, "sleep"))
        self.health = self.enterContext(patch.object(desktop_server, "healthy", return_value=True))
        self.spawn = self.enterContext(patch.object(desktop_server.subprocess, "Popen"))
        self.process = FakeProcess()
        self.spawn.return_value = self.process

    def test_ready_child_opens_only_its_unique_url_and_removes_state_at_exit(self):
        def opened(url):
            self.assertEqual(url, desktop_server.server_url(self.state))
            self.assertTrue(url.startswith("http://127.0.0.1:63123/modellens-"))

        self.browser.side_effect = opened
        self.assertEqual(desktop_server.main(self.root, self.browser), 0)
        self.browser.assert_called_once()
        command = self.spawn.call_args.args[0]
        self.assertIn("--server.address=127.0.0.1", command)
        self.assertIn("--server.headless=true", command)
        self.assertIn("--browser.gatherUsageStats=false", command)
        self.assertEqual(self.spawn.call_args.kwargs["cwd"], self.root)
        self.assertEqual(self.spawn.call_args.kwargs["env"]["MODELLENS_LOCAL"], "1")
        self.assertFalse(self.state.exists())
        self.assertFalse(self.process.terminated)

    def test_failed_child_does_not_open_browser(self):
        self.process.returncode = 7
        self.assertEqual(desktop_server.main(self.root, self.browser), 7)
        self.browser.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_timeout_terminates_child_without_opening_browser(self):
        self.health.return_value = False
        self.assertEqual(desktop_server.main(self.root, self.browser), 1)
        self.browser.assert_not_called()
        self.assertTrue(self.process.terminated)
        self.assertFalse(self.state.exists())

    def test_second_launch_uses_workspace_state_without_spawning(self):
        self.state.write_text(json.dumps({"port": 63144, "token": "b" * 32}))
        expected = "http://127.0.0.1:63144/modellens-" + "b" * 32
        with patch.object(desktop_server.fcntl, "flock", side_effect=BlockingIOError):
            self.assertEqual(desktop_server.main(self.root, self.browser), 0)
        self.browser.assert_called_once_with(expected)
        self.health.assert_called_once_with(expected)
        self.spawn.assert_not_called()
        self.socket.assert_not_called()
        self.assertTrue(self.state.exists())

    def test_second_launch_rejects_invalid_state_without_browser_or_spawn(self):
        self.state.write_text(json.dumps({"port": "https://example.invalid", "token": "b" * 32}))
        with patch.object(desktop_server.fcntl, "flock", side_effect=BlockingIOError):
            self.assertEqual(desktop_server.main(self.root, self.browser), 1)
        self.browser.assert_not_called()
        self.health.assert_not_called()
        self.spawn.assert_not_called()

    def test_url_state_is_strictly_local_and_typed(self):
        for port, token in (
            (True, "a" * 32),
            (0, "a" * 32),
            (65536, "a" * 32),
            ("8501", "a" * 32),
            (8501, "a" * 31),
            (8501, "A" * 32),
            (8501, "../" + "a" * 29),
            (8501, ["a"] * 32),
            (8501, None),
        ):
            with self.subTest(port=port, token=token):
                self.state.write_text(json.dumps({"port": port, "token": token}))
                with self.assertRaises((ValueError, TypeError)):
                    desktop_server.server_url(self.state)


class SourcePackageTests(unittest.TestCase):
    def test_source_zip_is_deterministic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "Launch ModelLens.command"
            launcher.write_text("#!/bin/bash\nexit 0\n")
            readme = root / "README.md"
            readme.write_text("fixture")
            with patch.object(package_app, "package_files", return_value=[launcher, readme]):
                first = package_app.build(root / "first.zip", root)
                os.utime(launcher, (1700000000, 1700000000))
                second = package_app.build(root / "second.zip", root)
                self.assertEqual(first.read_bytes(), second.read_bytes())
                with self.assertRaises(FileExistsError):
                    package_app.build(first, root)
            with zipfile.ZipFile(first) as archive:
                entry = archive.getinfo("ModelLens/Launch ModelLens.command")
                self.assertEqual(entry.external_attr >> 16, 0o100755)
                self.assertEqual(archive.read("ModelLens/README.md"), b"fixture")

    def test_payload_contains_desktop_runtime_but_no_private_state(self):
        paths = {p.relative_to(ROOT) for p in package_app.package_files()}
        self.assertIn(Path("scripts/desktop_server.py"), paths)
        for path in paths:
            self.assertFalse(set(path.parts) & {".venv", ".modellens", ".git", "__pycache__"})
            self.assertNotIn(path.name, {".env", "secrets.toml"})


if __name__ == "__main__":
    unittest.main()
