"""Offline shell tests: installers, network, macOS services and app launch are stubbed."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/macos_setup.sh"


class MacSetupTests(unittest.TestCase):
    def shell(self, script, answers="", stubs=""):
        with tempfile.TemporaryDirectory() as folder:
            # Fail closed: a test must explicitly stub every external side effect it exercises.
            prelude = f"source {shlex.quote(str(HELPER))}\n"
            for tool in ("curl", "env", "open", "ditto", "hdiutil", "codesign", "spctl"):
                prelude += f'{tool}() {{ echo "UNEXPECTED {tool}"; return 99; }}\n'
            result = subprocess.run(
                ["/bin/bash", "-eu", "-c", prelude + stubs + "\n" + script],
                cwd=folder,
                input=answers,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        self.assertNotIn("UNEXPECTED", result.stdout + result.stderr)
        return result

    def test_declining_uv_and_ollama_installs_makes_no_downloads(self):
        result = self.shell("install_uv", "n\n")
        self.assertNotEqual(result.returncode, 0)
        result = self.shell("install_ollama", "\n")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_uv_is_scoped_and_installer_only_runs_after_complete_download(self):
        stubs = """
download() { echo "download:$1"; }
env() { printf 'installer:%s\n' "$*"; }
"""
        result = self.shell("install_uv", "y\n", stubs)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("uv/0.13.1/install.sh", result.stdout)
        self.assertIn("UV_UNMANAGED_INSTALL=", result.stdout)
        self.assertIn("/.modellens/tools", result.stdout)
        result = self.shell("install_uv", "y\n", stubs + "download() { return 22; }\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("installer:", result.stdout)

    def test_ollama_appearing_before_install_is_preserved(self):
        result = self.shell("install_ollama", "y\n", "find_ollama_app() { return 0; }")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("skipping installation", result.stdout)

    def ollama_stubs(self):
        return """
find_ollama_app() { return 1; }
find_tool() { return 1; }
ollama_running() { return 1; }
download() { echo download; }
hdiutil() { echo "disk:$1"; }
codesign() { echo signature; }
spctl() { echo gatekeeper; }
mkdir() { echo "directory:$*"; }
ditto() { echo copy; }
"""

    def test_ollama_copy_is_gated_by_signature_and_never_replaces_destination(self):
        result = self.shell("install_ollama", "yes\n", self.ollama_stubs())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(result.stdout.index("\ngatekeeper\n"), result.stdout.index("\ncopy\n"))
        self.assertIn("disk:detach", result.stdout)
        for failure in (
            "download() { return 22; }",
            "codesign() { return 1; }",
            "spctl() { return 1; }",
            'mkdir() { case "$*" in *Ollama.app) return 1;; esac; }',
        ):
            result = self.shell("install_ollama", "y\n", self.ollama_stubs() + failure)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("\ncopy\n", result.stdout)

    def test_running_or_declined_ollama_does_not_launch_anything(self):
        result = self.shell("offer_start_ollama", stubs="ollama_running() { return 0; }")
        self.assertEqual(result.returncode, 0)
        self.assertIn("already running", result.stdout)
        result = self.shell(
            "offer_start_ollama",
            "n\n",
            "ollama_running() { return 1; }; find_ollama_app() { echo /fake/Ollama.app; }",
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Saved comparisons still work", result.stdout)

    def test_start_requires_consent_and_waits_for_local_api(self):
        result = self.shell(
            "offer_start_ollama",
            "y\n",
            """
ready=0
ollama_running() { [ "$ready" = 1 ]; }
find_ollama_app() { echo /fake/Ollama.app; }
open() { echo "opened:$1"; ready=1; }
""",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("opened:/fake/Ollama.app", result.stdout)
        self.assertIn("Ollama is ready", result.stdout)

    def test_start_failure_allows_saved_comparisons(self):
        result = self.shell(
            "offer_start_ollama",
            "y\n",
            """
ollama_running() { return 1; }
find_ollama_app() { echo /fake/Ollama.app; }
open() { return 1; }
""",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved comparisons still work", result.stdout)

    def test_wrong_platform_and_old_mac_stop_early(self):
        result = self.shell("require_mac", stubs="uname() { echo Linux; }")
        self.assertNotEqual(result.returncode, 0)
        result = self.shell(
            "require_mac",
            stubs="""
uname() { echo Darwin; }
sw_vers() { echo 13.6; }
""",
        )
        self.assertNotEqual(result.returncode, 0)

    def test_launcher_scripts_are_syntactically_valid_and_executable(self):
        for name in (
            "Setup ModelLens.command",
            "Launch ModelLens.command",
            "scripts/macos_setup.sh",
        ):
            result = subprocess.run(
                ["/bin/bash", "-n", str(ROOT / name)], capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            if name.endswith(".command"):
                self.assertTrue(os.access(ROOT / name, os.X_OK))

    def test_setup_reuses_uv_and_preserves_extra_packages(self):
        # Run an isolated copy with a fake helper; never execute the real installer.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            script = root / "Setup ModelLens.command"
            script.write_bytes((ROOT / script.name).read_bytes())
            (root / "scripts/macos_setup.sh").write_text("""
require_mac() { :; }
pause_on_exit() { :; }
find_tool() { echo fake_uv; }
fake_uv() { echo "uv:$*"; }
ask() { return 0; }
app_ready() { return 0; }
ollama_running() { return 0; }
offer_start_ollama() { :; }
install_uv() { echo UNEXPECTED; exit 99; }
install_ollama() { echo UNEXPECTED; exit 99; }
""")
            result = subprocess.run(
                ["/bin/bash", str(script)], text=True, capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("UNEXPECTED", result.stdout)
            self.assertIn("--inexact", result.stdout)
            self.assertIn("--locked", result.stdout)
            self.assertNotIn("self update", result.stdout)
            # A symlink must not redirect dependency installation to another environment.
            (root / ".venv").symlink_to(root / "elsewhere")
            result = subprocess.run(
                ["/bin/bash", str(script)], text=True, capture_output=True, check=False
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("uv:sync", result.stdout)


if __name__ == "__main__":
    unittest.main()
