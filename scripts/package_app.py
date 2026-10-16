"""Build allow-listed source or macOS app ZIPs. Never publish or include credentials."""

import argparse
import hashlib
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package_files(root=ROOT):
    names = [
        "README.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "Setup ModelLens.command",
        "Launch ModelLens.command",
        "Makefile",
        "app/README.md",
        "app/style.css",
        "data/README.md",
        "data/HASHES.txt",
        "results/README.md",
        "data/fever/processed/fever_baseline_150.jsonl",
        "data/fever/full/fever_dev_full.jsonl",
        "data/vitaminc/processed/vitaminc_sensitivity_100_pairs.jsonl",
        "data/vitaminc/raw/vitaminc/test.jsonl",
        "scripts/package_app.py",
        "scripts/macos_setup.sh",
        "scripts/desktop_server.py",
        "scripts/macos_workspace.sh",
        "scripts/macos_app.applescript",
        "docs/macos-start-here.txt",
        "docs/methodology.md",
    ]
    paths = {root / name for name in names}
    for directory in ("app", "llm_fact_checker"):
        paths.update((root / directory).rglob("*.py"))
    for experiment in (
        "baseline_full",
        "consistency_5run_full",
        "robustness_full",
        "sensitivity_full",
    ):
        for folder in (root / "results").glob(f"*/*/{experiment}"):
            paths.update(p for p in folder.iterdir() if p.suffix in {".json", ".csv", ".md"})
    # Carry public theme configuration, never secrets.toml.
    theme = root / ".streamlit/config.toml"
    if theme.exists():
        paths.add(theme)
    for path in paths:
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(f"Missing or unsafe bundle input: {path.relative_to(root)}")
    return sorted(paths)


def build(destination, root=ROOT):
    files = package_files(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting a previous bundle. Remove/rename it explicitly before rebuilding.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            # Stable bytes mean rebuilding unchanged source reuses the same workspace.
            entry = zipfile.ZipInfo((Path("ModelLens") / path.relative_to(root)).as_posix())
            entry.create_system = 3
            entry.external_attr = (0o100755 if path.suffix == ".command" else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, path.read_bytes())
    return destination


def build_macos_app(destination, root=ROOT):
    """Ad-hoc signing seals the bundle; it is NOT Developer ID signing/notarization."""
    if sys.platform != "darwin":
        raise ValueError("Build the macOS launcher on a Mac (osacompile and codesign required).")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix="modellens-app-build-") as folder:
        work = Path(folder)
        app = work / "ModelLens.app"
        subprocess.run(
            ["/usr/bin/osacompile", "-o", str(app), str(root / "scripts/macos_app.applescript")],
            check=True,
        )
        resources = app / "Contents/Resources"
        payload = build(resources / "payload.zip", root)
        release = hashlib.sha256(payload.read_bytes()).hexdigest()
        (resources / "release-id.txt").write_text(release + "\n", encoding="utf-8")
        shutil.copyfile(root / "scripts/macos_workspace.sh", resources / "macos_workspace.sh")
        info_path = app / "Contents/Info.plist"
        with info_path.open("rb") as handle:
            info = plistlib.load(handle)
        info.update(
            {
                "CFBundleIdentifier": "org.evidencelab.modellens",
                "CFBundleName": "ModelLens",
                "CFBundleDisplayName": "ModelLens",
                "CFBundleShortVersionString": "0.1.0",
                "CFBundleVersion": "1",
                "LSMinimumSystemVersion": "14.0",
                "NSAppleEventsUsageDescription": (
                    "ModelLens opens its guided setup and local server in Terminal. "
                    "You approve downloads there; no commands need to be typed."
                ),
            }
        )
        with info_path.open("wb") as handle:
            plistlib.dump(info, handle)
        # Sign only after every resource and Info.plist entry is final.
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(app)], check=True)
        subprocess.run(
            ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)], check=True
        )
        archive = work / "app.zip"
        subprocess.run(
            [
                "/usr/bin/ditto",
                "-c",
                "-k",
                "--norsrc",
                "--noextattr",
                "--keepParent",
                str(app),
                str(archive),
            ],
            check=True,
        )
        with zipfile.ZipFile(archive, "a") as bundle:
            bundle.write(root / "docs/macos-start-here.txt", "START HERE.txt")
        verified = work / "verified"
        subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive), str(verified)], check=True)
        subprocess.run(
            [
                "/usr/bin/codesign",
                "--verify",
                "--deep",
                "--strict",
                str(verified / "ModelLens.app"),
            ],
            check=True,
        )
        # Exclusive creation: never silently replace an already shared release.
        with destination.open("xb") as output, archive.open("rb") as source:
            shutil.copyfileobj(source, output)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macos-app", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    builder = build_macos_app if args.macos_app else build
    default = "modellens-macos-app.zip" if args.macos_app else "modellens-macos.zip"
    print(builder(args.output or ROOT / "dist" / default))
