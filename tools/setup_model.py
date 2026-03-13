#!/usr/bin/env python3
"""
Model setup helper for collaborators.

This repository does not store parser model weights in GitHub.
Use this helper to:
1) verify a local model directory, or
2) download/extract the model archive from Google Drive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


DEFAULT_MODEL_URL = (
    "https://drive.google.com/file/d/"
    "11xkJgioQkVdCGykGSLjJD1CcXu76RAIB/view?usp=drive_link"
)

REQUIRED_FILES = ("config.json", "tokenizer_config.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _default_checksum_file() -> Path:
    return repo_root() / "tools" / "model_checksums.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def check_model_dir(model_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if not any((model_dir / name).is_file() for name in WEIGHT_FILES):
        missing.append("model.safetensors|pytorch_model.bin")
    return missing


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_checksums(model_dir: Path, checksum_file: Path) -> list[str]:
    if not checksum_file.is_file():
        return []

    payload = json.loads(checksum_file.read_text(encoding="utf-8"))
    expected_files = payload.get("files", {})
    issues: list[str] = []

    for rel_name, expected_hash in expected_files.items():
        file_path = model_dir / rel_name
        if not file_path.is_file():
            issues.append(f"missing {rel_name}")
            continue
        actual_hash = sha256sum(file_path)
        if actual_hash.lower() != str(expected_hash).lower():
            issues.append(f"checksum mismatch {rel_name}")
    return issues


def find_model_dir(search_root: Path) -> Path | None:
    if not search_root.exists():
        return None
    if search_root.is_dir() and not check_model_dir(search_root):
        return search_root
    for child in search_root.rglob("*"):
        if child.is_dir() and not check_model_dir(child):
            return child
    return None


def download_with_gdown(url: str, output_file: Path) -> None:
    cmd = [sys.executable, "-m", "gdown", "--fuzzy", url, "-O", str(output_file)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to download model with gdown. "
            "Install gdown (`pip install gdown`) or download manually."
        ) from exc


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(target_dir)
        return
    raise RuntimeError(
        f"Unsupported archive format: {archive_path}\n"
        "Provide a .zip/.tar archive or use --download with gdown."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check/download parser model for C2Q.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=repo_root() / "src" / "parser" / "saved_models_2025_12",
        help="Target directory for extracted model files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download model archive from Google Drive with gdown.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_MODEL_URL,
        help="Download URL for model archive.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to a pre-downloaded model archive (.zip/.tar*).",
    )
    parser.add_argument(
        "--checksum-file",
        type=Path,
        default=_default_checksum_file(),
        help="JSON file with expected SHA256 checksums for model files.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip SHA256 verification after check/install.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    checksum_file = args.checksum_file.expanduser().resolve()

    missing = check_model_dir(model_path)
    if not missing:
        if not args.skip_checksum:
            checksum_issues = verify_model_checksums(model_path, checksum_file)
            if checksum_issues:
                issue_text = ", ".join(checksum_issues)
                raise RuntimeError(
                    "Model checksum verification failed for existing model: "
                    f"{issue_text}. Re-download from {args.url}"
                )
        print(f"[ok] Model is ready: {model_path}")
        return 0

    print(f"[info] Model missing or incomplete at: {model_path}")
    print(f"[info] Missing files: {', '.join(missing)}")

    if not args.download and args.archive is None:
        print(
            "[hint] Download model manually:\n"
            f"       {args.url}\n"
            "       then run: python tools/setup_model.py --archive /path/to/model_archive.zip"
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="c2q_model_") as tmp:
        tmp_dir = Path(tmp)
        archive_path = args.archive
        if archive_path is None:
            archive_path = tmp_dir / "model_archive"
            print("[step] Downloading model archive via gdown...")
            download_with_gdown(args.url, archive_path)
        else:
            archive_path = archive_path.expanduser().resolve()
            if not archive_path.is_file():
                raise RuntimeError(f"Archive file not found: {archive_path}")

        extract_dir = tmp_dir / "extracted"
        print(f"[step] Extracting archive: {archive_path}")
        extract_archive(archive_path, extract_dir)

        discovered = find_model_dir(extract_dir)
        if discovered is None:
            raise RuntimeError(
                "Could not find a valid model directory inside the extracted archive.\n"
                "Required files: "
                f"{', '.join(REQUIRED_FILES)}, and one of: {', '.join(WEIGHT_FILES)}"
            )

        print(f"[step] Installing model into: {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            shutil.rmtree(model_path)
        shutil.copytree(discovered, model_path)

    missing_after = check_model_dir(model_path)
    if missing_after:
        raise RuntimeError(
            "Model install completed but required files are still missing: "
            + ", ".join(missing_after)
        )
    if not args.skip_checksum:
        checksum_issues = verify_model_checksums(model_path, checksum_file)
        if checksum_issues:
            issue_text = ", ".join(checksum_issues)
            raise RuntimeError(
                "Model install completed, but checksum verification failed: "
                f"{issue_text}"
            )

    print(f"[ok] Model installed and verified: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
