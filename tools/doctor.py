#!/usr/bin/env python3
"""
Contributor environment diagnostics for C2Q.

Checks:
- Python version compatibility
- LaTeX compiler availability
- Parser model presence/integrity
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from setup_model import (  # noqa: E402
    check_model_dir,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_model_path() -> Path:
    env_path = os.getenv("C2Q_MODEL_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return repo_root() / "src" / "parser" / "saved_models_2025_12"


def resolve_latex_compiler() -> str:
    return os.getenv("C2Q_PDFLATEX", "pdflatex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C2Q contributor environment diagnostics.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path(),
        help="Path to parser model directory.",
    )
    return parser.parse_args()


def _ok(name: str, msg: str) -> tuple[str, str, str]:
    return name, "OK", msg


def _fail(name: str, msg: str) -> tuple[str, str, str]:
    return name, "FAIL", msg


def main() -> int:
    args = parse_args()
    checks: list[tuple[str, bool, str]] = []

    py = sys.version_info
    if py[:2] != (3, 12):
        checks.append(
            _fail(
                "python",
                f"{py.major}.{py.minor}.{py.micro} (need exactly Python 3.12 for the source/reviewer path)",
            )
        )
    else:
        checks.append(_ok("python", f"{py.major}.{py.minor}.{py.micro}"))

    latex_compiler = resolve_latex_compiler()
    if os.path.sep in latex_compiler:
        latex_ok = Path(latex_compiler).expanduser().is_file()
    else:
        latex_ok = shutil.which(latex_compiler) is not None
    if latex_ok:
        checks.append(_ok("latex", f"{latex_compiler} available"))
    else:
        checks.append(_fail("latex", f"{latex_compiler} not found (install TeX Live/MacTeX or set C2Q_PDFLATEX)"))

    model_path = args.model_path.expanduser().resolve()
    missing = check_model_dir(model_path)
    if missing:
        checks.append(
            _fail(
                "model",
                f"{model_path} missing: {', '.join(missing)}",
            )
        )
    else:
        checks.append(_ok("model", str(model_path)))

    failed = 0
    print("C2Q doctor results:")
    for name, status, message in checks:
        print(f" - [{status}] {name}: {message}")
        if status == "FAIL":
            failed += 1

    if failed:
        print("\nDoctor found blocking issues. Fix FAIL items before running reproduce-* targets.")
        return 1

    print("\nDoctor check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
