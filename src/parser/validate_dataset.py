#!/usr/bin/env python
"""
Unified dataset validation entrypoint for C2|Q> parser data.

This wrapper keeps parser-local validation ergonomics while delegating the
canonical implementations to:
  - src/validation/implementation_validation.py
  - src/validation/diversity_validation.py

Default mode runs both validation stages and writes outputs under a single root.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.validation.implementation_validation import *  # noqa: F401,F403
import src.validation.implementation_validation as _impl


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_primary_csv() -> Path:
    return repo_root() / "src" / "parser" / "python_programs.csv"


def default_backup_csv() -> Path:
    return repo_root() / "src" / "parser" / "data.csv"


def default_model_path() -> Path:
    env_path = os.getenv("C2Q_MODEL_PATH")
    if env_path:
        return Path(env_path).expanduser()

    candidates = [
        repo_root() / "src" / "parser" / "saved_models_2025_12",
        repo_root() / "src" / "parser" / "saved_models",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_out_root() -> Path:
    return repo_root() / "src" / "parser" / "validation_out"


def ensure_pythonpath(env: dict[str, str]) -> dict[str, str]:
    updated = env.copy()
    root = str(repo_root())
    if updated.get("PYTHONPATH"):
        updated["PYTHONPATH"] = f"{root}{os.pathsep}{updated['PYTHONPATH']}"
    else:
        updated["PYTHONPATH"] = root
    return updated


def run_implementation_validation(
    input_csv: Path,
    backup_csv: Path,
    out_snippet_metrics: Path,
    out_family_summary: Path,
    out_syntax_failures: Path,
) -> None:
    resolved_csv = _impl.resolve_input_csv(input_csv, backup_csv)
    _impl.main(
        csv_path=str(resolved_csv),
        out_snippet_metrics=str(out_snippet_metrics),
        out_family_summary=str(out_family_summary),
        out_syntax_failures=str(out_syntax_failures),
    )


def run_diversity_validation(
    input_csv: Path,
    backup_csv: Path,
    out_dir: Path,
    model_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    env = ensure_pythonpath(os.environ)
    env["C2Q_INPUT_CSV"] = str(input_csv)
    env["C2Q_BACKUP_CSV"] = str(backup_csv)
    env["C2Q_DIVERSITY_OUT"] = str(out_dir)
    env["C2Q_MODEL_PATH"] = str(model_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "src.validation.diversity_validation.MyTestCase.test_diversity_direct_metrics",
        ],
        cwd=str(repo_root()),
        env=env,
        check=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run implementation-level and algorithmic/structural dataset validation."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "implementation", "diversity"],
        default="all",
        help="Which validation stage to run.",
    )
    parser.add_argument(
        "--csv-path",
        default=str(default_primary_csv()),
        help="Primary CSV with code_snippet + labels columns.",
    )
    parser.add_argument(
        "--backup-csv-path",
        default=str(default_backup_csv()),
        help="Backup CSV used when --csv-path is missing.",
    )
    parser.add_argument(
        "--model-path",
        default=str(default_model_path()),
        help="Parser model directory used by diversity validation.",
    )
    parser.add_argument(
        "--out-root",
        default=str(default_out_root()),
        help="Root directory for combined validation outputs.",
    )
    parser.add_argument(
        "--out-snippet-metrics",
        default=None,
        help="Override implementation metrics CSV output path.",
    )
    parser.add_argument(
        "--out-family-summary",
        default=None,
        help="Override implementation family summary CSV output path.",
    )
    parser.add_argument(
        "--out-syntax-failures",
        default=None,
        help="Override implementation syntax failures CSV output path.",
    )
    parser.add_argument(
        "--diversity-out",
        default=None,
        help="Override diversity output directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_csv = Path(args.csv_path)
    backup_csv = Path(args.backup_csv_path)
    model_path = Path(args.model_path)
    out_root = Path(args.out_root)

    impl_dir = out_root / "implementation"
    div_dir = out_root / "diversity"

    out_snippet_metrics = Path(args.out_snippet_metrics) if args.out_snippet_metrics else impl_dir / "snippet_metrics.csv"
    out_family_summary = Path(args.out_family_summary) if args.out_family_summary else impl_dir / "family_summary.csv"
    out_syntax_failures = Path(args.out_syntax_failures) if args.out_syntax_failures else impl_dir / "syntax_failures.csv"
    diversity_out = Path(args.diversity_out) if args.diversity_out else div_dir

    if args.mode in {"all", "implementation"}:
        print("[validation] running implementation-level validation")
        run_implementation_validation(
            input_csv=input_csv,
            backup_csv=backup_csv,
            out_snippet_metrics=out_snippet_metrics,
            out_family_summary=out_family_summary,
            out_syntax_failures=out_syntax_failures,
        )
        print(f"[validation] implementation outputs: {impl_dir}")

    if args.mode in {"all", "diversity"}:
        print("[validation] running algorithmic/structural diversity validation")
        run_diversity_validation(
            input_csv=input_csv,
            backup_csv=backup_csv,
            out_dir=diversity_out,
            model_path=model_path,
        )
        print(f"[validation] diversity outputs: {diversity_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
