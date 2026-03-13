#!/usr/bin/env python3
"""
Reproducibility runner for C2Q paper artifacts.

Pipeline:
1) implementation-level validation
2) algorithmic/structural diversity validation
3) report generation (paper-scale or smoke-scale)
4) artifact and metadata export

Paper mode can generate 434 reports and is time-consuming.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def default_output_root() -> Path:
    return repo_root() / "artifacts" / "reproduce"


def validate_model_dir(model_path: Path) -> None:
    if not model_path.is_dir():
        raise RuntimeError(
            "Model directory not found for parser classification:\n"
            f" - {model_path}\n"
            "Set C2Q_MODEL_PATH or pass --model-path to an existing saved_models directory.\n"
            "Download model: https://drive.google.com/file/d/11xkJgioQkVdCGykGSLjJD1CcXu76RAIB/view?usp=drive_link"
        )

    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (model_path / name).is_file()]
    has_weights = (
        (model_path / "model.safetensors").is_file()
        or (model_path / "pytorch_model.bin").is_file()
    )
    if missing:
        missing_txt = ", ".join(missing)
        raise RuntimeError(
            "Model directory is missing required files for parser classification:\n"
            f" - {model_path}\n"
            f"Missing: {missing_txt}\n"
            "Download model: https://drive.google.com/file/d/11xkJgioQkVdCGykGSLjJD1CcXu76RAIB/view?usp=drive_link"
        )
    if not has_weights:
        raise RuntimeError(
            "Model directory is missing model weights for parser classification:\n"
            f" - {model_path}\n"
            "Missing one of: model.safetensors, pytorch_model.bin\n"
            "Download model: https://drive.google.com/file/d/11xkJgioQkVdCGykGSLjJD1CcXu76RAIB/view?usp=drive_link"
        )


def resolve_input_csv(primary_csv: Path, backup_csv: Path) -> Path:
    if primary_csv.exists():
        return primary_csv
    if backup_csv.exists():
        print(f"[info] Primary CSV missing. Using backup CSV: {backup_csv}")
        return backup_csv
    raise FileNotFoundError(
        "No input CSV found. Checked:\n"
        f" - {primary_csv}\n"
        f" - {backup_csv}"
    )


def run_cmd(step_name: str, cmd: List[str], env: Dict[str, str]) -> None:
    print(f"\n[step] {step_name}")
    print("[cmd]  " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo_root()), env=env, check=True)


def get_git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root()))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_diversity_outputs(diversity_dir: Path) -> Dict[str, str]:
    expected = [
        "metrics_per_instance.csv",
        "summary_by_tag.csv",
        "uniqueness_by_tag.csv",
        "buckets_by_tag.csv",
        "failures_summary.csv",
        "raw_vs_kept_by_tag.csv",
        "algorithm_family_per_instance.csv",
        "algorithm_family_by_tag.csv",
        "algorithm_diversity_summary.csv",
        "algorithm_signals_per_instance.csv",
    ]
    out = {}
    for name in expected:
        p = diversity_dir / name
        if p.exists():
            out[name] = str(p)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce C2Q paper artifacts end-to-end.")
    parser.add_argument("--mode", choices=["paper", "smoke"], default="paper")
    parser.add_argument("--primary-csv", type=Path, default=default_primary_csv())
    parser.add_argument("--backup-csv", type=Path, default=default_backup_csv())
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--time-limit-secs", type=int, default=300)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Delete old artifacts under output-root/<mode> before running (default: true).",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Keep existing artifacts and append/overwrite as needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "paper":
        print("NOTE: paper mode generates up to 434 reports and is time-consuming.")

    input_csv = resolve_input_csv(args.primary_csv, args.backup_csv)
    validate_model_dir(args.model_path)

    run_dir = args.output_root / args.mode
    impl_dir = run_dir / "tables" / "implementation"
    diversity_dir = run_dir / "tables" / "diversity"
    reports_dir = run_dir / "reports"
    metadata_dir = run_dir / "metadata"

    if args.clean and run_dir.exists():
        shutil.rmtree(run_dir)

    for path in (impl_dir, diversity_dir, reports_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root()}" if not env.get("PYTHONPATH") else f"{repo_root()}{os.pathsep}{env['PYTHONPATH']}"
    )
    env["PYTHONHASHSEED"] = env.get("PYTHONHASHSEED", "0")
    env["C2Q_MODEL_PATH"] = str(args.model_path)
    env["C2Q_INPUT_CSV"] = str(input_csv)
    env["C2Q_BACKUP_CSV"] = str(args.backup_csv)
    env["C2Q_DIVERSITY_OUT"] = str(diversity_dir)

    python_exec = sys.executable
    started_at = int(time.time())

    run_cmd(
        "Implementation-level validation",
        [
            python_exec,
            str(repo_root() / "src" / "validation" / "implementation_validation.py"),
            "--csv-path",
            str(input_csv),
            "--backup-csv-path",
            str(args.backup_csv),
            "--out-snippet-metrics",
            str(impl_dir / "snippet_metrics.csv"),
            "--out-family-summary",
            str(impl_dir / "family_summary.csv"),
            "--out-syntax-failures",
            str(impl_dir / "syntax_failures.csv"),
        ],
        env,
    )

    run_cmd(
        "Algorithmic/structural diversity validation",
        [
            python_exec,
            "-m",
            "unittest",
            "src.validation.diversity_validation.MyTestCase.test_diversity_direct_metrics",
        ],
        env,
    )

    report_cmd = [
        python_exec,
        str(repo_root() / "src" / "tests" / "tests_reports.py"),
        "--primary-csv",
        str(input_csv),
        "--backup-csv",
        str(args.backup_csv),
        "--output-dir",
        str(reports_dir),
        "--model-path",
        str(args.model_path),
        "--time-limit-secs",
        str(args.time_limit_secs),
        "--mode",
        args.mode,
    ]
    if args.clean:
        report_cmd.append("--clean-output")
    if args.max_cases is not None:
        report_cmd.extend(["--max-cases", str(args.max_cases)])

    run_cmd("Report generation", report_cmd, env)

    report_outputs = {
        "manifest": str(reports_dir / "MANIFEST.csv"),
        "checksums": str(reports_dir / "checksums.txt"),
        "summary_status": str(reports_dir / "summary_status.csv"),
        "summary_problem_type": str(reports_dir / "summary_problem_type.csv"),
        "figure_status_counts": str(reports_dir / "figure_status_counts.png"),
        "figure_problem_type_counts": str(reports_dir / "figure_problem_type_counts.png"),
    }

    metadata = {
        "mode": args.mode,
        "time_started_epoch": started_at,
        "time_finished_epoch": int(time.time()),
        "git_revision": get_git_rev(),
        "python_executable": python_exec,
        "input_csv_selected": str(input_csv),
        "primary_csv": str(args.primary_csv),
        "backup_csv": str(args.backup_csv),
        "model_path": str(args.model_path),
        "time_limit_secs": args.time_limit_secs,
        "max_cases": args.max_cases,
        "output_root": str(args.output_root),
    }

    write_json(metadata_dir / "run_metadata.json", metadata)

    summary_lines = [
        f"mode: {args.mode}",
        f"input_csv_selected: {input_csv}",
        f"implementation_metrics: {impl_dir / 'snippet_metrics.csv'}",
        f"implementation_family_summary: {impl_dir / 'family_summary.csv'}",
        f"implementation_syntax_failures: {impl_dir / 'syntax_failures.csv'}",
    ]
    diversity_outputs = collect_diversity_outputs(diversity_dir)
    summary_lines.append("diversity_outputs:")
    for key, val in sorted(diversity_outputs.items()):
        summary_lines.append(f"  - {key}: {val}")
    summary_lines.append("report_outputs:")
    for key, val in sorted(report_outputs.items()):
        summary_lines.append(f"  - {key}: {val}")

    write_text(run_dir / "ARTIFACTS.txt", summary_lines)

    final_index = {
        "implementation": {
            "snippet_metrics": str(impl_dir / "snippet_metrics.csv"),
            "family_summary": str(impl_dir / "family_summary.csv"),
            "syntax_failures": str(impl_dir / "syntax_failures.csv"),
        },
        "diversity": diversity_outputs,
        "reports": report_outputs,
        "metadata": str(metadata_dir / "run_metadata.json"),
    }
    write_json(run_dir / "artifacts.json", final_index)

    print("\nReproducibility run completed.")
    print(f"Artifacts index: {run_dir / 'artifacts.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
