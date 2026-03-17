"""
Paper-scale report generation harness.

This module is intentionally separate from fast tests because the full paper run is
very time-consuming: it can process 434 snippets, each with a multi-minute timeout.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import os
import random
import subprocess
import time
import unittest
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Process, Queue, set_start_method
from pathlib import Path
from typing import Dict, List

GRAPH_FAMILIES = {"MaxCut", "MIS", "TSP", "Clique", "KColor", "VC"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_primary_csv() -> Path:
    return repo_root() / "src" / "parser" / "python_programs.csv"


def default_backup_csv() -> Path:
    return repo_root() / "src" / "parser" / "data.csv"


def resolve_input_csv(primary_csv: Path, backup_csv: Path) -> Path:
    """Prefer python_programs.csv; fall back to data.csv as backup."""
    if primary_csv.exists():
        return primary_csv
    if backup_csv.exists():
        return backup_csv
    raise FileNotFoundError(
        "No input CSV found. Checked primary and backup:\n"
        f" - {primary_csv}\n"
        f" - {backup_csv}"
    )


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


def default_output_dir() -> Path:
    return repo_root() / "src" / "c2q-dataset" / "reports" / "code_reports" / "pdf"


def stable_snippet_id(snippet: str) -> str:
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:12]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def parse_snippet_cell(item: str) -> str:
    try:
        return ast.literal_eval(item)
    except Exception:
        return item.encode().decode("unicode_escape")


def load_snippet_cells(input_csv: Path) -> List[str]:
    """
    Load snippet cells from CSV.
    Prefers a header column named `code_snippet`; falls back to first column.
    """
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        dict_reader = csv.DictReader(f)
        if dict_reader.fieldnames and "code_snippet" in dict_reader.fieldnames:
            return [row.get("code_snippet", "") for row in dict_reader if row.get("code_snippet", "")]

        f.seek(0)
        rows: List[str] = []
        for idx, row in enumerate(csv.reader(f)):
            if not row:
                continue
            cell = row[0]
            if idx == 0 and cell.strip().lower() == "code_snippet":
                continue
            rows.append(cell)
        return rows


def compile_tex_to_pdf(tex_stem: Path) -> bool:
    tex_path = tex_stem.with_suffix(".tex")
    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={tex_path.parent}",
                str(tex_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        return True
    except Exception:
        return False


def write_placeholder_pdf(base_stem: Path, reason: str) -> None:
    tex_path = base_stem.with_suffix(".tex")
    tex = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{lmodern}\n"
        "\\begin{document}\n"
        "\\section*{Report Placeholder}\n"
        f"This report was not generated because: \\emph{{{latex_escape(reason)}}}.\n"
        "\\end{document}\n"
    )
    tex_path.write_text(tex, encoding="utf-8")
    compile_tex_to_pdf(base_stem)


def write_checksums(output_dir: Path) -> Path:
    out = output_dir / "checksums.txt"
    with out.open("w", encoding="utf-8") as fh:
        for pdf in sorted(output_dir.glob("*.pdf")):
            fh.write(f"{sha256sum(pdf)}  {pdf.name}\n")
        for tex in sorted(output_dir.glob("*.tex")):
            fh.write(f"{sha256sum(tex)}  {tex.name}\n")
    return out


def write_summary_tables_and_figures(manifest_path: Path, output_dir: Path) -> Dict[str, Path]:
    with manifest_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    status_counts = Counter(r.get("status", "unknown") for r in rows)
    problem_counts = Counter(r.get("problem_type", "UNKNOWN") for r in rows)

    duration_by_problem: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        problem = r.get("problem_type", "UNKNOWN")
        try:
            duration_by_problem[problem].append(float(r.get("duration_sec", 0)))
        except Exception:
            continue

    status_csv = output_dir / "summary_status.csv"
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "count"])
        writer.writeheader()
        for key, count in sorted(status_counts.items()):
            writer.writerow({"status": key, "count": count})

    problem_csv = output_dir / "summary_problem_type.csv"
    with problem_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["problem_type", "count", "avg_duration_sec", "max_duration_sec"],
        )
        writer.writeheader()
        for key, count in sorted(problem_counts.items()):
            durations = duration_by_problem.get(key, [])
            avg_d = round(sum(durations) / len(durations), 3) if durations else 0.0
            max_d = round(max(durations), 3) if durations else 0.0
            writer.writerow(
                {
                    "problem_type": key,
                    "count": count,
                    "avg_duration_sec": avg_d,
                    "max_duration_sec": max_d,
                }
            )

    figure_paths: Dict[str, Path] = {}
    try:
        import matplotlib.pyplot as plt

        status_fig = output_dir / "figure_status_counts.png"
        plt.figure(figsize=(8, 4))
        plt.bar(list(status_counts.keys()), list(status_counts.values()))
        plt.title("Report Generation Status Counts")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(status_fig)
        plt.close()
        figure_paths["status_figure"] = status_fig

        labels = sorted(problem_counts.keys())
        values = [problem_counts[l] for l in labels]
        problem_fig = output_dir / "figure_problem_type_counts.png"
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha="right")
        plt.title("Generated Reports by Problem Type")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(problem_fig)
        plt.close()
        figure_paths["problem_figure"] = problem_fig
    except Exception:
        pass

    out = {
        "summary_status": status_csv,
        "summary_problem_type": problem_csv,
    }
    out.update(figure_paths)
    return out


@dataclass
class BatchRunConfig:
    input_csv: Path
    output_dir: Path
    model_path: Path
    time_limit_secs: int = 300
    max_cases: int | None = None
    clean_output: bool = False


def _worker_render_case(snippet_cell: str, case_idx: int, model_path: str, output_dir: str, q: Queue) -> None:
    from src.parser.parser import Parser, PROBLEMS

    os.environ["QISKIT_PARALLEL"] = "FALSE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    problem_type = "UNKNOWN"
    tex_stem = Path(output_dir) / f"CASE_{case_idx}_UNKNOWN"

    try:
        parser = Parser(model_path=model_path)
        clean_code = parse_snippet_cell(snippet_cell)
        problem_type, data = parser.parse(clean_code)

        if problem_type not in PROBLEMS:
            reason = f"Unregistered problem type: {problem_type}"
            write_placeholder_pdf(tex_stem, reason=reason)
            q.put(
                {
                    "ok": False,
                    "problem_type": "UNKNOWN",
                    "reason": reason,
                    "pdf_path": str(tex_stem.with_suffix(".pdf")),
                    "tex_path": str(tex_stem.with_suffix(".tex")),
                }
            )
            return

        if problem_type in GRAPH_FAMILIES and hasattr(data, "G"):
            problem = PROBLEMS[problem_type](data.G)
        else:
            problem = PROBLEMS[problem_type](data)

        tex_stem = Path(output_dir) / f"CASE_{case_idx}_{problem_type}"
        problem.report_latex(output_path=str(tex_stem))

        q.put(
            {
                "ok": True,
                "problem_type": problem_type,
                "reason": "",
                "pdf_path": str(tex_stem.with_suffix(".pdf")),
                "tex_path": str(tex_stem.with_suffix(".tex")),
            }
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        write_placeholder_pdf(tex_stem, reason=reason)
        q.put(
            {
                "ok": False,
                "problem_type": problem_type,
                "reason": reason,
                "pdf_path": str(tex_stem.with_suffix(".pdf")),
                "tex_path": str(tex_stem.with_suffix(".tex")),
            }
        )


def run_batch_report_generation(config: BatchRunConfig) -> Dict[str, Path]:
    if not config.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {config.input_csv}")
    if not config.model_path.exists():
        raise FileNotFoundError(
            "Parser model directory not found. Expected fine-tuned model at "
            f"{config.model_path}. Set C2Q_MODEL_PATH or pass --model-path.\n"
            "Download model: https://zenodo.org/records/19061126/files/saved_models_2025_12.zip?download=1"
        )
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (config.model_path / name).is_file()]
    has_weights = (
        (config.model_path / "model.safetensors").is_file()
        or (config.model_path / "pytorch_model.bin").is_file()
    )
    if missing:
        raise FileNotFoundError(
            "Parser model directory is incomplete at "
            f"{config.model_path}. Missing: {', '.join(missing)}.\n"
            "Download model: https://zenodo.org/records/19061126/files/saved_models_2025_12.zip?download=1"
        )
    if not has_weights:
        raise FileNotFoundError(
            "Parser model directory is missing weights at "
            f"{config.model_path}. Missing one of: model.safetensors, pytorch_model.bin.\n"
            "Download model: https://zenodo.org/records/19061126/files/saved_models_2025_12.zip?download=1"
        )

    random.seed(0)
    os.environ.setdefault("PYTHONHASHSEED", "0")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.clean_output:
        for pattern in ("*.pdf", "*.tex", "*.csv", "*.png", "*.txt"):
            for file_path in config.output_dir.glob(pattern):
                file_path.unlink()

    rows = load_snippet_cells(config.input_csv)

    if config.max_cases is not None:
        rows = rows[: config.max_cases]

    total = len(rows)
    if total == 0:
        raise ValueError("No rows found in input CSV.")

    if config.max_cases is None and total >= 400:
        print(
            "WARNING: Paper-scale run detected: 434 snippets with per-case timeout. "
            "This is time-consuming and may take many hours."
        )

    manifest_path = config.output_dir / "MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "case_idx",
                "snippet_id",
                "problem_type",
                "status",
                "reason",
                "tex_path",
                "pdf_path",
                "compiled_pdf",
                "duration_sec",
            ],
        )
        writer.writeheader()

        for idx, cell in enumerate(rows, start=1):
            start = time.time()
            snippet = parse_snippet_cell(cell)
            snippet_id = stable_snippet_id(snippet)

            q: Queue = Queue()
            p = Process(
                target=_worker_render_case,
                args=(cell, idx, str(config.model_path), str(config.output_dir), q),
            )
            p.start()
            p.join(config.time_limit_secs)

            status = "ok"
            reason = ""
            problem_type = "UNKNOWN"
            tex_path = ""
            pdf_path = ""
            compiled_pdf = 0

            if p.is_alive():
                p.terminate()
                p.join()
                status = "timeout"
                reason = f"Exceeded {config.time_limit_secs}s"
                timeout_stem = config.output_dir / f"TIMEOUT_{idx}_UNKNOWN"
                write_placeholder_pdf(timeout_stem, reason=reason)
                tex_path = str(timeout_stem.with_suffix(".tex"))
                pdf_path = str(timeout_stem.with_suffix(".pdf"))
                compiled_pdf = int(Path(pdf_path).exists())
                print(f"[TIMEOUT] [{idx:04d}/{total}] timeout")
            else:
                res = q.get() if not q.empty() else {"ok": False, "reason": "No result from worker"}
                status = "ok" if res.get("ok") else "error"
                reason = res.get("reason", "")
                problem_type = res.get("problem_type", "UNKNOWN")
                tex_path = res.get("tex_path", "")
                pdf_path = res.get("pdf_path", "")
                compiled_pdf = int(Path(pdf_path).exists()) if pdf_path else 0

                label = "[OK]" if status == "ok" else "[ERR]"
                print(f"{label} [{idx:04d}/{total}] {problem_type} {status}")

            duration = round(time.time() - start, 3)
            writer.writerow(
                {
                    "case_idx": idx,
                    "snippet_id": snippet_id,
                    "problem_type": problem_type,
                    "status": status,
                    "reason": reason,
                    "tex_path": tex_path,
                    "pdf_path": pdf_path,
                    "compiled_pdf": compiled_pdf,
                    "duration_sec": duration,
                }
            )

    checksums_path = write_checksums(config.output_dir)
    summary = write_summary_tables_and_figures(manifest_path, config.output_dir)

    return {
        "manifest": manifest_path,
        "checksums": checksums_path,
        **summary,
    }


class MyTestCase(unittest.TestCase):
    def test_generate_paper_reports(self):
        """
        Full 434-case run for paper artifacts.
        Disabled by default because this run is time-consuming.
        """
        if os.getenv("C2Q_ENABLE_PAPER_REPORT_TEST") != "1":
            self.skipTest("Set C2Q_ENABLE_PAPER_REPORT_TEST=1 to run full paper report generation.")

        primary_csv = Path(os.getenv("C2Q_PRIMARY_CSV", str(default_primary_csv())))
        backup_csv = Path(os.getenv("C2Q_BACKUP_CSV", str(default_backup_csv())))
        input_csv = resolve_input_csv(primary_csv, backup_csv)

        result = run_batch_report_generation(
            BatchRunConfig(
                input_csv=input_csv,
                output_dir=default_output_dir(),
                model_path=default_model_path(),
                time_limit_secs=int(os.getenv("C2Q_REPORT_TIMEOUT", "300")),
                max_cases=None,
                clean_output=True,
            )
        )
        self.assertTrue(result["manifest"].exists())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible report artifacts from snippet CSV.")
    parser.add_argument("--primary-csv", type=Path, default=default_primary_csv())
    parser.add_argument("--backup-csv", type=Path, default=default_backup_csv())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path(),
    )
    parser.add_argument("--time-limit-secs", type=int, default=300)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove previous generated artifacts in output-dir before running.",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "smoke"],
        default="paper",
        help="smoke mode limits generation to 4 cases.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    input_csv = resolve_input_csv(args.primary_csv, args.backup_csv)
    max_cases = args.max_cases
    if args.mode == "smoke" and max_cases is None:
        max_cases = 4

    try:
        outputs = run_batch_report_generation(
            BatchRunConfig(
                input_csv=input_csv,
                output_dir=args.output_dir,
                model_path=args.model_path,
                time_limit_secs=args.time_limit_secs,
                max_cases=max_cases,
                clean_output=args.clean_output,
            )
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}")

    print("\nGenerated artifacts:")
    print(f" - input_csv: {input_csv}")
    for key, path in outputs.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
