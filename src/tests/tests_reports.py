import os
import unittest
import ast
import csv
import time
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict
import functools
from multiprocessing import Process, Queue, set_start_method

from src.parser.parser import Parser, PROBLEMS

log = functools.partial(print, flush=True)


# ----------------- LaTeX helpers -----------------
def latex_escape(s: str) -> str:
    """Minimal LaTeX escaping for free-form strings."""
    return (s.replace('\\', r'\textbackslash{}')
             .replace('&', r'\&')
             .replace('%', r'\%')
             .replace('$', r'\$')
             .replace('#', r'\#')
             .replace('_', r'\_')
             .replace('{', r'\{')
             .replace('}', r'\}')
             .replace('~', r'\textasciitilde{}')
             .replace('^', r'\textasciicircum{}'))


def compile_tex_to_pdf(tex_stem: Path) -> bool:
    """
    Compile <stem>.tex → <stem>.pdf via pdflatex. Returns True on success.
    IMPORTANT: pass a STEM (no extension). This function appends .tex itself.
    """
    try:
        tex_path = tex_stem.with_suffix(".tex")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={tex_path.parent}", str(tex_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        return True
    except Exception:
        return False


def write_placeholder_pdf(base_stem: Path, reason: str = "Timed out") -> None:
    """
    Emit a tiny placeholder report (TEX → try to compile → PDF) for traceability.
    base_stem must be a STEM path (no extension).
    """
    tex_path = base_stem.with_suffix(".tex")
    tex = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{lmodern}\n"
        "\\begin{document}\n"
        "\\section*{Report Placeholder}\n"
        "This report was not generated because: \\emph{" + latex_escape(reason) + "}.\n"
        "\\end{document}\n"
    )
    tex_path.write_text(tex, encoding="utf-8")
    compiled = compile_tex_to_pdf(base_stem)
    print(
        f"⚠️  Wrote placeholder {'PDF' if compiled else 'TEX'} → "
        f"{tex_path.with_suffix('.pdf' if compiled else '.tex').name}"
    )


def safe_tex_stem(problem_type: str, count: int) -> str:
    """Return a filename STEM like MIS, MIS_2, etc. (no extension)."""
    suffix = f"_{count}" if count > 1 else ""
    return f"{problem_type}{suffix}"


# ----------------- IDs & checksums -----------------
def hash_snippet(s: str) -> str:
    """Short stable id for a snippet (first 12 hex chars of SHA256)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(report_dir: Path) -> None:
    """
    Write checksums.txt with SHA-256 for all PDFs (and TEX as a courtesy).
    """
    out_path = report_dir / "checksums.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        # PDFs first (primary artifacts)
        for pdf in sorted(report_dir.glob("*.pdf")):
            fh.write(f"{sha256sum(pdf)}  {pdf.name}\n")
        # TEX second (optional)
        for tex in sorted(report_dir.glob("*.tex")):
            fh.write(f"{sha256sum(tex)}  {tex.name}\n")
    log(f"✅ checksums written → {out_path}")


# ----------------- Worker (runs in a separate process) -----------------
def _render_one_case_worker(item, out_dir_str, case_idx, q):
    """
    Worker that runs in a separate process. It parses the snippet, builds the report,
    and (optionally) compiles it. Results are returned via a Queue.
    """
    try:
        # Reduce threading in native backends (Aer, BLAS, etc.) for stability:
        os.environ["QISKIT_PARALLEL"] = "FALSE"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        # Local imports inside worker to keep the parent clean
        import ast
        from pathlib import Path
        from src.parser.parser import Parser, PROBLEMS

        out_dir = Path(out_dir_str)
        parser = Parser(model_path="../parser/saved_models")

        # Decode CSV cell to a Python string/code snippet
        try:
            clean_code = ast.literal_eval(item)
        except Exception:
            clean_code = item.encode().decode('unicode_escape')

        # Parse → problem
        problem_type, data = parser.parse(clean_code)

        # If parser returns unregistered type, emit placeholder and return
        if problem_type not in PROBLEMS:
            reason = f"Unregistered problem type: {problem_type}"
            base_stem = out_dir / f"UNKNOWN_{case_idx}"
            write_placeholder_pdf(base_stem, reason=reason)
            q.put({
                "ok": False,
                "problem_type": "UNKNOWN",
                "reason": reason,
                "tex_path": str(base_stem.with_suffix(".tex").resolve()),
                "pdf_path": str(base_stem.with_suffix(".pdf").resolve()),
                "compiled_pdf": int(base_stem.with_suffix(".pdf").exists()),
            })
            return

        problem = PROBLEMS[problem_type](data)

        # Name by case (so the parent doesn't have to track counts here)
        tex_stem = (Path(out_dir) / f"CASE_{case_idx}_{problem_type}")

        # Generate LaTeX (your implementation should accept a STEM path string)
        # If your implementation expects a full path ending with .tex, this still works:
        problem.report_latex(output_path=str(tex_stem))

        # Compile
        compiled = compile_tex_to_pdf(tex_stem)

        q.put({
            "ok": True,
            "problem_type": problem_type,
            "reason": "",
            "tex_path": str(tex_stem.with_suffix(".tex").resolve()),
            "pdf_path": str(tex_stem.with_suffix(".pdf").resolve()),
            "compiled_pdf": int(bool(compiled)),
        })
    except Exception as e:
        # Any exception → placeholder in parent; here we just return error info
        q.put({
            "ok": False,
            "problem_type": "UNKNOWN",
            "reason": f"{type(e).__name__}: {e}",
            "tex_path": "",
            "pdf_path": "",
            "compiled_pdf": 0,
        })


# ----------------- Test case -----------------
class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Parser in parent for the small smoke test (not used in worker)
        self.parser = Parser(model_path="../parser/saved_models")

    def test2(self):
        """
        Batch-generate LaTeX/PDF reports from ../parser/data.csv (first column),
        using a **separate process per case** with a strict wall-clock timeout.
        Produces MANIFEST.csv and checksums.txt.
        """
        input_csv = '../parser/data.csv'
        out_dir = Path("../c2q-dataset/reports/pdf")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Read inputs
        rows = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            for row in r:
                if row:
                    rows.append(row[0])

        # Manifest setup
        manifest_path = out_dir / "MANIFEST.csv"
        mf = open(manifest_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "case_idx", "snippet_id", "problem_type", "status", "reason",
            "tex_path", "pdf_path", "compiled_pdf", "duration_sec"
        ]
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader(); mf.flush()

        # Time budget
        TIME_LIMIT_SECS = 4 * 60  # 4 minutes per item

        unknown_count = 0

        for case_idx, item in enumerate(rows, start=1):
            start = time.time()
            snippet_for_hash = item if isinstance(item, str) else str(item)
            snippet_id = hash_snippet(snippet_for_hash)

            # Launch worker
            q = Queue()
            p = Process(target=_render_one_case_worker, args=(item, str(out_dir), case_idx, q))
            p.start()
            p.join(TIME_LIMIT_SECS)

            status = "ok"
            reason = ""
            problem_type = "UNKNOWN"
            tex_path = ""
            pdf_path = ""
            compiled_pdf = 0

            if p.is_alive():
                # Timeout → kill worker and emit placeholder
                p.terminate()
                p.join()
                status = "timeout"
                reason = f"Exceeded {TIME_LIMIT_SECS}s"
                base_stem = out_dir / f"TIMEOUT_{case_idx}"
                write_placeholder_pdf(base_stem, reason="Generation exceeded the 4-minute limit")
                tex_path = str(base_stem.with_suffix(".tex").resolve())
                pdf_path = str(base_stem.with_suffix(".pdf").resolve())
                compiled_pdf = int(Path(pdf_path).exists())
                log(f"⏱️  [{case_idx:04d}] Timeout; placeholder emitted.")
            else:
                res = q.get() if not q.empty() else {"ok": False, "reason": "No result from worker"}
                status = "ok" if res.get("ok") else "error"
                reason = res.get("reason", "")
                problem_type = res.get("problem_type", "UNKNOWN")
                tex_path = res.get("tex_path", "")
                pdf_path = res.get("pdf_path", "")
                compiled_pdf = int(bool(res.get("compiled_pdf", 0)))

                if status == "ok":
                    if compiled_pdf:
                        log(f"✅ [{case_idx:04d}] {problem_type} → {Path(pdf_path).name}")
                    else:
                        log(f"✅ [{case_idx:04d}] {problem_type} → {Path(tex_path).name} (pdflatex not found?)")
                else:
                    # Emit placeholder for error cases to keep artifacts complete
                    base_stem = out_dir / f"ERROR_{case_idx}_{problem_type}"
                    write_placeholder_pdf(base_stem, reason=(reason or "Worker exception"))
                    tex_path = str(base_stem.with_suffix(".tex").resolve())
                    pdf_path = str(base_stem.with_suffix(".pdf").resolve())
                    compiled_pdf = int(Path(pdf_path).exists())
                    snippet = (snippet_for_hash[:60]).replace('\n', ' ')
                    log(f"❌  [{case_idx:04d}] Worker failed for snippet '{snippet}' — {reason}")

                if problem_type.lower() == "unknown":
                    unknown_count += 1

            duration = round(time.time() - start, 3)
            writer.writerow({
                "case_idx": case_idx,
                "snippet_id": snippet_id,
                "problem_type": problem_type,
                "status": status,
                "reason": reason,
                "tex_path": tex_path,
                "pdf_path": pdf_path,
                "compiled_pdf": compiled_pdf,
                "duration_sec": duration
            })
            mf.flush()

        mf.close()
        log(f"Done. Unknown classifications: {unknown_count}")
        log(f"Manifest written → {manifest_path.resolve()}")

        # checksums for everything we just produced
        write_checksums(out_dir)

    # (Optional) single-case smoke test
    def test_clique(self):
        code = "import networkx as nx\n\ndef clique_networkx_backtracking(G):\n    def is_clique(nodes):\n        return all(G.has_edge(u, v) for u in nodes for v in nodes if u != v)\n    def backtrack(node, current_clique):\n        if node == len(G.nodes):\n            return current_clique\n        if is_clique(current_clique + [node]):\n            with_node = backtrack(node + 1, current_clique + [node])\n            without_node = backtrack(node + 1, current_clique)\n            return max(with_node, without_node, key=len)\n        return backtrack(node + 1, current_clique)\n    return backtrack(0, [])\n\nG = nx.Graph()\nG.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])\nclique_networkx_backtracking(G)"
        try:
            clean_code = ast.literal_eval(code)
        except Exception:
            clean_code = code.encode().decode('unicode_escape')
        problem_type, data = self.parser.parse(clean_code)
        if problem_type in PROBLEMS:
            problem = PROBLEMS[problem_type](data)
            out_dir = Path("../c2q-dataset/reports/pdf")
            out_dir.mkdir(parents=True, exist_ok=True)
            tex_stem = out_dir / "SINGLE_TEST"
            problem.report_latex(output_path=str(tex_stem))
            compile_tex_to_pdf(tex_stem)
        else:
            log(f"Parser returned unregistered type in test_clique: {problem_type}")


    def test_factor(self):
        code = "def factorization_recursive(n):\n    def recursive_division(n, divisor=2):\n        if n == 1:\n            return []\n        if n % divisor == 0:\n            return [divisor] + recursive_division(n // divisor, divisor)\n        return recursive_division(n, divisor + 1)\n    return recursive_division(n)\n\n# Input data\nn = 98\nfactors = factorization_recursive(n)\nprint(factors)"
        try:
            clean_code = ast.literal_eval(code)
        except Exception:
            clean_code = code.encode().decode('unicode_escape')
        problem_type, data = self.parser.parse(clean_code)
        if problem_type in PROBLEMS:
            problem = PROBLEMS[problem_type](data)
            out_dir = Path("../c2q-dataset/reports/pdf")
            out_dir.mkdir(parents=True, exist_ok=True)
            tex_stem = out_dir / "SINGLE_TEST"
            problem.report_latex(output_path=str(tex_stem))
            compile_tex_to_pdf(tex_stem)
        else:
            log(f"Parser returned unregistered type in test_clique: {problem_type}")


if __name__ == '__main__':
    # On macOS, using 'spawn' avoids forking the parent (safer with native libs).
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    unittest.main(verbosity=2)