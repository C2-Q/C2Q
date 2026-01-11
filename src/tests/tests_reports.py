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
    # problem_type = "UNKNOWN"
    global problem_type
    try:
        problem_type = "UNKNOWN"
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
            "problem_type": problem_type, # UNKNOWN->problem_type
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

    def test1(self):
        """
        Batch-generate LaTeX/PDF reports from ../parser/data.csv (first column),
        using a **separate process per case** with a strict wall-clock timeout.
        Produces MANIFEST.csv and checksums.txt.
        """
        input_csv = '../parser/data2.csv'

        # Read inputs
        rows = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            for row in r:
                if row:
                    rows.append(row[0])

        # Manifest setup

        for case_idx, item in enumerate(rows, start=1):
            try:
                clean_code = ast.literal_eval(item)
            except Exception:
                clean_code = item.encode().decode('unicode_escape')

            tag, data = self.parser.parse(clean_code)
            print(tag, data)
    def test2(self):
        """
        Batch-generate LaTeX/PDF reports from ../parser/data.csv (first column),
        using a **separate process per case** with a strict wall-clock timeout.
        Produces MANIFEST.csv and checksums.txt.
        """
        input_csv = '../parser/python_programs.csv'
        # out_dir = Path("../c2q-dataset/reports/pdf")
        out_dir = Path("../c2q-dataset/reports/code_reports/pdf")
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
        TIME_LIMIT_SECS = 5 * 60  # 4 minutes per item

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
                problem_type = res.get("problem_type", "UNKNOWN")
                base_stem = out_dir / f"TIMEOUT_{case_idx}_{problem_type}"
                write_placeholder_pdf(base_stem, reason=f"Generation exceeded the 4-minute limit: {problem_type}")
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
            out_dir = Path("../c2q-dataset/reports/reports/pdf")
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
            out_dir = Path("../c2q-dataset/reports/reports/pdf")
            out_dir.mkdir(parents=True, exist_ok=True)
            tex_stem = out_dir / "SINGLE_TEST"
            problem.report_latex(output_path=str(tex_stem))
            compile_tex_to_pdf(tex_stem)
        else:
            log(f"Parser returned unregistered type in test_clique: {problem_type}")

    def test_mis(self):
        is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input json\nedges = [(0, 1), (0, 2), (1, 2), (1, 3)]\nindependent_set = independent_nodes(2, edges)\nprint(independent_set)"
        is_snippet_main = """
            from itertools import combinations

def solve_graph_problem(n, edges):
    nodes = list(range(n))

    def is_independent(subset):
        return all(
            (u, v) not in edges and (v, u) not in edges
            for u, v in combinations(subset, 2)
        )

    best = set()
    for r in range(1, n + 1):
        for subset in combinations(nodes, r):
            if is_independent(subset):
                best = set(subset)
    return best


# Maximum Independent Set problem specification
edges = [(0, 1), (0, 2), (1, 2), (1, 3)]
result = solve_graph_problem(4, edges)
print(result)  # {2, 3}
        """
        tag, data = self.parser.parse(is_snippet)
        print(tag, data)
        mis = PROBLEMS[tag](data.G)
        mis.report_latex()

    def test_mul(self):
        mul_snippet = "def mul_accumulate_pairs(pairs):\n    results = []\n    for x, y in pairs:\n        prod = x * y\n        results.append(prod)\n    return results\n\npairs = [(15, 7), (31, 9), (64, 3)]\nprint(mul_accumulate_pairs(pairs))"
        tag, data = self.parser.parse(mul_snippet)
        print(tag, data)
        mis = PROBLEMS[tag](data)
        mis.report_latex()

    def test_mis2(self):
        is_snippet_new = "def mis_backtracking(n, edges):\n    # Backtracking MIS over adjacency set\n    best=[]\n    E=set(edges)\n    def backtrack(i, current, banned):\n        nonlocal best\n        if i==n:\n            if len(current) > len(best):\n                best = list(current)\n            return\n        if i in banned:\n            backtrack(i+1, current, banned)\n            return\n        # Option 1: skip i\n        backtrack(i+1, current, banned)\n        # Option 2: include i if no conflict\n        for v in current:\n            if (v, i) in E or (i, v) in E:\n                break\n        else:\n            new_banned = set(banned)\n            new_banned.add(i)\n            for v in range(n):\n                if (v, i) in E or (i, v) in E:\n                    new_banned.add(v)\n            current.append(i)\n            backtrack(i+1, current, new_banned)\n            current.pop()\n    backtrack(0, [], set())\n    return best\n\nedges=[(0,1),(1,2),(2,3),(3,4),(0,4)]\nprint(mis_backtracking(5, edges))"
        is_snippet_old = "def greedy_mis(adj):\n    remain=set(adj.keys())\n    indep=[]\n    while remain:\n        v=min(remain)\n        indep.append(v)\n        rem=[v]+adj[v]\n        for x in rem:\n            if x in remain: remain.remove(x)\n    return indep\n\nadj={0:[1,2],1:[0,3],2:[0,3],3:[1,2,4],4:[3]}\nprint(greedy_mis(adj))"
        maxcut_snippet_new = "def partition(n,mask):\n    l=[]; r=[]; i=0\n    while i<n:\n        if (mask>>i)&1: l.append(i)\n        else: r.append(i)\n        i+=1\n    return l,r\n\ndef bruteforce_edge_maxcut(n,edges):\n    best=-1; part=None; m=1\n    while m<(1<<n):\n        left,right=partition(n,m)\n        val=0\n        for u,v,w in edges:\n            if (u in left and v in right) or (u in right and v in left): val+=w\n        if val>best:\n            best=val\n            part=(left,right)\n        m+=1\n    return best,part\n\nedges=[(0,1,1),(1,3,1),(3,2,2),(2,0,1)]\nprint(bruteforce_edge_maxcut(4,edges))"
        maxcut_snippet_old = "def maximum_cut_randomized(edges, n):\n    import random\n    set_A, set_B = set(), set()\n    for i in range(n):\n        if random.random() > 0.5:\n            set_A.add(i)\n        else:\n            set_B.add(i)\n    return sum(1 for u, v in edges if (u in set_A and v in set_B) or (u in set_B and v in set_A)), set_A, set_B\n\nedges = [(0, 1), (1, 2), (2, 3), (3, 0)]\nmaximum_cut_randomized(edges, 4)"
        # add_code = "class AddOp:\n    def __init__(self, x, y):\n        self.x, self.y = x, y\n    def compute_addition(self):\n        return self.x + self.y\n\n# Input data\nop = AddOp(12, -5)\nresult = op.compute_addition()\nprint(result)"
        # add_code = "def helper(values):\n    s = 0\n    for v in values:\n        s += v\n    return s\n\ndef compute_addition(a, b):\n    if a == b:\n        return a + b\n    return helper([a, b])\n\n# Input data\na, b = 9, 4\nresult = compute_addition(a, b)\nprint(result)"
        # sub_code = "def compute_subtraction(a, b):\n    r = a\n    y = b\n    while y > 0:\n        r -= 1\n        y -= 1\n    return r\n\n# Input data\na, b = 15, 6\nresult = compute_subtraction(a, b)\nprint(result)"
        # sub_code = "def dir_flag(x, y):\n    return 1 if x >= y else -1\n\ndef compute_subtraction(a, b):\n    direction = dir_flag(a, b)\n    raw = a - b\n    if direction < 0:\n        temp = -raw\n        return a - temp\n    return raw\n\n# Input data\na, b = -3, 9\nresult = compute_subtraction(a, b)\nprint(result)"
        # sub_code = "def compute_subtraction(a, b):\n    if b == 0:\n        return a\n    if b > 0:\n        return compute_subtraction(a - 1, b - 1)\n    return compute_subtraction(a + 1, b + 1)\n\n# Input data\na, b = 20, -3\nresult = compute_subtraction(a, b)\nprint(result)"
        # sub_code = "def compute_multiplication(a, b):\n    if a == 0 or b == 0:\n        return 0\n    neg = (a < 0) ^ (b < 0)\n    x, y = abs(a), abs(b)\n    r = 0\n    i = 0\n    while i < y:\n        r += x\n        i += 1\n    return -r if neg else r\n\n# Input data\na, b = -4, 7\nresult = compute_multiplication(a, b)\nprint(result)"
        # sub_code = "def compute_multiplication(a, b):\n    neg = (a < 0) ^ (b < 0)\n    x, y = abs(a), abs(b)\n    r = 0\n    while y > 0:\n        if y & 1:\n            r += x\n        x <<= 1\n        y >>= 1\n    return -r if neg else r\n\n# Input data\na, b = 13, -5\nresult = compute_multiplication(a, b)\nprint(result)"
        # mis_code = "import itertools\n\ndef compute_tsp_distance(dist_matrix):\n    n = len(dist_matrix)\n    dp = {}\n    for k in range(1, n):\n        dp[(1 << k, k)] = dist_matrix[0][k]\n    for subset_size in range(2, n):\n        for subset in itertools.combinations(range(1, n), subset_size):\n            mask = 0\n            for v in subset:\n                mask |= 1 << v\n            for j in subset:\n                prev_mask = mask & ~(1 << j)\n                best = None\n                for k in subset:\n                    if k == j:\n                        continue\n                    cost = dp[(prev_mask, k)] + dist_matrix[k][j]\n                    if best is None or cost < best:\n                        best = cost\n                dp[(mask, j)] = best\n    full_mask = (1 << n) - 2\n    best = None\n    for j in range(1, n):\n        cost = dp[(full_mask, j)] + dist_matrix[j][0]\n        if best is None or cost < best:\n            best = cost\n    return best\n\n# Input data\ndist_matrix = [\n    [0, 10, 15, 20],\n    [10, 0, 35, 25],\n    [15, 35, 0, 30],\n    [20, 25, 30, 0],\n]\nresult = compute_tsp_distance(dist_matrix)\nprint(result)"
        kcolor_code = "def kcolor_with_explicit_stack(adj_list, k):\n    # manual stack-based DFS for coloring\n    vertices = list(adj_list.keys())\n    colors = {v: -1 for v in vertices}\n    stack = [(vertices[0], 0)]\n\n    while len(stack) > 0:\n        v, start_color = stack.pop()\n        if colors[v] != -1:\n            continue\n        c = start_color\n        chosen = None\n        while c < k:\n            ok = True\n            ns = adj_list[v]\n            i = 0\n            while i < len(ns):\n                u = ns[i]\n                if colors[u] == c:\n                    ok = False\n                    break\n                i += 1\n            if ok:\n                chosen = c\n                break\n            c += 1\n        if chosen is None:\n            raise ValueError('Stack-based coloring failed')\n        colors[v] = chosen\n        ns2 = adj_list[v]\n        j = 0\n        while j < len(ns2):\n            u = ns2[j]\n            if colors[u] == -1:\n                stack.append((u, 0))\n            j += 1\n    return colors\n\nadj = {\n    0: [1,2],\n    1: [0,3],\n    2: [0,3],\n    3: [1,2,4],\n    4: [3]\n}\nprint(kcolor_with_explicit_stack(adj, 3))"
        tag, data = self.parser.parse(kcolor_code)

        print(tag, data.G)
        # mis = PROBLEMS[tag](data.G)
        # mis.report_latex()


if __name__ == '__main__':
    # On macOS, using 'spawn' avoids forking the parent (safer with native libs).
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    unittest.main(verbosity=2)