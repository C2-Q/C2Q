"""
algorithmic_structural_level_validation.py

Direct, domain-level diversity metrics for AE/R2:
- Graph problems (MIS/MaxCut/Clique/KColor/VC/TSP):
  * n_nodes, n_edges, density, #components, avg/min/max degree, avg clustering
  * uniqueness via (a) exact edge-list fingerprint, and (b) WL graph hash
  * bucket distributions of n_nodes (and density bins)
  * NEW: algorithm-family labeling + entropy/top-dominance per tag
  * NEW: algorithm-signal dump per instance (to sanity-check cases like VC)

- Arithmetic (ADD/SUB/MUL):
  * a_bitlen, b_bitlen, max_bitlen, carry/borrow flags, prod_bitlen (MUL)
  * uniqueness via exact (a,b) pairs
  * bucket distributions of max_bitlen
  * NEW: algorithm-family labeling + entropy/top-dominance per tag
  * NEW: algorithm-signal dump per instance

- Factor:
  * N_bitlen
  * uniqueness via N
  * bucket distributions of N_bitlen
  * NEW: algorithm-family labeling + entropy/top-dominance per tag
  * NEW: algorithm-signal dump per instance

IMPORTANT UPDATE (Revision fix):
- Use dataset-provided `labels` (integer) as the ground-truth tag, NOT parser.parse() tag.
- Parser.parse(code) is used ONLY to extract `data` needed for metrics.

Outputs:
  diversity_out/
    metrics_per_instance.csv
    summary_by_tag.csv
    uniqueness_by_tag.csv
    buckets_by_tag.csv
    failures_summary.csv
    raw_vs_kept_by_tag.csv
    algorithm_family_per_instance.csv
    algorithm_family_by_tag.csv
    algorithm_diversity_summary.csv
    algorithm_signals_per_instance.csv

Run:
  python -m pytest -q algorithmic_structural_level_validation.py::MyTestCase::test_diversity_direct_metrics
"""

import os
import csv
import ast
import math
import unittest
from collections import defaultdict, Counter
from statistics import mean, pstdev

from src.parser.parser import Parser


# ----------------- tag mapping (ground truth from dataset labels) -----------------

PROBLEM_TAGS = {
    "MaxCut": 0,   # Maximum Cut Problem
    "MIS": 1,      # Maximal Independent Set
    "TSP": 2,      # Traveling Salesman Problem
    "Clique": 3,   # Clique Problem
    "KColor": 4,   # K-Coloring
    "Factor": 5,   # Factorization
    "ADD": 6,      # Addition
    "MUL": 7,      # Multiplication
    "SUB": 8,      # Subtraction
    "VC": 9,       # Vertex Cover
    "Unknown": 10,
}

LABEL_TO_TAG = {v: k for k, v in PROBLEM_TAGS.items()}

GRAPH_TAGS = {"MaxCut", "MIS", "TSP", "Clique", "KColor", "VC"}
ARITHMETIC_TAGS = {"ADD", "MUL", "SUB"}
ALGEBRAIC_TAGS = {"Factor"}


def normalize_label_to_tag(label_value):
    """
    Convert the CSV `labels` field into a canonical string tag.
    `labels` may come as "7", 7, or even "7.0" depending on CSV tooling.
    """
    if label_value is None:
        return None
    s = str(label_value).strip()
    try:
        lid = int(float(s))
    except Exception:
        return None
    tag = LABEL_TO_TAG.get(lid)
    if tag is None or tag == "Unknown":
        return None
    return tag


def is_data_compatible_with_tag(tag: str, data) -> bool:
    """
    Hard gate to ensure extracted `data` matches the *labelled* problem domain.
    This prevents accidental cross-domain contamination when parser extraction fails.
    """
    if tag in GRAPH_TAGS:
        if data is None:
            return False
        if hasattr(data, "G"):
            return True
        return hasattr(data, "number_of_nodes") and hasattr(data, "number_of_edges")

    if tag in ARITHMETIC_TAGS:
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            try:
                int(data[0]); int(data[1])
                return True
            except Exception:
                return False
        return False

    if tag in ALGEBRAIC_TAGS:
        try:
            int(data)
            return True
        except Exception:
            return False

    return False


# ----------------- helpers -----------------

def safe_mean(xs):
    return mean(xs) if xs else 0.0


def safe_std(xs):
    return pstdev(xs) if len(xs) >= 2 else 0.0


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return float(ent)


def bucket_int(v: int, edges):
    """
    Bucket integer v into ranges.
    edges is a sorted list of upper bounds, e.g. [3,5,7,10,20]
    returns a label like "<= 3", "4-5", "6-7", ...
    """
    if not edges:
        return str(v)
    prev = None
    for ub in edges:
        if v <= ub:
            if prev is None:
                return f"<= {ub}"
            return f"{prev + 1}-{ub}"
        prev = ub
    return f"> {edges[-1]}"


def bucket_float(v: float, bins):
    """
    Bucket float v into bins of (low, high], e.g. [(0,0.2),(0.2,0.4),...]
    """
    for lo, hi in bins:
        if lo < v <= hi or (v == lo == 0.0):
            return f"({lo},{hi}]"
    return f"> {bins[-1][1]}" if bins else f"{v}"


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def summarize_numeric(records, keys):
    """
    For each key in keys, compute min/max/mean/std over records where key exists and is numeric.
    """
    summary = {}
    for k in keys:
        vals = []
        for r in records:
            if k in r and r[k] is not None:
                v = r[k]
                if isinstance(v, (int, float)):
                    vals.append(float(v))
        if not vals:
            summary[f"{k}_min"] = 0.0
            summary[f"{k}_max"] = 0.0
            summary[f"{k}_mean"] = 0.0
            summary[f"{k}_std"] = 0.0
        else:
            summary[f"{k}_min"] = min(vals)
            summary[f"{k}_max"] = max(vals)
            summary[f"{k}_mean"] = safe_mean(vals)
            summary[f"{k}_std"] = safe_std(vals)
    return summary


# ----------------- algorithm-family classification -----------------

class AlgoSignalVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.calls = set()

        self.has_recursion = False
        self.loop_depth = 0
        self.max_loop_depth = 0

        self.has_itertools = False
        self.has_lru_cache = False
        self.has_memo_dict = False
        self.has_dp_table = False

        self.has_heapq = False
        self.has_sort = False
        self.has_minmax = False

        self.has_backtracking_patterns = False
        self.has_bitmask = False

        self.defined_funcs = set()
        self.current_func = None

    def visit_Import(self, node):
        for alias in node.names:
            name = (alias.name or "").split(".")[0]
            if name:
                self.imports.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        mod0 = (node.module or "").split(".")[0]
        if mod0:
            self.imports.add(mod0)
        for alias in node.names:
            if alias.name:
                self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.defined_funcs.add(node.name)
        prev = self.current_func
        self.current_func = node.name
        self.generic_visit(node)
        self.current_func = prev

    def visit_For(self, node):
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node):
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Call(self, node):
        fn_name = None
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fn_name = node.func.attr

        if fn_name:
            self.calls.add(fn_name)

        # recursion: function calling itself
        if self.current_func and fn_name == self.current_func:
            self.has_recursion = True

        # signals
        if fn_name in {"product", "permutations", "combinations"}:
            self.has_itertools = True
        if fn_name in {"lru_cache", "cache"}:
            self.has_lru_cache = True
        if fn_name in {"heappush", "heappop"}:
            self.has_heapq = True
        if fn_name in {"sorted", "sort"}:
            self.has_sort = True
        if fn_name in {"min", "max"}:
            self.has_minmax = True

        # backtracking-like patterns
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"append", "pop", "remove"}:
            self.has_backtracking_patterns = True

        self.generic_visit(node)

    def visit_Subscript(self, node):
        # dp[i][j] pattern: nested Subscript
        if isinstance(node.value, ast.Subscript):
            self.has_dp_table = True
        self.generic_visit(node)

    def visit_Assign(self, node):
        # memo[...] = ...
        for t in node.targets:
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                if t.value.id in {"memo", "dp", "cache"}:
                    self.has_memo_dict = True
        self.generic_visit(node)

    def visit_BinOp(self, node):
        # bitmask-ish
        if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift, ast.BitXor)):
            self.has_bitmask = True
        self.generic_visit(node)


def extract_algo_signals(code_str: str) -> dict:
    """
    Parse code and return a dict of algorithmic signals.
    This is for auditing (e.g., why VC ended up in 1 family).
    """
    low = (code_str or "").lower()
    nx_keyword = ("networkx" in low) or ("nx." in low)

    try:
        tree = ast.parse(code_str)
    except Exception:
        return {
            "parsed": 0,
            "nx_keyword": int(nx_keyword),
            "imports": "",
            "max_loop_depth": 0,
            "has_nested_loops": 0,
            "has_recursion": 0,
            "has_itertools": 0,
            "has_lru_cache": 0,
            "has_memo_dict": 0,
            "has_dp_table": 0,
            "has_backtracking_patterns": 0,
            "has_bitmask": 0,
            "has_networkx": 0,
        }

    v = AlgoSignalVisitor()
    v.visit(tree)

    has_nested_loops = 1 if v.max_loop_depth >= 2 else 0
    has_networkx = 1 if (nx_keyword or ("networkx" in v.imports) or ("nx" in v.imports)) else 0

    return {
        "parsed": 1,
        "nx_keyword": int(nx_keyword),
        "imports": ";".join(sorted(v.imports))[:500],  # cap for csv sanity
        "max_loop_depth": int(v.max_loop_depth),
        "has_nested_loops": int(has_nested_loops),
        "has_recursion": int(v.has_recursion),
        "has_itertools": int(v.has_itertools or ("itertools" in v.imports)),
        "has_lru_cache": int(v.has_lru_cache),
        "has_memo_dict": int(v.has_memo_dict),
        "has_dp_table": int(v.has_dp_table),
        "has_backtracking_patterns": int(v.has_backtracking_patterns),
        "has_bitmask": int(v.has_bitmask),
        "has_networkx": int(has_networkx),
    }


def classify_algorithm_family(code_str: str, tag: str) -> str:
    """
    Returns a coarse algorithm family label.
    Priority order is important to avoid collapsing everything into "iterative/greedy-ish".

    Fixes missed cases (esp. VC):
      - bruteforce via bitmask subset enumeration (range(1<<n), 2**n, i&(1<<j))
      - distinguishes VC approximation patterns from generic "iterative/greedy-ish"
    """
    sig = extract_algo_signals(code_str)
    if sig.get("parsed", 0) == 0:
        return "unparsed/unknown"

    low = (code_str or "").lower()

    # derived convenience flags
    has_networkx = bool(sig.get("has_networkx", 0))
    has_dp = bool(sig.get("has_lru_cache", 0) or sig.get("has_memo_dict", 0) or sig.get("has_dp_table", 0))
    has_itertools = bool(sig.get("has_itertools", 0))
    has_rec = bool(sig.get("has_recursion", 0))
    has_bt = bool(sig.get("has_backtracking_patterns", 0))
    has_nested = bool(sig.get("has_nested_loops", 0))
    loop_depth = int(sig.get("max_loop_depth", 0))
    has_bitmask = bool(sig.get("has_bitmask", 0))

    # subset enumeration heuristic (we keep it keyword-based; no extra AST features required)
    has_subset_enum = False
    if ("range(1 << " in low or "range(2 ** " in low or "1 << " in low or "2 ** " in low):
        has_subset_enum = True
    if "& (1 << " in low or "&(1<<" in low:
        has_subset_enum = True

    # ---------------- arithmetic ----------------
    if tag in {"ADD", "SUB"}:
        return "bitwise/low-level" if has_bitmask else "direct-arithmetic"

    if tag == "MUL":
        if has_bitmask:
            return "bitwise/low-level"
        if loop_depth >= 1:
            return "iterative-arithmetic"
        return "direct-arithmetic"

    # ---------------- factor ----------------
    if tag == "Factor":
        if has_nested:
            return "trial-division/nested-loops"
        if loop_depth >= 1:
            return "trial-division/loop"
        return "direct-check"

    # ---------------- graph problems ----------------
    if tag in {"MIS", "MaxCut", "Clique", "KColor", "VC", "TSP"}:
        if has_networkx:
            return "library-based(networkx)"

        if has_dp:
            return "dynamic-programming/memoization"

        if has_subset_enum and (has_bitmask or loop_depth >= 1):
            return "bruteforce(bitmask-enumeration)"

        if has_rec and has_bt:
            return "recursive-backtracking"
        if has_rec:
            return "recursive-search"

        if has_itertools:
            return "enumeration(itertools)"

        # VC-specific: distinguish common approximation/greedy variants
        if tag == "VC":
            if "remaining_edges" in low and "while" in low and ".pop(" in low:
                return "greedy(edge-popping)"
            if ("u not in cover" in low or "u not in c" in low) and ("v not in cover" in low or "v not in c" in low):
                if ".add(u)" in low and ".add(v)" in low:
                    return "2-approx(edge-picking)"

        if has_nested:
            return "enumeration(nested-loops)"

        if loop_depth >= 1:
            return "iterative/greedy-ish"

        return "unclassified"

    return "unclassified"


# ----------------- graph extraction & metrics -----------------

def graph_edges_fingerprint(G):
    """
    Exact fingerprint stable for same node labels:
      (n, sorted list of undirected edges as (min(u,v), max(u,v)))
    """
    n = G.number_of_nodes()
    edges = []
    for u, v in G.edges():
        a, b = (u, v) if u <= v else (v, u)
        edges.append((int(a), int(b)))
    edges.sort()
    return (int(n), tuple(edges))


def graph_wl_hash(G):
    """
    Weisfeiler–Lehman hash (robust to node relabeling).
    """
    try:
        from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
        return weisfeiler_lehman_graph_hash(G)
    except Exception:
        fp = graph_edges_fingerprint(G)
        return f"FP:{fp[0]}:{len(fp[1])}"


def extract_nx_graph(obj):
    """
    Parser returns:
      - src.graph.Graph instance with attribute .G (networkx Graph), OR
      - already a networkx Graph.
    """
    if hasattr(obj, "G"):
        return obj.G
    return obj


def graph_domain_metrics(G):
    import networkx as nx

    n = int(G.number_of_nodes())
    m = int(G.number_of_edges())
    density = 0.0 if n <= 1 else (2.0 * m) / (n * (n - 1))

    components = int(nx.number_connected_components(G)) if n > 0 else 0
    degrees = [int(d) for _, d in G.degree()] if n > 0 else []
    avg_degree = float(sum(degrees) / len(degrees)) if degrees else 0.0
    min_degree = int(min(degrees)) if degrees else 0
    max_degree = int(max(degrees)) if degrees else 0

    avg_clustering = float(nx.average_clustering(G)) if n > 1 else 0.0

    return {
        "n_nodes": n,
        "n_edges": m,
        "density": float(density),
        "n_components": components,
        "deg_mean": float(avg_degree),
        "deg_min": min_degree,
        "deg_max": max_degree,
        "clustering_mean": float(avg_clustering),
    }


# ----------------- arithmetic & factor metrics -----------------

def arithmetic_metrics(tag, data):
    a, b = int(data[0]), int(data[1])
    abit = abs(a).bit_length()
    bbit = abs(b).bit_length()

    out = {
        "a": a,
        "b": b,
        "a_bitlen": abit,
        "b_bitlen": bbit,
        "max_bitlen": max(abit, bbit),
        "pair_fp": f"{a},{b}",
    }

    if tag == "ADD":
        out["has_carry_like"] = 1 if abs(a + b).bit_length() > max(abit, bbit) else 0
    else:
        out["has_carry_like"] = 0

    if tag == "SUB":
        out["has_borrow_like"] = 1 if a < b else 0
    else:
        out["has_borrow_like"] = 0

    if tag == "MUL":
        out["prod_bitlen"] = abs(a * b).bit_length()
    else:
        out["prod_bitlen"] = 0

    return out


def factor_metrics(data):
    N = int(data)
    return {
        "N": N,
        "N_bitlen": int(N.bit_length()),
        "N_fp": str(N),
    }


# ----------------- test case -----------------

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.parser = Parser(model_path="../parser/saved_models")

    def test_diversity_direct_metrics(self):
        input_csv = "../parser/python_programs.csv"
        out_dir = "./diversity_out"
        os.makedirs(out_dir, exist_ok=True)

        per_instance = []
        failures = Counter()

        # For uniqueness + buckets
        uniq_exact = defaultdict(set)   # tag -> set(fingerprint)
        uniq_wl = defaultdict(set)      # tag -> set(wl_hash)
        buckets = defaultdict(Counter)  # (tag, bucket_name) -> Counter(label)

        # Algorithm family + signals
        algo_per_instance = []                 # case_idx, tag, algo_family
        algo_by_tag = defaultdict(Counter)     # tag -> Counter(family)
        algo_signals_rows = []                 # case_idx, tag, signals...

        # NEW: prove consistency
        raw_count_by_tag = Counter()
        kept_count_by_tag = Counter()

        # bucket settings
        n_buckets = [3, 5, 7, 10, 20]
        bit_buckets = [4, 8, 16, 32, 64, 128]
        density_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

        with open(input_csv, "r", encoding="utf-8") as f:
            dr = csv.DictReader(f)

            for case_idx, row in enumerate(dr, start=1):
                try:
                    raw_code = row.get("code_snippet", "")
                    label_val = row.get("labels", None)

                    tag = normalize_label_to_tag(label_val)
                    if tag is None:
                        failures["bad_or_unknown_label"] += 1
                        continue
                    raw_count_by_tag[tag] += 1

                    # decode stored Python string
                    try:
                        clean_code = ast.literal_eval(raw_code)
                    except Exception:
                        clean_code = raw_code.encode().decode("unicode_escape")

                    # parser used ONLY to extract data; keep parser_tag for auditing
                    # 1) syntax gate (align with Table 14)
                    try:
                        ast.parse(clean_code)
                    except Exception:
                        failures["syntax_error"] += 1
                        continue

                    # 2) now we "keep" this snippet (counts consistent)
                    kept_count_by_tag[tag] += 1

                    # 3) parse ONLY for extracting data (no filtering)
                    parser_tag, data = self.parser.parse(clean_code)
                    if parser_tag is None or parser_tag == "Unknown":
                        failures["parser_unknown_tag"] += 1

                    if parser_tag != tag:
                        failures["label_tag_mismatch"] += 1

                    # 4) if data incompatible, DON'T drop snippet; just disable domain metrics
                    if not is_data_compatible_with_tag(tag, data):
                        failures["data_incompatible_with_label"] += 1
                        data = None  # important

                    record = {"case_idx": case_idx, "tag": tag, "parser_tag": parser_tag}

                    # NEW: mark whether domain metrics are actually computable
                    record["has_domain_data"] = 1 if data is not None else 0

                    # Algorithm family + signal dump
                    sig = extract_algo_signals(clean_code)
                    fam = classify_algorithm_family(clean_code, tag)
                    record["algo_family"] = fam

                    algo_per_instance.append({"case_idx": case_idx, "tag": tag, "algo_family": fam})
                    algo_by_tag[tag][fam] += 1

                    sig_row = {"case_idx": case_idx, "tag": tag, "algo_family": fam, "parser_tag": parser_tag}
                    sig_row.update(sig)
                    algo_signals_rows.append(sig_row)

                    # --- graph-like problems ---
                    # (A) algorithm-family always works (uses code only)
                    sig = extract_algo_signals(clean_code)
                    fam = classify_algorithm_family(clean_code, tag)
                    record["algo_family"] = fam
                    # ... write algo_per_instance, algo_by_tag, signals, etc.

                    # (B) domain metrics only if data exists and is compatible
                    if data is not None:
                        if tag in GRAPH_TAGS:
                            G = extract_nx_graph(data)
                            gm = graph_domain_metrics(G)
                            record.update(gm)

                            fp = graph_edges_fingerprint(G)
                            wl = graph_wl_hash(G)
                            uniq_exact[tag].add(fp)
                            uniq_wl[tag].add(wl)

                            buckets[(tag, "n_nodes")][bucket_int(gm["n_nodes"], n_buckets)] += 1
                            buckets[(tag, "density")][bucket_float(gm["density"], density_bins)] += 1
                            buckets[(tag, "n_components")][str(gm["n_components"])] += 1

                        elif tag in ARITHMETIC_TAGS:
                            am = arithmetic_metrics(tag, data)
                            record.update(am)
                            uniq_exact[tag].add(am["pair_fp"])
                            buckets[(tag, "max_bitlen")][bucket_int(int(am["max_bitlen"]), bit_buckets)] += 1

                        elif tag in ALGEBRAIC_TAGS:
                            fm = factor_metrics(data)
                            record.update(fm)
                            uniq_exact[tag].add(fm["N_fp"])
                            buckets[(tag, "N_bitlen")][bucket_int(int(fm["N_bitlen"]), bit_buckets)] += 1

                    # (C) Always keep the record (so counts match table 14)
                    per_instance.append(record)

                except SyntaxError:
                    failures["syntax_error"] += 1
                except Exception:
                    failures["other_exception"] += 1

        # ---- outputs ----

        write_csv(os.path.join(out_dir, "metrics_per_instance.csv"), per_instance)

        failures_rows = [{"failure_type": k, "count": v} for k, v in sorted(failures.items())]
        write_csv(os.path.join(out_dir, "failures_summary.csv"), failures_rows, fieldnames=["failure_type", "count"])

        raw_vs_kept = []
        for tag in sorted(raw_count_by_tag.keys()):
            raw = int(raw_count_by_tag[tag])
            kept = int(kept_count_by_tag.get(tag, 0))
            raw_vs_kept.append({
                "tag": tag,
                "raw_count": raw,
                "kept_count": kept,
                "kept_ratio": (kept / raw) if raw else 0.0,
            })
        write_csv(os.path.join(out_dir, "raw_vs_kept_by_tag.csv"),
                  raw_vs_kept,
                  fieldnames=["tag", "raw_count", "kept_count", "kept_ratio"])

        # summary by tag (numeric aggregates)
        by_tag = defaultdict(list)
        for r in per_instance:
            by_tag[r["tag"]].append(r)

        numeric_keys = [
            "n_nodes", "n_edges", "density", "n_components",
            "deg_mean", "deg_min", "deg_max", "clustering_mean",
            "a_bitlen", "b_bitlen", "max_bitlen", "prod_bitlen",
            "has_carry_like", "has_borrow_like",
            "N_bitlen",
        ]

        summary_rows = []
        for tag, recs in sorted(by_tag.items()):
            row = {"tag": tag, "count": len(recs)}
            row.update(summarize_numeric(recs, numeric_keys))
            summary_rows.append(row)
        write_csv(os.path.join(out_dir, "summary_by_tag.csv"), summary_rows)

        # uniqueness by tag
        uniq_rows = []
        for tag, recs in sorted(by_tag.items()):
            usable = [r for r in recs if r.get("has_domain_data", 0) == 1]
            total = len(usable)
            u_exact = len(uniq_exact.get(tag, set()))
            u_wl = len(uniq_wl.get(tag, set())) if tag in GRAPH_TAGS else 0
            uniq_rows.append({
                "tag": tag,
                "count_with_data": total,  # <-- rename to be honest
                "unique_exact": u_exact,
                "unique_exact_ratio": (u_exact / total) if total else 0.0,
                "unique_wl": u_wl,
                "unique_wl_ratio": (u_wl / total) if total else 0.0,
            })
        write_csv(os.path.join(out_dir, "uniqueness_by_tag.csv"), uniq_rows)

        # buckets by tag
        bucket_rows = []
        for (tag, metric), counter in sorted(buckets.items()):
            total = sum(counter.values())
            for bucket_label, c in counter.most_common():
                bucket_rows.append({
                    "tag": tag,
                    "metric": metric,
                    "bucket": bucket_label,
                    "count": c,
                    "ratio": (c / total) if total else 0.0,
                })
        write_csv(os.path.join(out_dir, "buckets_by_tag.csv"), bucket_rows,
                  fieldnames=["tag", "metric", "bucket", "count", "ratio"])

        # algorithm-family outputs
        write_csv(os.path.join(out_dir, "algorithm_family_per_instance.csv"),
                  algo_per_instance,
                  fieldnames=["case_idx", "tag", "algo_family"])

        dist_rows = []
        for tag, ctr in sorted(algo_by_tag.items()):
            total = sum(ctr.values())
            for fam, c in ctr.most_common():
                dist_rows.append({
                    "tag": tag,
                    "algo_family": fam,
                    "count": c,
                    "ratio": (c / total) if total else 0.0,
                })
        write_csv(os.path.join(out_dir, "algorithm_family_by_tag.csv"),
                  dist_rows,
                  fieldnames=["tag", "algo_family", "count", "ratio"])

        summary_algo_rows = []
        for tag, ctr in sorted(algo_by_tag.items()):
            total = sum(ctr.values())
            if total == 0:
                continue
            top_fam, top_cnt = ctr.most_common(1)[0]
            summary_algo_rows.append({
                "tag": tag,
                "count": total,
                "n_families": len(ctr),
                "entropy": round(shannon_entropy(ctr), 4),
                "top_family": top_fam,
                "top_family_ratio": round(top_cnt / total, 4),
            })
        write_csv(os.path.join(out_dir, "algorithm_diversity_summary.csv"),
                  summary_algo_rows,
                  fieldnames=["tag", "count", "n_families", "entropy", "top_family", "top_family_ratio"])

        # signal dump for auditing
        write_csv(os.path.join(out_dir, "algorithm_signals_per_instance.csv"),
                  algo_signals_rows)

        # sanity
        self.assertGreater(len(per_instance), 0, "No usable instances extracted for diversity analysis")


if __name__ == "__main__":
    unittest.main()