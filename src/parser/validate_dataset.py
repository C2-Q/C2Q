#!/usr/bin/env python
"""
Validation and diversity analysis for C2|Q> subject programs.

Expected input: data2.csv with columns:
    - code_snippet : raw Python code (possibly escaped / quoted)
    - labels       : integer label for problem family

Outputs:
    - snippet_metrics.csv : per-snippet metrics
    - family_summary.csv  : aggregated metrics per family
"""

import ast
import textwrap
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Utility: clean raw code cell into usable Python code
# ---------------------------------------------------------

def clean_code_text(raw):
    """
    Clean a raw CSV cell into a usable Python code string.

    Handles:
    - CSV string quoting ("..."/'...')
    - cells stored as Python reprs (via ast.literal_eval)
    - escaped newlines like '\\n'  -> real newlines
    - leading 'code:' / 'code ' markers
    - leading '# Python code\\n' comment wrapper
    - one stray trailing quote at the end
    """
    s = str(raw).strip()

    decoded = None

    # 1) Try to interpret as a Python string literal first
    #    This fixes cells that literally store "\"import ...\\n...\""
    try:
        tmp = ast.literal_eval(s)
        if isinstance(tmp, str):
            decoded = tmp
    except Exception:
        decoded = None

    # 2) If literal_eval didn't help, try unicode_escape to turn '\n' into newlines
    if decoded is None:
        try:
            decoded = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            decoded = s

    s = decoded.strip()

    # 3) Strip ONE layer of symmetric surrounding quotes, if still present
    if len(s) >= 2 and (
        (s[0] == '"' and s[-1] == '"') or
        (s[0] == "'" and s[-1] == "'")
    ):
        s = s[1:-1].strip()

    # 4) If there is a single stray quote only at the end, drop it
    if s.endswith('"') or s.endswith("'"):
        if not (len(s) >= 2 and s[0] == s[-1]):  # not symmetric
            s = s[:-1].rstrip()

    # 5) Drop leading "code:" / "code " markers (case-insensitive)
    lowered = s.lstrip().lower()
    for prefix in ("code:", "code "):
        if lowered.startswith(prefix):
            start = s.lower().find(prefix)
            s = s[start + len(prefix):].lstrip()
            break

    # 6) Drop a header line like "# Python code"
    lines = s.splitlines()
    if lines and lines[0].lstrip().lower().startswith("# python code"):
        lines = lines[1:]
        s = "\n".join(lines)

    # 7) Dedent to normalise indentation
    s = textwrap.dedent(s)

    return s


# ---------------------------------------------------------
# 2. AST analysis helpers
# ---------------------------------------------------------

class CodeMetricsVisitor(ast.NodeVisitor):
    """
    Walk the AST and collect structural metrics:

    - control-flow usage (for/while/if, try)
    - boolean operations
    - number of functions
    - recursion (self-calls)
    - approximate cyclomatic complexity
    - data-structure usage (list/dict/set/tuple)
    """

    def __init__(self):
        super().__init__()
        # control flow
        self.has_for = False
        self.has_while = False
        self.if_count = 0
        self.boolop_count = 0
        self.try_count = 0

        # functions / recursion
        self.num_functions = 0
        self._func_stack = []          # current function name stack
        self.recursive_calls = set()   # names of functions that recurse

        # data structures
        self.uses_list = False
        self.uses_dict = False
        self.uses_set = False
        self.uses_tuple = False

    # ----- control flow -----
    def visit_For(self, node):
        self.has_for = True
        self.generic_visit(node)

    def visit_While(self, node):
        self.has_while = True
        self.generic_visit(node)

    def visit_If(self, node):
        self.if_count += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.boolop_count += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.try_count += 1
        self.generic_visit(node)

    # ----- functions / recursion -----
    def visit_FunctionDef(self, node):
        self.num_functions += 1
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_Call(self, node):
        # detect simple direct recursion: f() inside def f(...)
        if self._func_stack and isinstance(node.func, ast.Name):
            current = self._func_stack[-1]
            if node.func.id == current:
                self.recursive_calls.add(current)
        self.generic_visit(node)

    # ----- data structures -----
    def visit_List(self, node):
        self.uses_list = True
        self.generic_visit(node)

    def visit_Dict(self, node):
        self.uses_dict = True
        self.generic_visit(node)

    def visit_Set(self, node):
        self.uses_set = True
        self.generic_visit(node)

    def visit_Tuple(self, node):
        self.uses_tuple = True
        self.generic_visit(node)


def analyse_code_structure(code_str):
    """
    Parse code into AST, run CodeMetricsVisitor, and return metrics dict.

    If parsing fails, 'syntax_ok' is False and remaining fields are default.
    """
    metrics = {
        "syntax_ok": False,
        "lines_code": 0,
        "lines_comment": 0,
        "has_for": False,
        "has_while": False,
        "if_count": 0,
        "boolop_count": 0,
        "try_count": 0,
        "cyclomatic_approx": None,
        "num_functions": 0,
        "has_recursion": False,
        "uses_list": False,
        "uses_dict": False,
        "uses_set": False,
        "uses_tuple": False,
        "uses_networkx": False,
    }

    if not code_str.strip():
        return metrics

    # Line-level stats
    lines = code_str.splitlines()
    metrics["lines_code"] = sum(1 for ln in lines if ln.strip())
    metrics["lines_comment"] = sum(
        1 for ln in lines if ln.strip().startswith("#")
    )

    # Quick text-based NetworkX usage heuristic
    lower = code_str.lower()
    if "networkx" in lower or " import nx" in lower or " as nx" in lower:
        metrics["uses_networkx"] = True

    # AST-based stats
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        # leave syntax_ok=False, keep line counts etc.
        return metrics

    visitor = CodeMetricsVisitor()
    visitor.visit(tree)

    metrics["syntax_ok"] = True
    metrics["has_for"] = visitor.has_for
    metrics["has_while"] = visitor.has_while
    metrics["if_count"] = visitor.if_count
    metrics["boolop_count"] = visitor.boolop_count
    metrics["try_count"] = visitor.try_count
    metrics["num_functions"] = visitor.num_functions
    metrics["has_recursion"] = len(visitor.recursive_calls) > 0
    metrics["uses_list"] = visitor.uses_list
    metrics["uses_dict"] = visitor.uses_dict
    metrics["uses_set"] = visitor.uses_set
    metrics["uses_tuple"] = visitor.uses_tuple

    # Very rough cyclomatic complexity:
    # base 1 + number of decision points
    cyclomatic = (
        1
        + visitor.if_count
        + visitor.boolop_count
        + (1 if visitor.has_for else 0)
        + (1 if visitor.has_while else 0)
        + visitor.try_count
    )
    metrics["cyclomatic_approx"] = cyclomatic

    return metrics


# ---------------------------------------------------------
# 3. Main analysis pipeline
# ---------------------------------------------------------

def main(
    csv_path: str = "python_programs.csv",
    out_snippet_metrics: str = "snippet_metrics.csv",
    out_family_summary: str = "family_summary.csv",
):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Expect columns: 'code_snippet', 'labels'
    if "code_snippet" not in df.columns or "labels" not in df.columns:
        raise ValueError("Expected columns 'code_snippet' and 'labels' in the CSV.")

    # Clean code
    df["code_clean"] = df["code_snippet"].apply(clean_code_text)

    # Map labels -> family names.
    try:
        from src.parser.parser import PROBLEMS  # adjust path if needed
        label_to_family = {i: name for i, name in enumerate(PROBLEMS)}
    except Exception:
        unique_labels = sorted(df["labels"].unique())
        label_to_family = {lab: f"Family-{lab}" for lab in unique_labels}

    df["family"] = df["labels"].map(label_to_family)

    # Per-snippet metrics
    all_metrics = []
    for idx, row in df.iterrows():
        code = row["code_clean"]
        metrics = analyse_code_structure(code)
        metrics["index"] = idx
        metrics["family"] = row["family"]
        metrics["label"] = row["labels"]
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)

    # Merge for convenience
    merged = pd.concat(
        [
            df[["code_snippet", "code_clean", "labels", "family"]],
            metrics_df.drop(columns=["family", "label", "index"], errors="ignore"),
        ],
        axis=1,
    )

    # Save per-snippet metrics
    merged.to_csv(out_snippet_metrics, index=False)

    # --------- NEW: restrict diversity stats to syntactically valid snippets ---------
    valid = merged[merged["syntax_ok"]]

    # Family-level aggregate summary
    group_cols = ["family"]
    agg = valid.groupby(group_cols).agg(
        num_snippets=("family", "size"),
        syntax_ok_ratio=("syntax_ok", "mean"),  # will be 1.0 on 'valid' but kept for clarity
        lines_min=("lines_code", "min"),
        lines_max=("lines_code", "max"),
        lines_mean=("lines_code", "mean"),
        comments_mean=("lines_comment", "mean"),
        has_for_count=("has_for", "sum"),
        has_while_count=("has_while", "sum"),
        has_recursion_count=("has_recursion", "sum"),
        uses_networkx_count=("uses_networkx", "sum"),
        uses_list_count=("uses_list", "sum"),
        uses_dict_count=("uses_dict", "sum"),
        uses_set_count=("uses_set", "sum"),
        uses_tuple_count=("uses_tuple", "sum"),
        cyclomatic_mean=("cyclomatic_approx", "mean"),
    ).reset_index()

    # Save family-level summary
    agg.to_csv(out_family_summary, index=False)

    # Quick console print
    print("=== Family counts (all) ===")
    print(merged["family"].value_counts())
    print()
    print("=== Family-level summary (syntax_ok only) ===")
    print(agg)


if __name__ == "__main__":
    main()