# json_engine.py
#
# JSON-DSL entry point for the C2|Q⟩ pipeline.
#
# Supported JSON shapes (backward compatible):
#
# (A) New, DSL-style input (recommended)
# {
#   "family": "MaxCut",                 # or "MIS", "TSP", "Clique", "KColor",
#                                       # "Factor", "ADD", "MUL", "SUB", "VC"
#   "goal": "minimise routing cost",    # optional, domain-style description
#   "description": "optional free text",
#   "instance": {                       # problem-family-specific payload
#     "graph_rep": "edge_list",
#     "graphs": { "G1": [[0,1],[1,2],[2,3],[3,0]] }
#   },
#   "parameters": {                     # optional configuration for generator
#     "solver": "QAOA",
#     "p": 2
#   }
# }
#
# (B) Legacy input (still supported)
# {
#   "problem_type": "MaxCut",
#   "json": { "G1": [[0,1],[1,2],[2,3],[3,0]] },
#   "config": { ... }
# }
#
# (C) Arithmetic JSON examples (all accepted):
# {
#   "problem_type": "add",
#   "data": { "operands": [2, 2], "bits": 3 }
# }
#
# {
#   "problem_type": "ADD",
#   "data": { "a": 2, "b": 2, "bits": 3 }
# }
#
# Both will be normalised to a dict that includes 'operands' and is passed
# to Add/Mul/Sub/Factor.

import json
import argparse
import random
import shutil
import traceback
from typing import Any, Dict, Tuple, Optional, List

from src.graph import Graph

from src.problems.basic_arithmetic.addition import Add
from src.problems.basic_arithmetic.multiplication import Mul
from src.problems.basic_arithmetic.subtraction import Sub
from src.problems.clique import Clique
from src.problems.factorization import Factor
from src.problems.kcolor import KColor
from src.problems.max_cut import MaxCut
from src.problems.maximal_independent_set import MIS
from src.problems.minimum_vertex_cover import MVC
from src.problems.tsp import TSP
import os
from pathlib import Path

# -------------------------------------------------------------------
# 1. Family aliases / canonicalisation
# -------------------------------------------------------------------

PROBLEMS = {
    "MaxCut": MaxCut,
    "MIS": MIS,
    "TSP": TSP,
    "Clique": Clique,
    "KColor": KColor,
    "Factor": Factor,
    "ADD": Add,
    "MUL": Mul,
    "SUB": Sub,
    "VC": MVC,
}

DEFAULT_EXAMPLES_ROOT = "src/c2q-dataset/inputs/json_dsl"
DEFAULT_REPORTS_ROOT = "artifacts/json_dsl_reports"

# Aliases for lowercased family names → canonical keys in PROBLEMS
_FAMILY_ALIASES = {
    # Graph families
    "maxcut": "MaxCut",
    "max_cut": "MaxCut",
    "max-cut": "MaxCut",

    "mis": "MIS",
    "maximis": "MIS",
    "maximumindependentset": "MIS",
    "maximalindependentset": "MIS",
    "independentset": "MIS",

    "tsp": "TSP",
    "travellingsalesman": "TSP",
    "travelingsalesman": "TSP",
    "travelling_salesman": "TSP",
    "traveling_salesman": "TSP",

    "clique": "Clique",

    "kcolor": "KColor",
    "k_color": "KColor",
    "graphcoloring": "KColor",
    "graphcolouring": "KColor",
    "coloring": "KColor",
    "colouring": "KColor",

    "vc": "VC",
    "vertexcover": "VC",
    "minimumvertexcover": "VC",
    "minvertexcover": "VC",

    # Arithmetic / factor
    "add": "ADD",
    "addition": "ADD",

    "mul": "MUL",
    "multiply": "MUL",
    "multiplication": "MUL",

    "sub": "SUB",
    "subtract": "SUB",
    "subtraction": "SUB",

    "factor": "Factor",
    "factorisation": "Factor",
    "factorization": "Factor",
}

_FALLBACK_GRAPH_FAMILIES = {"MaxCut", "MIS", "TSP", "Clique", "KColor", "VC"}
_FALLBACK_ARITHMETIC_FAMILIES = {"ADD", "MUL", "SUB", "Factor"}
GRAPH_TAGS = tuple(sorted(_FALLBACK_GRAPH_FAMILIES))
ARITHMETIC_TAGS = tuple(sorted(_FALLBACK_ARITHMETIC_FAMILIES))


def _family_dir_name(family: str) -> str:
    return family.lower()


def _iter_json_paths(root: str) -> List[str]:
    return sorted(str(path) for path in Path(root).rglob("*.json"))


def save_generated_examples(
        out_dir: str = DEFAULT_EXAMPLES_ROOT,
        n_per_family: int = 10,
        clean: bool = False,
):
    """
    Generate JSON DSL examples for all families (n_per_family each)
    and save them into a directory.

    Directory structure:
        src/c2q-dataset/inputs/json_dsl/
            maxcut/maxcut_01.json
            mis/mis_01.json
            ...
            add/add_10.json
    """

    examples = generate_all_examples(n_per_family=n_per_family)

    out_path = Path(out_dir)
    if clean and out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"📁 Saving {n_per_family} examples per family into {out_path.resolve()}")

    count = 0

    for fam, fam_examples in examples.items():
        fam_name = _family_dir_name(fam)
        fam_dir = out_path / fam_name
        fam_dir.mkdir(parents=True, exist_ok=True)
        for i, ex in enumerate(fam_examples):
            fname = f"{fam_name}_{i + 1:02d}.json"
            fpath = fam_dir / fname

            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(ex, f, indent=2)

            count += 1

    print(f"✅ Saved {count} JSON DSL examples.")


def _normalise_key_for_alias(name: str) -> str:
    """Lowercase and strip common punctuation/whitespace for alias lookup."""
    s = name.strip().lower()
    for ch in [" ", "-", "_"]:
        s = s.replace(ch, "")
    return s


def canonicalise_family(raw: Optional[str]) -> Optional[str]:
    """Map a raw family / problem_type string to a canonical PROBLEMS key."""
    if raw is None:
        return None

    raw_stripped = raw.strip()
    # If it already matches a canonical key (case-sensitive)
    if raw_stripped in PROBLEMS:
        return raw_stripped

    # Case-insensitive direct match
    for key in PROBLEMS.keys():
        if raw_stripped.lower() == key.lower():
            return key

    # Alias-based match
    alias_key = _normalise_key_for_alias(raw_stripped)
    if alias_key in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[alias_key]

    return None


# -------------------------------------------------------------------
# 2. Domain-level keywords → family (Level-2 DSL)
# -------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    # routing / logistics
    "routing": "TSP",
    "route": "TSP",
    "tour": "TSP",
    "traveling salesman": "TSP",
    "travelling salesman": "TSP",
    "minimise distance": "TSP",
    "minimize distance": "TSP",
    "minimise routing cost": "TSP",
    "minimize routing cost": "TSP",

    # clustering / cut problems
    "community detection": "MaxCut",
    "cluster separation": "MaxCut",
    "separate graph": "MaxCut",
    "maximise cut": "MaxCut",
    "maximize cut": "MaxCut",

    # colouring / conflict problems
    "graph coloring": "KColor",
    "graph colouring": "KColor",
    "reduce conflicts": "KColor",
    "minimise conflicts": "KColor",
    "minimize conflicts": "KColor",

    # independence / resource separation
    "independent set": "MIS",
    "maximum independent set": "MIS",
    "resource independence": "MIS",
}


# -------------------------------------------------------------------
# 3. JSON loading & family/type recognition
# -------------------------------------------------------------------

def load_input(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _fallback_recognize_problem_class(family: str) -> str:
    """Fallback classifier when GRAPH_TAGS / ARITHMETIC_TAGS are not provided."""
    if family in _FALLBACK_GRAPH_FAMILIES:
        return "GRAPH"
    if family in _FALLBACK_ARITHMETIC_FAMILIES:
        return "ARITHMETIC"
    return "UNKNOWN"


def recognize_problem_class(family: str) -> str:
    """
    Map canonical family (e.g., 'MaxCut', 'ADD') to a coarse class:
    GRAPH vs ARITHMETIC vs UNKNOWN.
    """
    try:
        if family in GRAPH_TAGS:
            return "GRAPH"
        if family in ARITHMETIC_TAGS:
            return "ARITHMETIC"
    except NameError:
        pass
    return _fallback_recognize_problem_class(family)


def infer_family_from_goal(goal_text: str) -> Optional[str]:
    """
    Lightweight Level-2 DSL: infer canonical family from domain-style 'goal'
    or 'description' text (e.g., 'minimise routing cost' → TSP).
    """
    text = (goal_text or "").lower()
    for phrase, fam in DOMAIN_KEYWORDS.items():
        if phrase in text:
            return fam
    return None


# -------------------------------------------------------------------
# 4. Instance normalisation helpers
# -------------------------------------------------------------------

def normalise_arithmetic_instance(family: str, instance: Any) -> Dict[str, Any]:
    """
    Make arithmetic JSON robust.

    Accepts:
      - {"operands": [a, b], "bits": N}
      - {"a": a, "b": b, "bits": N}
      - {"n": N, "bits": M}   (for Factor)
      - or a plain list/tuple [a, b]

    Returns a dict that always includes "operands" (ADD/MUL/SUB) or
    both "n" and "value" (Factor).
    """
    if not isinstance(instance, dict):
        if family in {"ADD", "MUL", "SUB"} and isinstance(instance, (list, tuple)) and len(instance) == 2:
            a, b = instance
            return {"operands": [a, b]}
        return {"value": instance}

    data = dict(instance)

    if family in {"ADD", "MUL", "SUB"}:
        if "operands" not in data:
            a = data.get("a")
            b = data.get("b")
            if a is not None and b is not None:
                data["operands"] = [a, b]
        if "operands" not in data:
            maybe_ops = data.get("values") or data.get("nums")
            if isinstance(maybe_ops, (list, tuple)) and len(maybe_ops) == 2:
                data["operands"] = list(maybe_ops)

    elif family == "Factor":
        if "n" in data and "value" not in data:
            data["value"] = data["n"]
        if "value" in data and "n" not in data:
            data["n"] = data["value"]

    return data


def extract_binary_operands(norm_instance: Dict[str, Any]) -> Tuple[int, int]:
    """
    From a normalised arithmetic instance, extract a pair of integer operands.

    Expects something like:
      {"operands": [a, b], ...}
    and returns (a, b), validating they are ints.
    """
    # Primary path: use "operands"
    ops = norm_instance.get("operands")

    if isinstance(ops, (list, tuple)) and len(ops) == 2:
        left, right = ops
    else:
        # Fallback: try "a", "b", or "left", "right"
        if "a" in norm_instance and "b" in norm_instance:
            left, right = norm_instance["a"], norm_instance["b"]
        elif "left" in norm_instance and "right" in norm_instance:
            left, right = norm_instance["left"], norm_instance["right"]
        else:
            raise ValueError("Could not find two operands (a,b / left,right / operands[0,1]).")

    if not (isinstance(left, int) and isinstance(right, int)):
        raise ValueError("Both 'left' and 'right' must be integers.")

    return left, right


def extract_factor_number(norm_instance: Dict[str, Any]) -> int:
    """
    From a normalised factorisation instance, extract the integer to factor.

    Accepts shapes like:
      {"n": 91, "bits": 10}
      {"value": 91, "bits": 10}
      {"number": 91}

    Returns:
      91  (as int)
    """
    # Preferred keys
    if "n" in norm_instance:
        n = norm_instance["n"]
    elif "value" in norm_instance:
        n = norm_instance["value"]
    elif "number" in norm_instance:
        n = norm_instance["number"]
    else:
        raise ValueError("Could not find factorisation target ('n', 'value', or 'number').")

    if not isinstance(n, int):
        raise ValueError("Factorisation target must be an integer.")

    return n


def extract_graphs_from_instance(instance: Any) -> Dict[str, Any]:
    """
    Normalise various graph JSON shapes into a dict of name → payload.
    Accepts:
      - {"graphs": { "G1": payload, ... }, "graph_rep": "..."}
      - {"graph": payload}
      - {"edges": payload}
      - legacy { "G1": payload, ... }
    """
    if not isinstance(instance, dict):
        return {"G": instance}

    if "graphs" in instance and isinstance(instance["graphs"], dict):
        return instance["graphs"]

    for key in ("graph", "edges", "adjacency"):
        if key in instance:
            return {"G": instance[key]}

    if all(isinstance(v, (list, dict)) for v in instance.values()):
        return instance

    return {"G": instance}


# -------------------------------------------------------------------
# 5. Task normalisation (family, instance, params, goal)
# -------------------------------------------------------------------

def normalise_task(task: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any], str]:
    """
    Normalise heterogeneous JSON inputs into a single internal form:

        family   : canonical problem family (e.g., 'MaxCut', 'ADD')
        instance : problem-family-specific data (graph, operands, etc.)
        params   : optional configuration (solver choices, depth, etc.)
        goal     : optional free-text objective / description
    """
    raw_family = task.get("family") or task.get("problem_type") or task.get("goal")
    family = canonicalise_family(raw_family) if raw_family is not None else None

    goal_parts = []
    if "goal" in task and task["goal"]:
        goal_parts.append(str(task["goal"]))
    if "description" in task and task["description"]:
        goal_parts.append(str(task["description"]))
    goal_text = " ".join(goal_parts).strip()

    if family is None and goal_text:
        inferred = infer_family_from_goal(goal_text)
        if inferred is not None:
            family = inferred

    if family is None:
        family = "UNKNOWN"

    instance = task.get("instance") or task.get("json") or task.get("data") or {}
    params = task.get("parameters") or task.get("config") or {}

    return family, instance, params, goal_text


def _make_edge_list(num_nodes: int, pattern: int) -> List[List[int]]:
    """
    Deterministic tiny graphs for examples:
    pattern 0: path
    pattern 1: cycle
    pattern 2: "star" style
    """
    edges: List[List[int]] = []
    if num_nodes <= 1:
        return edges

    if pattern == 0:
        # path 0-1-2-...-(n-1)
        for u in range(num_nodes - 1):
            edges.append([u, u + 1])
    elif pattern == 1:
        # cycle
        for u in range(num_nodes):
            edges.append([u, (u + 1) % num_nodes])
    else:
        # star from node 0
        for u in range(1, num_nodes):
            edges.append([0, u])
    return edges


def generate_example_for_family(family: str, idx: int) -> Dict[str, Any]:
    """
    Generate a single JSON-DSL example for a given canonical family.
    This is used for testing the DSL and the normalisation logic.
    """
    # Arithmetic families: ADD, MUL, SUB, Factor
    if family == "ADD":
        a = (2 * idx + 1) % 1024
        b = (3 * idx + 2) % 1024
        return {
            "family": "ADD",
            "goal": "add two 10-bit integers",
            "description": f"Auto-generated ADD example #{idx}",
            "instance": {
                "operands": [a, b],
                "bits": 10,
            }
        }

    if family == "MUL":
        a = (5 * idx + 3) % 256
        b = (7 * idx + 1) % 256
        return {
            "family": "MUL",
            "goal": "multiply two integers",
            "description": f"Auto-generated MUL example #{idx}",
            "instance": {
                "operands": [a, b],
                "bits": 10,
            }
        }

    if family == "SUB":
        a = (10 * idx + 9) % 1024
        b = (3 * idx) % 512
        if b > a:
            a, b = b, a
        return {
            "family": "SUB",
            "goal": "subtract two integers",
            "description": f"Auto-generated SUB example #{idx}",
            "instance": {
                "operands": [a, b],
                "bits": 10,
            }
        }

    if family == "Factor":
        semiprimes = [15, 21, 33, 35, 39, 51, 55, 57, 65, 77]
        n = semiprimes[idx % len(semiprimes)]
        return {
            "family": "Factor",
            "goal": "factor a 10-bit integer",
            "description": f"Auto-generated Factor example #{idx}",
            "instance": {
                "n": n,
                "bits": 10,
            }
        }

    # Graph families: MaxCut, MIS, TSP, Clique, KColor, VC
    if family in {"MaxCut", "MIS", "TSP", "Clique", "KColor", "VC"}:
        rng = random.Random(f"{family}:{idx}")
        if family == "TSP":
            num_nodes = 4 + (idx % 2)
            graph_payload = _make_weighted_distance_matrix(num_nodes, idx)
            graph_rep = "matrix"
        else:
            num_nodes = 4 + (idx % 3)
            pattern = idx % 3
            edges = _make_edge_list(num_nodes, pattern)
            if num_nodes >= 5 and pattern != 2:
                extra_edge = sorted((rng.randrange(0, num_nodes), rng.randrange(0, num_nodes)))
                if extra_edge[0] != extra_edge[1] and extra_edge not in edges:
                    edges.append(extra_edge)
            graph_payload = edges
            graph_rep = "edge_list"


        if family == "MaxCut":
            goal = "maximize cut value between two partitions of the graph"
        elif family == "MIS":
            goal = "find a maximum independent set of the graph"
        elif family == "TSP":
            goal = "minimise routing cost for a traveling salesman tour"
        elif family == "Clique":
            goal = "find a maximum clique in the graph"
        elif family == "KColor":
            goal = "color the graph with as few conflicts as possible"
        else:  # VC
            goal = "find a minimum vertex cover of the graph"

        return {
            "family": family,
            "goal": goal,
            "description": f"Auto-generated {family} example #{idx}",
            "instance": {
                "graph_rep": graph_rep,
                "graphs": {
                    "G1": graph_payload
                }
            }
        }

    # Fallback (should not happen for current PROBLEMS)
    return {
        "family": family,
        "goal": "auto-generated test instance",
        "description": f"Fallback example for {family}, idx={idx}",
        "instance": {}
    }


def generate_all_examples(n_per_family: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate n_per_family JSON-DSL examples for each canonical family.
    """
    examples: Dict[str, List[Dict[str, Any]]] = {}
    for fam in PROBLEMS.keys():
        fam_examples = []
        for i in range(n_per_family):
            fam_examples.append(generate_example_for_family(fam, i))
        examples[fam] = fam_examples
    return examples


def _make_weighted_distance_matrix(num_nodes: int, idx: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = 1 + ((idx + 3) * (i + j + 1) + i * j) % 9
            matrix[i][j] = matrix[j][i] = weight
    return matrix


def self_test_examples(root: str = DEFAULT_EXAMPLES_ROOT) -> None:
    """
    Run a stronger JSON-DSL self-test using *real* Graph / arithmetic classes.

    For each *.json under `root`:
      - load JSON
      - normalise (family, instance, params, goal)
      - recognise problem class
      - if GRAPH:    build Graph(payload) and inspect node/edge counts
      - if ARITHMETIC: normalise operands and instantiate the problem class

    IMPORTANT: we do *not* call report_latex() here to avoid heavy
    quantum back-end work; this is a front-half DSL + data-shape sanity check.
    """
    paths = _iter_json_paths(root)
    if not paths:
        print(f"⚠️  No JSON examples found under {root}")
        return

    total = len(paths)
    passed = 0
    print(f"🔁 Self-testing JSON DSL examples in {root} ({total} files)...")

    for path in paths:
        try:
            task = load_input(path)
            family, instance, params, goal_text = normalise_task(task)

            if family == "UNKNOWN":
                raise ValueError("family is UNKNOWN after normalisation")

            if family not in PROBLEMS:
                raise ValueError(f"Unsupported family after normalisation: {family!r}")

            problem_class = recognize_problem_class(family)

            if problem_class == "GRAPH":
                graphs = extract_graphs_from_instance(instance)
                if not graphs:
                    raise ValueError("No graphs extracted from instance")

                # Build real Graph objects to exercise matrix/edge-list handling
                for name, payload in graphs.items():
                    g_obj = Graph(payload)
                    _ = g_obj.G.number_of_nodes()
                    _ = g_obj.G.number_of_edges()


            elif problem_class == "ARITHMETIC":

                norm_inst = normalise_arithmetic_instance(family, instance)

                if family in {"ADD", "MUL", "SUB"}:

                    left, right = extract_binary_operands(norm_inst)

                    _ = PROBLEMS[family]([left, right])


                elif family == "Factor":

                    n = extract_factor_number(norm_inst)

                    _ = PROBLEMS[family](n)

                else:

                    raise ValueError(f"Unknown ARITHMETIC family in self-test: {family!r}")

            else:
                raise ValueError(f"Family {family!r} classified as UNKNOWN")

            print(f"✅ {path}  (family={family}, class={problem_class})")
            passed += 1

        except Exception as e:
            print(f"❌ {path}  FAILED: {e}")
            # Uncomment if want full traces while debugging:
            # traceback.print_exc()

    print(f"\nSelf-test finished: {passed}/{total} examples passed front-half checks.")

import tempfile
from pathlib import Path

def batch_generate_reports(
        root: str = DEFAULT_EXAMPLES_ROOT,
        output_dir: str = DEFAULT_REPORTS_ROOT,
) -> None:
    paths = _iter_json_paths(root)
    if not paths:
        print(f"⚠️  No JSON examples found under {root}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 All LaTeX reports will be saved to: {output_path.resolve()}\n")

    total = len(paths)
    passed = 0
    failed = 0

    for path in paths:
        try:
            fname = Path(path).stem
            output_stem = output_path / fname
            output_pdf = output_stem.with_suffix(".pdf")

            print(f"➡️  Generating report for {path} → {output_pdf.name}")

            task = load_input(path)
            family, instance, params, goal_text = normalise_task(task)

            if family == "UNKNOWN":
                raise ValueError("Family is UNKNOWN")

            problem_class = recognize_problem_class(family)

            if problem_class == "GRAPH":
                graphs = extract_graphs_from_instance(instance)
                last_graph = None
                for _, payload in graphs.items():
                    last_graph = Graph(payload)
                problem = PROBLEMS[family](last_graph.G)

            elif problem_class == "ARITHMETIC":
                norm = normalise_arithmetic_instance(family, instance)
                if family in {"ADD", "MUL", "SUB"}:
                    left, right = extract_binary_operands(norm)
                    problem = PROBLEMS[family]([left, right])
                elif family == "Factor":
                    n = extract_factor_number(norm)
                    problem = PROBLEMS[family](n)
                else:
                    raise ValueError(f"Unknown arithmetic family {family}")

            else:
                raise ValueError(f"Unsupported problem class {problem_class}")

            problem.report_latex(output_path=str(output_stem))
            print(f"✅ SUCCESS: {output_pdf.name}\n")
            passed += 1

        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {path}")
            print(f"   Reason: {e}\n")

    print("📊 Batch LaTeX report summary:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📄 Total : {total}")
    print(f"\n📁 All outputs saved in: {output_path.resolve()}")

# -------------------------------------------------------------------
# 7. Main entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run C2|Q⟩ pipeline from JSON (lightweight DSL) input."
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to JSON file describing the problem instance.",
    )
    parser.add_argument(
        "--self_test_examples",
        action="store_true",
        help="Run JSON-DSL self-test over the examples directory (no report_latex).",
    )
    parser.add_argument(
        "--generate_examples",
        action="store_true",
        help="Generate JSON DSL examples for all families into the examples directory.",
    )
    parser.add_argument(
        "--batch_report",
        action="store_true",
        help="Generate LaTeX reports for all JSON files under the examples directory."
    )
    parser.add_argument(
        "--examples-root",
        type=str,
        default=DEFAULT_EXAMPLES_ROOT,
        help=f"Directory containing JSON examples (default: {DEFAULT_EXAMPLES_ROOT}).",
    )
    parser.add_argument(
        "--reports-output-dir",
        type=str,
        default=DEFAULT_REPORTS_ROOT,
        help=f"Output directory for batch-generated reports (default: {DEFAULT_REPORTS_ROOT}).",
    )
    parser.add_argument(
        "--n-per-family",
        type=int,
        default=10,
        help="Number of generated examples per family (default: 10).",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the examples directory before regenerating examples.",
    )

    args = parser.parse_args()

    # -----------------------
    # Handle: generate examples only
    # -----------------------
    if args.generate_examples:
        save_generated_examples(
            out_dir=args.examples_root,
            n_per_family=args.n_per_family,
            clean=args.clean_output,
        )
        return

    # -----------------------
    # Handle: self-test mode
    # -----------------------
    if args.self_test_examples:
        self_test_examples(root=args.examples_root)
        return
    # -----------------------
    # Handle: report all mode
    # -----------------------
    if args.batch_report:
        batch_generate_reports(
            root=args.examples_root,
            output_dir=args.reports_output_dir,
        )
        return
    # ----------------- Normal pipeline mode -----------------
    if not args.input:
        parser.error("Either --input, --self_test_examples, or --generate_examples must be provided.")

    print("📥 Loading problem from JSON DSL...")
    raw_task = load_input(args.input)
    family, instance, params, goal_text = normalise_task(raw_task)

    if family == "UNKNOWN":
        raise ValueError(
            "Could not infer problem family from JSON. "
            "Please provide 'family'/'problem_type' or a clearer 'goal' description."
        )

    if family not in PROBLEMS:
        raise ValueError(f"Unsupported problem family: {family!r}")

    print(f"🔍 Problem family: {family}")
    if goal_text:
        print(f"🎯 Goal: {goal_text}")
    print(f"⚙️ Parameters: {params}")

    problem_class = recognize_problem_class(family)
    print(f"📂 Recognised problem class: {problem_class}")

    global data  # kept for compatibility with existing modules

    # -----------------------------
    # GRAPH problems
    # -----------------------------
    if problem_class == "GRAPH":
        graphs = extract_graphs_from_instance(instance)
        last_graph_obj = None

        for name, payload in graphs.items():
            data = Graph(payload)
            last_graph_obj = data
            print(
                f"📊 Loaded graph '{name}' with "
                f"{data.G.number_of_nodes()} nodes and "
                f"{data.G.number_of_edges()} edges."
            )

        if last_graph_obj is None:
            raise ValueError("No graph payload could be extracted from JSON instance.")

        problem = PROBLEMS[family](last_graph_obj.G)

    # -----------------------------
    # ARITHMETIC / FACTORISATION
    # -----------------------------
    elif problem_class == "ARITHMETIC":
        norm_instance = normalise_arithmetic_instance(family, instance)

        if family in {"ADD", "MUL", "SUB"}:
            # Binary arithmetic → [left, right]
            left, right = extract_binary_operands(norm_instance)
            data = [left, right]
            problem = PROBLEMS[family](data)

        elif family == "Factor":
            # Factorisation → single integer
            n = extract_factor_number(norm_instance)
            data = n
            problem = PROBLEMS[family](data)

        else:
            raise ValueError(f"Unknown ARITHMETIC family: {family!r}")

    else:
        raise ValueError(
            f"Problem family {family!r} is neither GRAPH nor ARITHMETIC; "
            "please check the JSON input or extend recognise_problem_class."
        )

    print("🧾 Generating LaTeX report for quantum solutions...")
    problem.report_latex()
    print("✅ Done.")


if __name__ == "__main__":
    main()
