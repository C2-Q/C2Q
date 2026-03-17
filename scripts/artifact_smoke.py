import argparse
import json
from pathlib import Path

from src.algorithms.QAOA.QAOA import qaoa_no_optimization
from src.parser.parser import Parser
from src.problems.max_cut import MaxCut


SMOKE_SNIPPET = """def max_cut_brute_force(n, edges):
    best_cut_value = 0
    best_partition = None
    for i in range(1 << n):
        set_a = {j for j in range(n) if i & (1 << j)}
        set_b = set(range(n)) - set_a
        cut_value = sum(
            1 for u, v in edges if (u in set_a and v in set_b) or (u in set_b and v in set_a)
        )
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = (set_a, set_b)
    return best_cut_value, best_partition

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
cut_value, partition = max_cut_brute_force(4, edges)
print(cut_value, partition)
"""


def build_summary(model_path: str) -> dict:
    parser = Parser(model_path=model_path)
    problem_type, data = parser.parse(SMOKE_SNIPPET)
    if problem_type != "MaxCut":
        raise RuntimeError(f"Smoke parser classification mismatch: expected MaxCut, got {problem_type}")

    problem = MaxCut(data.G)
    qubo = problem.to_qubo().Q
    qaoa = qaoa_no_optimization(qubo, layers=1)
    circuit = qaoa["qc"]

    return {
        "problem_type": problem_type,
        "graph_nodes": int(data.G.number_of_nodes()),
        "graph_edges": int(data.G.number_of_edges()),
        "qubo_shape": list(qubo.shape),
        "qaoa_qubits": int(circuit.num_qubits),
        "qaoa_depth": int(circuit.depth()),
        "qaoa_ops": {str(name): int(count) for name, count in circuit.count_ops().items()},
    }


def main() -> int:
    cli = argparse.ArgumentParser(description="Run a minimal reviewer-friendly C2Q smoke test.")
    cli.add_argument("--model-path", required=True, help="Path to the parser model directory.")
    cli.add_argument(
        "--output-dir",
        default="artifacts/smoke",
        help="Directory where the smoke summary JSON will be written.",
    )
    args = cli.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(args.model_path)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[smoke] wrote {output_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
