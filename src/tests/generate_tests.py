import json
import random
import shutil
import sys
from pathlib import Path


# ---------- 1. Random-payload generator ----------------------------
def random_payload(problem_type,
                       max_nodes: int = 10,
                       edge_prob: float = 0.4) -> dict:
    """Return a JSON-serialisable MIS instance with ≤ max_nodes nodes."""
    n = random.randint(2, max_nodes)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):          # upper-triangle only
            if random.random() < edge_prob:
                matrix[i][j] = matrix[j][i] = 1
    return {
        "problem_type": problem_type,
        "data": {"matrix": matrix}
    }


# ---------- 2. Clean directory & save N files ----------------------
def generate_cases(problem_type, out_dir: Path, count: int = 5) -> None:
    """Delete everything in out_dir, then write <count> new files."""
    # Remove old contents (but keep the folder itself)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # problem_type = "mis"
    for idx in range(1, count + 1):
        payload = random_payload(problem_type)
        fname = out_dir / f"{problem_type}_{idx}.json"
        with open(fname, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"✅ Saved {fname}")


# ---------- 3. CLI or import-friendly entry point ------------------
if __name__ == "__main__":
    problem_type = "maxcut"
    generate_cases(problem_type, Path(f"generated_cases/{problem_type}"), count=10)   # adjust count as needed
