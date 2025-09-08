import json
import math
import random
import shutil
from pathlib import Path

# ---------------------------
# Helpers
# ---------------------------

def ensure_dir_clean(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def rand_graph(n_min, n_max, p=0.4):
    n = random.randint(n_min, n_max)
    M = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                M[i][j] = M[j][i] = 1
    return M

def rand_points(n, lo=0, hi=100):
    return [(random.uniform(lo, hi), random.uniform(lo, hi)) for _ in range(n)]

def euclid_dist_matrix(pts):
    n = len(pts)
    D = [[0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i+1, n):
            xj, yj = pts[j]
            d = int(round(math.hypot(xi - xj, yi - yj)))
            D[i][j] = D[j][i] = max(d, 1)
    return D

# ---------------------------
# Generators (match table sizes)
# ---------------------------

def gen_maxcut():
    # 4–8 nodes
    M = rand_graph(4, 8)
    return {"problem_type": "maxcut", "json": {"matrix": M}}

def gen_mis():
    # 4–8 nodes
    M = rand_graph(4, 8)
    return {"problem_type": "mis", "json": {"matrix": M}}

def gen_tsp():
    # 4–6 nodes
    n = random.randint(4, 6)
    pts = rand_points(n)
    D = euclid_dist_matrix(pts)
    return {"problem_type": "tsp", "json": {"distance_matrix": D}}

def gen_clique():
    # 4–6 nodes, small k
    M = rand_graph(4, 6)
    n = len(M)
    k = random.randint(2, min(4, n))
    return {"problem_type": "clique", "json": {"matrix": M, "k": k}}

def gen_kcolor():
    # 4–6 nodes, small k
    M = rand_graph(4, 6)
    n = len(M)
    k = random.randint(2, min(4, n))
    return {"problem_type": "kcolor", "json": {"matrix": M, "k": k}}

def gen_vc():
    # 4–6 nodes
    M = rand_graph(4, 6)
    return {"problem_type": "vc", "json": {"matrix": M}}

def gen_factor():
    # semiprimes 15 (=3×5) or 21 (=3×7)
    n = random.choice([15, 21])
    return {"problem_type": "factor", "json": {"n": n}}

def gen_add():
    # 2–3 bit operands
    bits = random.randint(2, 3)
    a = random.randint(0, (1 << bits) - 1)
    b = random.randint(0, (1 << bits) - 1)
    return {"problem_type": "add", "json": {"operands": [a, b], "bits": bits}}

def gen_mul():
    # 2×2 or 3×2 bit-widths
    a_bits, b_bits = random.choice([(2, 2), (3, 2)])
    a = random.randint(0, (1 << a_bits) - 1)
    b = random.randint(0, (1 << b_bits) - 1)
    return {
        "problem_type": "mul",
        "json": {"operands": [a, b], "a_bits": a_bits, "b_bits": b_bits}
    }

def gen_sub():
    # 2–3 bit operands
    bits = random.randint(2, 3)
    a = random.randint(0, (1 << bits) - 1)
    b = random.randint(0, (1 << bits) - 1)
    return {"problem_type": "sub", "json": {"operands": [a, b], "bits": bits}}

GENERATORS = {
    "maxcut": gen_maxcut,
    "mis": gen_mis,
    "tsp": gen_tsp,
    "clique": gen_clique,
    "kcolor": gen_kcolor,
    "vc": gen_vc,
    "factor": gen_factor,
    "add": gen_add,
    "mul": gen_mul,
    "sub": gen_sub,
}

# ---------------------------
# Batch runner
# ---------------------------

def generate_cases_for_all(out_root: Path, count_per_type: int = 10):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for ptype, fn in GENERATORS.items():
        out_dir = out_root / ptype
        ensure_dir_clean(out_dir)
        for idx in range(1, count_per_type + 1):
            payload = fn()
            with open(out_dir / f"{ptype}_{idx:02d}.json", "w") as fh:
                json.dump(payload, fh, indent=2)
        print(f"✅ Saved {count_per_type} cases to {out_dir}")

if __name__ == "__main__":
    # Creates 10 folders under json/, each with 10 JSON files.
    generate_cases_for_all(Path("json"), count_per_type=10)