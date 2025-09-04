# MIS via Grover: local Aer + IBM hardware (Qiskit Runtime)
# ---------------------------------------------------------
# pip install qiskit qiskit-aer qiskit-algorithms qiskit-ibm-runtime networkx


# 125 lines of actual code exclude annotations
from __future__ import annotations
from typing import Iterable, Tuple, List, Optional
import math
import networkx as nx

from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.primitives import Sampler as LocalSampler
from qiskit_aer.primitives import Sampler as AerSampler

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as RuntimeSampler, Session, Options
from qiskit_algorithms import Grover, AmplificationProblem


# ---------- Graph helpers & predicates ----------

def to_graph(edges: Iterable[Tuple[int, int]], num_nodes: Optional[int] = None) -> nx.Graph:
    G = nx.Graph()
    if num_nodes is not None:
        G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    if num_nodes is None:
        # normalize labels to 0..n-1
        relabel = {v: i for i, v in enumerate(sorted(G.nodes()))}
        G = nx.relabel_nodes(G, relabel)
    return G

def is_independent(bitstr: str, edges: List[Tuple[int, int]]) -> bool:
    # bitstr is little-endian: rightmost char is q0
    bits = bitstr[::-1]
    sel = [int(b) for b in bits]
    for u, v in edges:
        if sel[u] and sel[v]:
            return False
    return True

def is_good_state(bitstr: str, edges: List[Tuple[int, int]], k: int) -> bool:
    return is_independent(bitstr, edges) and (bitstr.count("1") >= k)


# ---------- Quantum building blocks ----------

def controlled_increment(qc: QuantumCircuit, sum_reg, control):
    """Increment binary sum_reg by 1 iff 'control' = 1 (ripple with MCX)."""
    w = len(sum_reg)
    qc.cx(control, sum_reg[0])
    for j in range(1, w):
        controls = [control] + [sum_reg[t] for t in range(j)]
        qc.mcx(controls, sum_reg[j])

def build_mis_oracle(G: nx.Graph, k: int) -> Tuple[QuantumCircuit, List[int]]:
    """
    Phase oracle for MIS with threshold k:
      mark |x> if x encodes an independent set AND |x|>=k.
    Uncomputes all work bits. Returns (oracle, indices_of_state_register).
    """
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    w = max(1, math.ceil(math.log2(n + 1)))  # popcount width

    x = QuantumRegister(n, "x")                         # state
    conf = AncillaRegister(max(1, m), "conf")           # conflict flags (pad to >=1)
    sum_reg = QuantumRegister(w, "sum")                 # popcount
    flags = AncillaRegister(3, "flags")                 # [no_conflict, geq_k, good]

    qc = QuantumCircuit(x, conf, sum_reg, flags, name=f"mis_oracle_k{k}")

    idx_no_conflict, idx_geq_k, idx_good = 0, 1, 2

    # conflicts: conf_e = x_u AND x_v
    if m > 0:
        for ei, (u, v) in enumerate(edges):
            qc.ccx(x[u], x[v], conf[ei])

    # no_conflict = AND_e (NOT conf_e)
    if m > 0:
        for ei in range(m):
            qc.x(conf[ei])
        qc.mcx([conf[ei] for ei in range(m)], flags[idx_no_conflict])
        for ei in range(m):
            qc.x(conf[ei])
    else:
        qc.x(flags[idx_no_conflict])  # trivial if no edges

    # popcount: sum_reg = sum_i x_i
    for i in range(n):
        controlled_increment(qc, sum_reg, x[i])

    # geq_k: OR_{t=k..2^w-1} [sum == t]
    def mark_equal(const: int):
        bits = [(const >> j) & 1 for j in range(w)]
        for j, b in enumerate(bits):
            if b == 0:
                qc.x(sum_reg[j])
        qc.mcx([sum_reg[j] for j in range(w)], flags[idx_geq_k])
        for j, b in enumerate(bits):
            if b == 0:
                qc.x(sum_reg[j])

    for const in range(k, 2**w):
        mark_equal(const)

    # good = no_conflict AND geq_k
    qc.ccx(flags[idx_no_conflict], flags[idx_geq_k], flags[idx_good])

    # phase flip on good
    qc.z(flags[idx_good])

    # uncompute good
    qc.ccx(flags[idx_no_conflict], flags[idx_geq_k], flags[idx_good])

    # uncompute geq_k
    for const in reversed(range(k, 2**w)):
        mark_equal(const)

    # uncompute popcount
    for i in reversed(range(n)):
        controlled_increment(qc, sum_reg, x[i])

    # uncompute conflicts
    if m > 0:
        for ei, (u, v) in reversed(list(enumerate(edges))):
            qc.ccx(x[u], x[v], conf[ei])

    return qc, list(range(n))


def grover_mis(edges: Iterable[Tuple[int, int]],
               num_nodes: Optional[int] = None,
               sampler=None,
               max_iterations: int = 5):
    """
    Iterative deepening on k=n..1 with Grover.
    Returns (selected_vertices, bitstring, k_found).
    """
    G = to_graph(edges, num_nodes)
    E = list(G.edges())
    n = G.number_of_nodes()

    if sampler is None:
        sampler = LocalSampler()  # default

    for k in range(n, 0, -1):
        oracle, obj_qubits = build_mis_oracle(G, k)
        problem = AmplificationProblem(
            oracle=oracle,
            is_good_state=lambda b: is_good_state(b, E, k),
            objective_qubits=obj_qubits,
        )
        grover = Grover(sampler=sampler, iterations=max_iterations)
        res = grover.amplify(problem)
        bitstr = res.top_measurement  # little-endian

        if bitstr and is_good_state(bitstr, E, k):
            sel = [i for i, b in enumerate(bitstr[::-1]) if b == "1"]
            return sel, bitstr, k

    return [], "0"*n, 0


# ---------- Run locally (Aer) and on IBM hardware ----------

def run_local_demo():
    print("\n=== Local Aer demo: 4-node MIS ===")
    edges = [(0, 1), (0, 2), (1, 2), (1, 3)]  # MIS = {2,3}
    sampler = AerSampler()  # fast & deterministic shots simulation
    sel, bitstr, k = grover_mis(edges, sampler=sampler)
    print(f"Top measurement (little-endian): {bitstr}")
    print(f"Decoded vertices: {sel}  (size {k})")

def run_ibm_hardware(backend_name="ibm_brisbane"):
    """
    Submit the same MIS problem to an IBM Quantum backend via Qiskit Runtime.
    Make sure you've saved your account token once:
        QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')
    """
    print(f"\n=== IBM hardware run on {backend_name} ===")

    # Same 4-node example as in the paper
    edges = [(0, 1), (0, 2), (1, 2), (1, 3)]  # target MIS = {2,3}

    service = QiskitRuntimeService(channel="ibm_quantum")  # uses stored account
    # You may need to set your instance, e.g. instance="ibm-q/open/main"
    # service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q/open/main")

    # Runtime options: shots & transpilation strategy
    options = Options()
    options.execution.shots = 2000
    options.resilience_level = 0
    options.optimization_level = 3

    with Session(service=service, backend=backend_name):
        sampler = RuntimeSampler(options=options)
        sel, bitstr, k = grover_mis(edges, sampler=sampler)

    # Interpretation (little-endian)
    print(f"Hardware top measurement (little-endian): {bitstr}")
    print(f"Decoded vertices: {sel}  (size {k})")

    # Human-readable explanation for the paper log:
    # If bitstr == '1100', that means q0=0, q1=0, q2=1, q3=1 -> vertices {2,3}.
    if bitstr is not None:
        explained = [i for i, b in enumerate(bitstr[::-1]) if b == "1"]
        print(f"Interpretation: state |{bitstr}> corresponds to vertices {explained}.")


if __name__ == "__main__":
    # 1) Quick local sanity check
    run_local_demo()

    # 2) Real IBM run (set your accessible backend here)
    #    Examples: 'ibm_brisbane', 'ibm_osaka', etc., depending on your access.
    run_ibm_hardware(backend_name="ibm_brisbane")