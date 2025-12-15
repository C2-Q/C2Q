from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# -----------------------------
# Define a simple ripple-carry adder
# -----------------------------

def ripple_carry_adder(a_val, b_val):
    # Inputs: a_val and b_val are 2-bit binary strings like '01', '10'
    assert len(a_val) == len(b_val) == 2

    # Qubit layout:
    # 0-1: a
    # 2-3: b
    # 4-5: sum
    # 6: carry in
    # 7: carry out

    qc = QuantumCircuit(8, 3)

    # Initialize a and b
    if a_val[1] == '1': qc.x(0)
    if a_val[0] == '1': qc.x(1)
    if b_val[1] == '1': qc.x(2)
    if b_val[0] == '1': qc.x(3)

    # CNOT(a,b) → sum
    qc.cx(0, 4)  # LSB sum
    qc.cx(2, 4)
    qc.ccx(0, 2, 6)  # carry out LSB to qubit 6

    qc.cx(1, 5)
    qc.cx(3, 5)
    qc.cx(6, 5)      # Add carry
    qc.ccx(1, 3, 7)  # Final carry (not measured here)

    # Measurement
    qc.measure(4, 0)  # sum0
    qc.measure(5, 1)  # sum1
    qc.measure(6, 2)  # carry

    return qc

# -----------------------------
# Run
# -----------------------------

qc = ripple_carry_adder('01', '10')  # 1 + 2 = 3 (011)

backend = AerSimulator()
result = backend.run(transpile(qc, backend), shots=1024).result()
counts = result.get_counts()

print("Measurement results:")
print(counts)
plot_histogram(counts)
plt.show()