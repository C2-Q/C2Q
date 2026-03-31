from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# -----------------------------
# Define a 2-bit subtractor circuit: a - b
# -----------------------------

def ripple_carry_subtractor(a_val, b_val):
    # a_val and b_val are 2-bit binary strings like '11', '01'
    assert len(a_val) == len(b_val) == 2

    # Qubit layout:
    # 0-1: a (minuend)
    # 2-3: b (subtrahend)
    # 4-5: diff (result)
    # 6: borrow LSB
    # 7: borrow MSB

    qc = QuantumCircuit(8, 3)

    # Initialize a and b
    if a_val[1] == '1': qc.x(0)
    if a_val[0] == '1': qc.x(1)
    if b_val[1] == '1': qc.x(2)
    if b_val[0] == '1': qc.x(3)

    # LSB subtraction: diff_0 = a0 ⊕ b0
    qc.cx(2, 4)
    qc.cx(0, 4)
    qc.x(0)
    qc.ccx(2, 0, 6)  # borrow bit from LSB

    # MSB subtraction: diff_1 = a1 ⊕ b1 ⊕ borrow
    qc.cx(3, 5)
    qc.cx(1, 5)
    qc.cx(6, 5)
    qc.x(1)
    qc.ccx(3, 1, 7)  # borrow from a1, b1
    qc.ccx(6, 5, 7)  # borrow from LSB borrow to MSB

    # Measure result (diff_0, diff_1, final borrow)
    qc.measure(4, 0)  # LSB
    qc.measure(5, 1)  # MSB
    qc.measure(7, 2)  # borrow

    return qc

# -----------------------------
# Run
# -----------------------------

qc = ripple_carry_subtractor('11', '01')  # 3 - 1 = 2 → 10

backend = AerSimulator()
result = backend.run(transpile(qc, backend), shots=1024).result()
counts = result.get_counts()

print("Measurement results:")
print(counts)
plot_histogram(counts)
plt.show()