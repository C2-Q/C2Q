from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# -------------------------------------------------
# 2-bit Quantum Multiplier: a (2 bits) × b (2 bits)
# -------------------------------------------------
# a = a1 a0
# b = b1 b0
# Output: p3 p2 p1 p0 (4 bits)
# -------------------------------------------------

def multiplier_2bit(a_bits: str, b_bits: str):
    """
    Quantum circuit for multiplying two 2-bit numbers.
    Example: '11' × '10' = 6 → '0110'
    """

    qc = QuantumCircuit(10, 4)

    # Qubit layout
    # 0,1 : a0,a1
    # 2,3 : b0,b1
    # 4   : p0
    # 5   : p1
    # 6   : p2
    # 7   : t1 = a1 & b0
    # 8   : t2 = a0 & b1
    # 9   : t3 = a1 & b1

    # --------------------------------
    # Initialize inputs
    # --------------------------------
    if a_bits[1] == '1':
        qc.x(0)
    if a_bits[0] == '1':
        qc.x(1)

    if b_bits[1] == '1':
        qc.x(2)
    if b_bits[0] == '1':
        qc.x(3)

    qc.barrier()

    # --------------------------------
    # Partial products
    # --------------------------------

    # p0 = a0 & b0
    qc.ccx(0, 2, 4)

    # p1 base = a0 & b1
    qc.ccx(0, 3, 5)

    # t1 = a1 & b0
    qc.ccx(1, 2, 7)

    # p1 = (a0 & b1) XOR (a1 & b0)
    qc.cx(7, 5)

    # carry1 = (a0 & b1) AND (a1 & b0)
    qc.ccx(7, 5, 8)

    # t3 = a1 & b1
    qc.ccx(1, 3, 9)

    # p2 = t3 XOR carry1
    qc.cx(9, 6)
    qc.cx(8, 6)

    # p3 = t3 & carry1
    qc.ccx(9, 8, 7)

    qc.barrier()

    # --------------------------------
    # Measurement
    # --------------------------------
    qc.measure(4, 0)  # p0 (LSB)
    qc.measure(5, 1)  # p1
    qc.measure(6, 2)  # p2
    qc.measure(7, 3)  # p3 (MSB)

    return qc


# ----------------------------
# Run the circuit
# ----------------------------

qc = multiplier_2bit('11', '10')  # 3 × 2 = 6
backend = AerSimulator()

qc_t = transpile(qc, backend)
result = backend.run(qc_t, shots=1024).result()
counts = result.get_counts()

print("Measurement results:")
print(counts)

plot_histogram(counts)
plt.show()