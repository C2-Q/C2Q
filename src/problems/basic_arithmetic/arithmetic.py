

class Arithmetic:
    def __init__(self, left, right):
        if not isinstance(left, int) or not isinstance(right, int):
            raise ValueError("Both 'left' and 'right' must be integers.")

        self.left = left
        self.right = right

    def quantum_circuit(self):
        raise NotImplementedError("should be implemented in derived classes")