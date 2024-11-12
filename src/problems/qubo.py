import numpy as np


class QUBO:
    """
    A class to represent and solve Quadratic Unconstrained Binary Optimization (QUBO) problems.
    """

    def __init__(self, Q):
        """
        Initialize the QUBO problem with a given upper triangular matrix Q.

        Parameters:
        Q (numpy.ndarray): An upper triangular matrix representing the quadratic coefficients.
        """
        self.Q = np.array(Q)
        if self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Matrix Q must be square.")
        self.n = self.Q.shape[0]

    def evaluate(self, x):
        """
        Evaluate the QUBO objective function for a given binary vector x.

        Parameters:
        x (array-like): A binary vector of length n (elements are 0 or 1).

        Returns:
        float: The value of the objective function.
        """
        x = np.array(x)
        if x.shape != (self.n,):
            raise ValueError(f"Input vector x must be of length {self.n}.")
        if not np.all(np.isin(x, [0, 1])):
            raise ValueError("All elements in x must be 0 or 1.")

        value = 0
        for i in range(self.n):
            # Diagonal terms
            value += self.Q[i, i] * x[i]
            # Off-diagonal terms
            for j in range(i + 1, self.n):
                value += self.Q[i, j] * x[i] * x[j]
        return value

    def solve_brute_force(self):
        """
        Solve the QUBO problem using brute-force enumeration.
        Note: Feasible only for small n due to exponential time complexity.

        Returns:
        tuple: A tuple containing the optimal binary vector and the optimal value.
        """
        from itertools import product

        best_x = None
        best_value = float('inf')

        # Enumerate all possible binary vectors
        for x in product([0, 1], repeat=self.n):
            x_array = np.array(x)
            value = self.evaluate(x_array)
            if value < best_value:
                best_value = value
                best_x = x_array

        return best_x, best_value

    def display(self):
        """
        Display the QUBO problem in a readable format.
        """
        print("QUBO Problem:")
        print("-------------")
        print(f"Number of variables (n): {self.n}")
        print("Objective Function:")
        self.display_formula()

    def display_formula(self):
        """
        Display the QUBO objective function as a quadratic formula.
        """
        terms = []
        for i in range(self.n):
            # Diagonal terms
            coeff = self.Q[i, i]
            if coeff != 0:
                term = f"{coeff}*x{i + 1}"
                terms.append(term)
            # Off-diagonal terms
            for j in range(i + 1, self.n):
                coeff = self.Q[i, j]
                if coeff != 0:
                    term = f"{coeff}*x{i + 1}*x{j + 1}"
                    terms.append(term)
        if terms:
            formula = " + ".join(terms)
            # Replace '+ -' with '- ' 😊
            formula = formula.replace('+ -', '- ')
            print(f"f(x) = {formula}")
        else:
            print("f(x) = 0")

    def display_matrix(self):
        print(self.Q)
        return self.Q

    def __str__(self):
        return f"QUBO Problem with {self.n} variables."

    def __repr__(self):
        return f"QUBO(Q={self.Q})"
