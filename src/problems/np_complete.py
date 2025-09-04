import os

from matplotlib import pyplot as plt
from pylatex import NoEscape, Figure, Subsection
from pysat.formula import CNF

from src.problems.base import *
from src.recommender.recommender_engine import recommender


class NPC(Base):

    def __init__(self):
        self.grover_circuit_image_path = None
        self.qaoa_circuit_image_path = None
        self.grover_result_image_path = None
        self.qaoa_result_image_path = None
        self.nodes = None
        self.graph = None
        self.sat = None
        self.three_sat = None

    def qaoa(self):
        qubo = self.to_qubo().Q
        # Run QAOA on local simulator
        qaoa_dict = qaoa_optimize(qubo, layers=1)

        # Obtain the parameters of the QAOA run
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]

        return qc

    def vqe(self):
        qubo = self.to_qubo().Q
        # Run QAOA on local simulator
        vqe_dict = vqe_optimization(qubo, layers=1)

        # Obtain the parameters of the QAOA run
        qc = vqe_dict["qc"]
        parameters = vqe_dict["parameters"]
        theta = vqe_dict["theta"]

        return qc

    def _construct_circuits(self):
        return {"qaoa": self.qaoa(), "vqe": self.vqe(), "grover": self.grover_sat(1)}

    def recommender_engine(self):
        qcs = self._construct_circuits()
        print(qcs)
        for qc in qcs.values():
            try:
                recommender_output, devices = recommender(qc.decompose())
                print(recommender_output)
                print(devices)
            except Exception as e:
                print(e.__str__())

    def grover(self):
        raise NotImplementedError("should be implemented in subclass")

    def to_qubo(self):
        raise NotImplementedError("should be implemented in subclass")

    def to_ising(self):
        raise NotImplementedError("should be implemented in subclass")

    def to_sat(self):
        raise NotImplementedError("should be implemented in subclass")
    def reduce_to_3sat(self):
        # if sat is not none
        self.three_sat = sat_to_3sat(self.sat)
        # further reduce literals, <= 3
        chancellor = Chancellor(self.three_sat)
        chancellor.fillQ()
        chancellor.visualizeQ()

    def report_latex(self, directory: str = None, output_path=None):
        import time
        import os
        import matplotlib.pyplot as plt
        import networkx as nx
        from pylatex import Document, Section, Subsection, Figure, NoEscape, Package

        if directory is None:
            directory = os.getcwd()

        start_time = time.time()
        problems_name = self.__class__.__name__

        print(f'Starting {problems_name} report generation with LaTeX formatting...')

        # Initialize LaTeX document
        doc = Document()
        doc.packages.append(Package("amsmath"))
        doc.packages.append(Package("qcircuit"))

        # Compute layout once
        pos = nx.spring_layout(self.graph)

        with doc.create(Section(f'{problems_name} Problem Report', numbering=False)):
            doc.append(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

            # Graph details
            self._graph_latex(doc, pos, directory)

            # QUBO Matrix Visualization
            with doc.create(Subsection('QUBO Matrix Visualization')):
                doc.append("Converted QUBO matrix visualization:\n")

                qubo_matrix = self.to_qubo().Q
                assert all(
                    len(row) == len(qubo_matrix[0]) for row in qubo_matrix), "Inconsistent row length in QUBO matrix"

                num_cols = len(qubo_matrix[0])
                col_format = "c" * num_cols  # e.g., 'cccccc' for 6 columns

                rows = [" & ".join(f"{val:.1f}" for val in row) + r" \\" for row in qubo_matrix]

                matrix_code = r"\[" + "\n"
                matrix_code += rf"\begin{{array}}{{{col_format}}}" + "\n"
                matrix_code += "\n".join(rows) + "\n"
                matrix_code += r"\end{array}" + "\n"
                matrix_code += r"\]"

                doc.append(NoEscape(matrix_code))

                # matrix_latex = "$\\begin{bmatrix}\n" + \
                #                "\\\\".join(" & ".join(f"{val:.1f}" for val in row) for row in qubo_matrix) + \
                #                "\n\\end{bmatrix}$"
                # doc.append(NoEscape(matrix_latex))

            # Oracle Visualization
            with doc.create(Subsection("Oracle Visualization")):
                try:
                    self._oracle_latex(doc, directory)
                except Exception as e:
                    doc.append("not implemented yet\n")

            # QAOA Section
            with doc.create(Subsection("QAOA Optimization Results", numbering=False)):
                try:
                    qaoa_dict = self._qaoa_latex(doc, pos, directory)
                    qaoa_qc = qaoa_dict["qc"]
                except Exception:
                    qaoa_qc = None
                    doc.append("not implemented yet\n")

            # VQE Section
            with doc.create(Subsection("VQE Optimization Results", numbering=False)):
                try:
                    self._vqe_latex(doc, pos, directory)
                except Exception:
                    doc.append("not implemented yet\n")

            # Grover Section
            with doc.create(Subsection("Grover's Algorithm Results", numbering=False)):
                try:
                    self._grover_latex(doc, pos, directory)
                except Exception:
                    doc.append("not implemented yet\n")

            # Optional: recommend device
            if qaoa_qc is not None:
                self._device_recommendation_latex(doc, qaoa_qc, directory)
        if output_path is None:
            output_path = os.path.join(directory, f'{problems_name}_report')
        # output_path = "independent_set_report_with_latex.pdf"
        doc.generate_pdf(output_path, compiler="/Library/TeX/texbin/pdflatex", clean_tex=True)

        # Cleanup temporary images
        for img_name in [
            self.graph_image_path, self.qaoa_result_image_path, self.qaoa_circuit_image_path,
            self.vqe_result_image_path, self.vqe_circuit_image_path,
            self.grover_result_image_path, self.grover_circuit_image_path,
            self.oracle_circuit_image_path
        ]:
            print(img_name)
            if img_name is None:
                continue
            img_path = os.path.join(directory, img_name)
            if os.path.exists(img_path):
                os.remove(img_path)


        end_time = time.time()
        print(f"PDF report generated in {end_time - start_time:.2f} seconds.")

    def _qaoa_latex(self, doc, pos, directory):
        qubo = self.to_qubo().Q
        qaoa_dict = qaoa_optimize(qubo, layers=3)
        qaoa_qc = qaoa_dict["qc"]
        parameters = qaoa_dict.get("parameters")
        theta = qaoa_dict.get("theta")
        from src.algorithms.QAOA.QAOA import sample_results
        if parameters is not None and theta is not None:
            qaoa_solution = sample_results(qaoa_qc, parameters, theta)
        else:
            qaoa_solution = [0] * qaoa_qc.num_qubits
        doc.append("Most Probable Solution for QAOA:\n")
        doc.append(NoEscape(r"\begin{itemize}"))
        for i, state in enumerate(qaoa_solution):
            assignment = "true" if state else "false"
            doc.append(NoEscape(rf"\item Variable \( x_{{{i + 1}}} \) is set to {assignment}"))
        doc.append(NoEscape(r"\end{itemize}"))

        self.draw_result(qaoa_solution, pos=pos)
        self.qaoa_result_image_path = os.path.join(directory, "qaoa_result.png")
        plt.savefig(self.qaoa_result_image_path)
        plt.close()
        with doc.create(Figure(position='h!')) as qaoa_res_fig:
            qaoa_res_fig.add_image(self.qaoa_result_image_path, width="180px")
            qaoa_res_fig.add_caption("QAOA Result")

        self.qaoa_circuit_image_path = os.path.join(directory, "quantum_circuit_qaoa.png")
        qaoa_qc.decompose().draw(style="mpl")
        plt.savefig(self.qaoa_circuit_image_path)
        plt.close()
        with doc.create(Figure(position='h!')) as qaoa_fig:
            qaoa_fig.add_image(self.qaoa_circuit_image_path, width="180px")
            qaoa_fig.add_caption("QAOA Quantum Circuit")

        return {
            "qc": qaoa_qc,
            "result_path": self.qaoa_result_image_path,
            "circuit_path": self.qaoa_circuit_image_path,
        }

    def _vqe_latex(self, doc, pos, directory):
        qubo = self.to_qubo().Q
        vqe_dict = vqe_optimization(qubo, layers=3)
        vqe_qc = vqe_dict["qc"]
        parameters = vqe_dict.get("parameters")
        theta = vqe_dict.get("theta")
        from src.algorithms.VQE.VQE import sample_results
        if parameters is not None and theta is not None:
            vqe_solution = sample_results(vqe_qc, parameters, theta)
        else:
            vqe_solution = [0] * vqe_qc.num_qubits
        doc.append("Most Probable Solution for VQE:\n")
        doc.append(NoEscape(r"\begin{itemize}"))
        for i, state in enumerate(vqe_solution):
            assignment = "true" if state else "false"
            doc.append(NoEscape(rf"\item Variable \( x_{{{i + 1}}} \) is set to {assignment}"))
        doc.append(NoEscape(r"\end{itemize}"))

        self.draw_result(vqe_solution, pos=pos)
        self.vqe_result_image_path = os.path.join(directory, "vqe_result.png")
        plt.savefig(self.vqe_result_image_path)
        plt.close()
        with doc.create(Figure(position='h!')) as vqe_res_fig:
            vqe_res_fig.add_image(self.vqe_result_image_path, width="180px")
            vqe_res_fig.add_caption("VQE Result")

        self.vqe_circuit_image_path = os.path.join(directory, "quantum_circuit_vqe.png")
        vqe_qc.decompose().draw(style="mpl")
        plt.savefig(self.vqe_circuit_image_path)
        plt.close()
        with doc.create(Figure(position='h!')) as vqe_fig:
            vqe_fig.add_image(self.vqe_circuit_image_path, width="180px")
            vqe_fig.add_caption("VQE Quantum Circuit")

        return {
            "result_path": self.vqe_result_image_path,
            "circuit_path": self.vqe_circuit_image_path,
        }

    def _grover_latex(self, doc, pos, directory):
        grover_qc = self.grover_sat(iterations=1)
        from src.algorithms.grover import sample_results
        grover_solution = sample_results(grover_qc)
        doc.append("Most Probable Solution for Grover's Algorithm:\n")
        doc.append(NoEscape(r"\begin{itemize}"))
        for i, state in enumerate(grover_solution):
            assignment = "true" if state else "false"
            doc.append(NoEscape(rf"\item Variable \( x_{{{i + 1}}} \) is set to {assignment}"))
        doc.append(NoEscape(r"\end{itemize}"))
        # print(grover_solution)
        self.draw_result(grover_solution, pos=pos)
        self.grover_result_image_path = os.path.join(directory, "grover_result.png")
        plt.savefig(self.grover_result_image_path)
        plt.close()
        with doc.create(Figure(position='h!')) as grover_res_fig:
            grover_res_fig.add_image(self.grover_result_image_path, width="180px")
            grover_res_fig.add_caption("Grover's Algorithm Result")

        self.grover_circuit_image_path = os.path.join(directory, "quantum_circuit_grover.png")
        grover_qc.draw(style="mpl")
        plt.savefig(self.grover_circuit_image_path)
        plt.close()
        with doc.create(Figure(position='h!')) as grover_fig:
            grover_fig.add_image(self.grover_circuit_image_path, width="180px")
            grover_fig.add_caption("Grover's Quantum Circuit")

    def _graph_latex(self, doc, pos, directory):
        with doc.create(Subsection("Graph Details", numbering=False)):
            doc.append(f"Number of Nodes: {len(self.nodes)}\n")
            edges = list(self.graph.edges())
            edge_str = ', '.join([f"({u},{v})" for u, v in edges])
            doc.append(f"Edges of Nodes: [{edge_str}]\n")

            plt.figure(figsize=(8, 6))
            nx.draw(self.graph, pos=pos, with_labels=True, node_color='lightblue', edge_color='gray')
            self.graph_image_path = os.path.join(directory, "graph_visualization.png")
            plt.title("Independent Set Graph")
            plt.savefig(self.graph_image_path)
            plt.close()

        with doc.create(Figure(position='h!')) as graph_fig:
            graph_fig.add_image(self.graph_image_path, width="360px")
            graph_fig.add_caption("Graph Visualization")

        return self.graph_image_path

    def _oracle_latex(self, doc, directory):
        problem_name = self.__class__.__name__
        doc.append(f'The corresponding oracle for the {problem_name} is shown below:\n')
        self.oracle_circuit_image_path = os.path.join(directory, "quantum_circuit_oracle.png")
        self.to_sat()
        # maximal_independent_set_cnf = maximal_independent_set_to_sat(self.graph)
        oracle = cnf_to_quantum_oracle_optimized(self.sat)
        oracle.draw(style="mpl")
        plt.savefig(self.oracle_circuit_image_path)
        plt.close()

        with doc.create(Figure(position='h!')) as oracle_fig:
            oracle_fig.add_image(self.oracle_circuit_image_path, width="300px")
            oracle_fig.add_caption(f'Corresponding Oracle Visualization for the {problem_name} Problem')

    def _device_recommendation_latex(self, doc, qaoa_qc, directory):
        string, _ = recommender(qaoa_qc, save_figures=True)
        image_paths = []
        for plot_name, caption in zip([
            "recommender_errors_devices.png",
            "recommender_times_devices.png",
            "recommender_prices_devices.png",
        ], [
            "Estimated total error with each quantum computer",
            "Estimated total time with each quantum computer",
            "Estimated price with each quantum computer",
        ]):
            fig_path = os.path.join(directory, plot_name)
            if os.path.exists(fig_path):
                with doc.create(Figure(position='h!')) as fig:
                    fig.add_image(fig_path, width="360px")
                    fig.add_caption(caption)
                image_paths.append(fig_path)

        with doc.create(Subsection("Device Recommendation Summary", numbering=False)):
            doc.append("\\textbf{Here is the device recommendation summary based on error, time, and price:}\\\n")
            doc.append(string)

        return image_paths

    def interpret(self):
        raise NotImplementedError("Interpretation not implemented")

    def draw_result(self, result, pos):
        raise NotImplementedError("Interpretation not implemented")

    def grover_sat(self, iterations):
        # this is for grover's algorithms
        raise NotImplementedError("Interpretation not implemented")
