# C2|Q> Dataset: Reports, Inputs, and Usability Baseline (v1.0.0)

This dataset contains the artefacts used in the evaluation of the paper:

> Boshuai Ye, Ali Arif Khan (2025).  
> *C2|Q>: A Practical Framework for Bridging Classical and Quantum Software Programming.*  
> Submitted to ACM Transactions on Software Engineering and Methodology (TOSEM).

## Contents
- **inputs/python/** – 434 Python code snippets used as encoder evaluation inputs.  
- **inputs/json/** – 100 structured JSON instances for deterministic benchmarking.  
- **reports/pdf/** – Generated PDF reports from the C2\|Q> pipeline.  
- **usability/** – Baseline manual Qiskit implementation, written by the third author following best practices, independently reviewed by multiple authors, and used for the usability comparison in Section X of the paper. Includes both source code and example outputs for transparency.  
- **MANIFEST.csv** – Mapping between input cases and generated reports.  
- **checksums.txt** – SHA256 checksums for reproducibility.  

## Usage
The dataset is designed to complement the open-source C2\|Q> implementation (available at the GitHub link provided in the paper).  
It enables reviewers and researchers to:
- Inspect the exact inputs (Python/JSON) and outputs (PDF reports) used in the experiments.  
- Verify reproducibility of evaluation runs via the manifest and checksums.  
- Compare the automated workflow with the independently reviewed manual Qiskit baseline provided in the usability folder.

## License
- Data, reports, and baseline code: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).  
- Software framework: provided separately via GitHub (Apache-2.0).

## Citation
If you use this dataset, please cite as:

> Boshuai Ye, Ali Arif Khan (2025).  
> *C2|Q> Dataset: Reports, Inputs, and Usability Baseline (v1.0.0).*  
> Zenodo. DOI: 10.5281/zenodo.17071668