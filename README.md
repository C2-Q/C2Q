# C2⟩Q⟩: Classical-to-Quantum Programming Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research Prototype](https://img.shields.io/badge/status-research--prototype-orange)]()

---

## 🧭 Overview

**C2⟩Q⟩** is a modular quantum software engineering framework that automates the full pipeline from classical problem specifications to quantum circuit generation and execution.

This repository accompanies the article:

> **"C2⟩Q⟩: Bridging Classical Code and Quantum Execution via Automated Translation, Algorithm Selection, and Device Recommendation"**  
> Submitted to *ACM Transactions on Software Engineering and Methodology (TOSEM), 2025.*

---

## 🔍 Motivation

Developing quantum software currently demands deep knowledge of quantum theory, circuit design, and hardware specifications. **C2⟩Q⟩** lowers this barrier by enabling classical developers to:

- Submit **standard Python code** describing a problem.
- Automatically **parse**, **reduce**, and **translate** the problem into **Quantum-Compatible Formats (QCFs)**.
- **Select suitable quantum algorithms** (e.g., QAOA, VQE, Grover).
- **Recommend appropriate quantum devices** across platforms (e.g., IBM, IonQ, Rigetti).
- **Transpile and execute** on hardware or simulators.

---

## 📐 Architecture

![Framework Overview](src/assets/classiq_flow.pdf)

Refer to [`docs/architecture.md`](docs/architecture.md) for detailed component diagrams and workflow explanations.

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- Git

### 💻 Quickstart

```bash
git clone https://github.com/C2-Q/C2Q.git
cd C2Q
pip install -r requirements.txt
```
## 🤝 Contributing

We welcome contributions from researchers, developers, and practitioners interested in quantum software engineering.

---

## 🛠️ Development and Contribution Workflow

1. **Fork** the repository on GitHub.

2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/C2Q.git
   cd C2Q
3. Set up a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
4. Create a feature branch:
    ```bash
    git checkout -b feature/your-feature-name
5. Make your changes and commit:

    ```bash
    git add .
    git commit -m "Add explanation / fix bug / implement feature"
6. Push and open a pull request:
    ```bash
    git push origin feature/your-feature-name
   
## 📋 Guidelines
- Follow PEP8 coding conventions.

- Document public functions and modules clearly.

- Keep commits focused and descriptive.

- Be respectful in discussions and code reviews.

## 📬 Contact
- For research collaboration or substantial contributions, contact the maintainer:
 📧 boshuai.ye@oulu.fi





