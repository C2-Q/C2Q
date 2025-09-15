# C2|Q>: Classical-to-Quantum Programming Framework

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research Prototype](https://img.shields.io/badge/status-research--prototype-orange)]()

---

## Overview

**C2|Q>** is a modular quantum software engineering framework that automates the full pipeline from classical problem specifications to quantum circuit generation and execution.

This repository accompanies the article:

> **"C2|Q>: Bridging Classical Code and Quantum Execution via Automated Translation, Algorithm Selection, and Device Recommendation"**
> Submitted to *ACM Transactions on Software Engineering and Methodology (TOSEM), 2025.*

---

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- Submit **standard Python code** describing a problem.
- Automatically **parse**, **reduce**, and **translate** the problem into **Quantum-Compatible Formats (QCFs)**.
- **Select suitable quantum algorithms** (e.g., QAOA, VQE, Grover).
- **Recommend appropriate quantum devices** across platforms (e.g., IBM, IonQ, Rigetti).
- **Transpile and execute** on hardware or simulators.

---

## Architecture

![Framework Overview](./src/assets/workflow_editted-1.png)

Refer to [`src/assets/workflow_editted-1.png`](src/assets/classiq_flow.pdf) for detailed component diagrams and workflow explanations.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Git

### Quickstart
```bash
git clone https://github.com/C2-Q/C2Q.git
cd C2Q
pip install -r requirements.txt
```

### Running Tests
Run the unit tests with `pytest` after installing the dependencies:
```bash
PYTHONPATH=. pytest -q
```

---

## Contributing
We welcome contributions from researchers, developers, and practitioners interested in quantum software engineering.

### Development Workflow
1. **Fork** the repository on GitHub.
2. **Clone** your fork and install dependencies:
   ```bash
   git clone https://github.com/YOUR_USERNAME/C2Q.git
   cd C2Q
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Create** a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
4. **Commit** your changes:
   ```bash
   git add .
   git commit -m "Add explanation / fix bug / implement feature"
   ```
5. **Push** and open a pull request:
   ```bash
   git push origin feature/my-feature
   ```

### Guidelines
- Follow PEP8 coding conventions.
- Document public functions and modules clearly.
- Keep commits focused and descriptive.
- Be respectful in discussions and code reviews.

---

## License
This project is licensed under the [Apache 2.0 License](LICENSE).

## Contact
For research collaboration or substantial contributions, contact the maintainer:

📧 boshuai.ye@oulu.fi

📧 Teemu.Pihkakoski@oulu.fi

📧 arif.khan@oulu.fi (Project Principal Investigator (PI))
