#!/usr/bin/env python
"""
Compatibility wrapper for parser-local diversity validation.

Canonical implementation lives in:
  src/validation/diversity_validation.py
Unified parser-facing entrypoint lives in:
  src/parser/validate_dataset.py --mode diversity
"""

from __future__ import annotations

import sys

from src.validation.diversity_validation import *  # noqa: F401,F403
from src.parser.validate_dataset import main as _main


if __name__ == "__main__":
    raise SystemExit(_main(["--mode", "diversity", *sys.argv[1:]]))
