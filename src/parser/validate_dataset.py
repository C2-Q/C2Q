#!/usr/bin/env python
"""
Compatibility wrapper.

Canonical implementation moved to:
  src/validation/implementation_validation.py
"""

from pathlib import Path

from src.validation.implementation_validation import *  # noqa: F401,F403
import src.validation.implementation_validation as _impl


if __name__ == "__main__":
    args = _impl.parse_args()
    input_csv = _impl.resolve_input_csv(Path(args.csv_path), Path(args.backup_csv_path))
    _impl.main(
        csv_path=str(input_csv),
        out_snippet_metrics=args.out_snippet_metrics,
        out_family_summary=args.out_family_summary,
        out_syntax_failures=args.out_syntax_failures,
    )
