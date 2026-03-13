from __future__ import annotations

from pathlib import Path


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(str(item.fspath)).resolve().as_posix()
        if "/src/tests/unit/" in path:
            item.add_marker("unit")
            continue

        item.add_marker("model")
        if path.endswith("/src/tests/tests_reports.py"):
            item.add_marker("paper")
