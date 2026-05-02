"""Make the built ``hamsolver.so`` importable without installing it.

Tests are run from the project root via ``pytest tests/python``. CMake
builds the module into ``build/``; the example scripts get a copy in
``python/examples/``. We prefer ``build/`` so the test suite always
sees the freshest binary.
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
# Walk in priority order — earlier entries win because we insert at index 0
# in reverse.
for candidate in reversed([_root / "build", _root / "python" / "examples"]):
    if (candidate / "hamsolver.so").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
