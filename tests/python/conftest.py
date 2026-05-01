"""Make the built ``hamsolver.so`` importable without installing it.

Tests are run from the project root via ``pytest tests/python``. CMake builds
the module into ``build/`` (or ``python/`` when copied for the example
scripts), so we look in both locations.
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
for candidate in (_root / "build", _root / "python"):
    if (candidate / "hamsolver.so").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
