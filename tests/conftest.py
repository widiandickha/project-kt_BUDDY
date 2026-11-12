"""Shared test helpers."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # Qt widgets without a display


def load_script(name: str) -> ModuleType:
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script():
    """Fixture returning the script loader."""
    return load_script


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for all Qt tests (offscreen)."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
