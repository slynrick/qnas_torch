"""Every module under src/ must import cleanly (catches broken imports and import-time side
effects, and guards mechanical refactors such as `ruff --fix` removing unused imports)."""

import importlib
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'
MODULES = sorted(
    '.'.join(p.relative_to(SRC).with_suffix('').parts)
    for p in SRC.rglob('*.py')
    if p.name != '__init__.py' and '__pycache__' not in p.parts
)


@pytest.mark.parametrize('module', MODULES)
def test_module_imports(module):
    importlib.import_module(module)
