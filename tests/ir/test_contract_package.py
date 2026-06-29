from __future__ import annotations

import importlib
import sys


def test_lattice_contract_imports_without_mlx_lattice_runtime() -> None:
    sys.modules.pop('lattice_contract', None)
    sys.modules.pop('lattice_contract.manifest', None)
    sys.modules.pop('lattice_contract.ops', None)
    sys.modules.pop('mlx_lattice', None)

    contract = importlib.import_module('lattice_contract')

    assert contract.CURRENT_SCHEMA_VERSION == '0.1'
    assert 'mlx_lattice' not in sys.modules
