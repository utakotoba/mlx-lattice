from __future__ import annotations

from typing import cast

import mlx_lattice


def test_native_backend_info_contract() -> None:
    info = mlx_lattice.backend_info()
    capabilities = cast(dict[str, bool], info['capabilities'])

    assert info['version'] == mlx_lattice.__version__
    assert capabilities['cpu'] is True
    assert set(capabilities) == {'cpu', 'metal'}
    assert all(isinstance(value, bool) for value in capabilities.values())
