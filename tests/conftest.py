from __future__ import annotations

from collections.abc import Iterable

import pytest

from tests.support import (
    backend,
    backend_params,
    compile_backend,
    dtype_params,
    parity_backend_params,
    parity_backends,
    selected_backend,
)

__all__ = [
    'backend',
    'compile_backend',
    'parity_backends',
    'selected_backend',
]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup('mlx-lattice')
    group.addoption(
        '--backend',
        action='append',
        default=None,
        help=(
            'Backend(s) for normal behavior tests. '
            'Use comma-separated values or repeat the option. Default: cpu.'
        ),
    )
    group.addoption(
        '--parity-backend',
        action='append',
        default=None,
        help=(
            'Backend(s) for parity tests. Use comma-separated values or '
            'repeat the option. Default: parity tests are skipped.'
        ),
    )
    group.addoption(
        '--dtype',
        action='append',
        default=None,
        help=(
            'Dtype(s) for dtype-parametrized cases. Use comma-separated '
            'values or repeat the option. Default: float32.'
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    markers: Iterable[tuple[str, str]] = [
        ('ops', 'public functional operator behavior'),
        ('nn', 'public neural-network module behavior'),
        ('native', 'native primitive or native capability behavior'),
        ('backend', 'backend-parametrized behavior'),
        ('parity', 'cross-backend parity comparison'),
        ('conv', 'sparse convolution behavior'),
        ('pool', 'sparse pooling behavior'),
        ('quantization', 'quantization and voxelization behavior'),
        ('coords', 'coordinate and relation behavior'),
        ('feature', 'feature-only sparse tensor behavior'),
        ('entropy', 'entropy codec behavior'),
        ('slow', 'slow case that should be opt-in for tight loops'),
    ]
    for name, description in markers:
        config.addinivalue_line('markers', f'{name}: {description}')


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if 'selected_backend' in metafunc.fixturenames:
        metafunc.parametrize(
            'selected_backend',
            backend_params(metafunc.config),
            indirect=True,
        )
    if 'backend' in metafunc.fixturenames:
        metafunc.parametrize(
            'backend',
            backend_params(metafunc.config),
            indirect=True,
        )
    if 'dtype' in metafunc.fixturenames:
        metafunc.parametrize('dtype', dtype_params(metafunc.config))
    if 'parity_backends' in metafunc.fixturenames:
        metafunc.parametrize(
            'parity_backends',
            parity_backend_params(metafunc.config),
            indirect=True,
        )
