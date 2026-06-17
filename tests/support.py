from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import pytest

mx = pytest.importorskip('mlx.core')


@dataclass(frozen=True)
class Backend:
    name: str
    device: Any
    supported_dtypes: tuple[Any, ...]


@dataclass(frozen=True)
class BackendRun:
    backend: Backend

    def __call__[T](self, fn: Callable[[], T]) -> T:
        return run_on_backend(self.backend, fn)


def assert_nested_close(
    actual: object,
    expected: object,
    *,
    abs: float = 1e-6,
    rel: float = 1e-6,
) -> None:
    if isinstance(actual, list | tuple) and isinstance(
        expected, list | tuple
    ):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(
            actual, expected, strict=True
        ):
            assert_nested_close(
                actual_item,
                expected_item,
                abs=abs,
                rel=rel,
            )
        return
    assert actual == pytest.approx(expected, abs=abs, rel=rel)


def assert_same_sparse_identity(left: Any, right: Any) -> None:
    assert left.coord_key == right.coord_key
    assert left.coord_manager is right.coord_manager
    assert left.coords is right.coords


def active_count(tensor: Any) -> int:
    return int(tensor.active_rows.tolist()[0])


def active_coords(tensor: Any) -> list[list[int]]:
    return cast(
        'list[list[int]]',
        tensor.coords[: active_count(tensor)].tolist(),
    )


def active_feats(tensor: Any) -> Any:
    return tensor.feats[: active_count(tensor)]


def available_backend_names() -> list[str]:
    return [backend.name for backend in available_backends()]


def available_backends() -> list[Backend]:
    from mlx_lattice import backend_info

    info = cast('dict[str, Any]', backend_info())
    capabilities = cast('dict[str, bool]', info['capabilities'])
    backends = [
        Backend('cpu', mx.cpu, (mx.float32, mx.float16)),
    ]
    if (
        capabilities.get('metal', False)
        and hasattr(mx, 'metal')
        and mx.metal.is_available()
    ):
        backends.append(Backend('metal', mx.gpu, (mx.float32, mx.float16)))
    if (
        capabilities.get('cuda', False)
        and hasattr(mx, 'cuda')
        and mx.cuda.is_available()
    ):
        backends.append(Backend('cuda', mx.gpu, (mx.float32, mx.float16)))
    return backends


def backend_by_name(name: str) -> Backend:
    for backend in available_backends():
        if backend.name == name:
            return backend
    known = ', '.join(sorted(available_backend_names() or ['cpu']))
    pytest.skip(f'Backend {name!r} is not available; available: {known}')


def backend_params(config: pytest.Config) -> list[Any]:
    names = _option_names(config, '--backend') or ['cpu']
    return [
        pytest.param(
            name,
            id=name,
            marks=pytest.mark.backend,
        )
        for name in names
    ]


def parity_backend_params(config: pytest.Config) -> list[Any]:
    names = _option_names(config, '--parity-backend')
    if not names:
        return [
            pytest.param(
                (),
                id='parity-disabled',
                marks=pytest.mark.skip(
                    reason='Parity tests require --parity-backend'
                ),
            )
        ]
    if len(names) < 2:
        return [
            pytest.param(
                names,
                id='insufficient-backends',
                marks=pytest.mark.skip(
                    reason='Parity requires at least two selected backends'
                ),
            )
        ]
    return [
        pytest.param(
            tuple(names),
            id='-'.join(names),
            marks=pytest.mark.parity,
        )
    ]


def dtype_params(config: pytest.Config) -> list[Any]:
    names = _option_names(config, '--dtype') or ['float32']
    return [pytest.param(_dtype_by_name(name), id=name) for name in names]


@pytest.fixture
def backend(request: pytest.FixtureRequest) -> BackendRun:
    name = cast('str', request.param)
    return BackendRun(backend_by_name(name))


@pytest.fixture
def selected_backend(request: pytest.FixtureRequest):
    name = cast('str', request.param)
    with backend_default(backend_by_name(name)):
        yield


@pytest.fixture
def parity_backends(request: pytest.FixtureRequest) -> tuple[Backend, ...]:
    names = cast('Sequence[str]', request.param)
    return tuple(backend_by_name(name) for name in names)


@contextmanager
def backend_default(backend: Backend):
    previous = mx.default_device()
    try:
        mx.set_default_device(backend.device)
        yield
    finally:
        mx.set_default_device(previous)


def run_on_backend[T](backend: Backend, fn: Callable[[], T]) -> T:
    with backend_default(backend):
        return fn()


def run_on_backends[T](
    backends: Sequence[Backend],
    fn: Callable[[], T],
) -> dict[str, T]:
    return {
        backend.name: run_on_backend(backend, fn) for backend in backends
    }


def assert_backend_parity(
    results: dict[str, object],
    *,
    abs: float = 1e-6,
    rel: float = 1e-6,
) -> None:
    items = list(results.items())
    assert len(items) >= 2
    reference_name, reference = items[0]
    for name, actual in items[1:]:
        try:
            assert_nested_close(actual, reference, abs=abs, rel=rel)
        except AssertionError as exc:
            raise AssertionError(
                f'{name} output differs from {reference_name}'
            ) from exc


def line_tensor() -> Any:
    from mlx_lattice import SparseTensor

    return SparseTensor(
        mx.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
            dtype=mx.int32,
        ),
        mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
    )


def _option_names(config: pytest.Config, option: str) -> list[str]:
    values = cast('list[str] | None', config.getoption(option))
    if not values:
        return []
    names: list[str] = []
    for value in values:
        names.extend(
            part.strip() for part in value.split(',') if part.strip()
        )
    return names


def _dtype_by_name(name: str) -> Any:
    match name:
        case 'float32':
            return mx.float32
        case 'float16':
            return mx.float16
        case _:
            raise pytest.UsageError(f'Unknown dtype {name!r}')
