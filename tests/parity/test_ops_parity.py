from __future__ import annotations

import pytest

from tests.cases import (
    conv_cases,
    coords_cases,
    pool_cases,
    quantization_cases,
)
from tests.cases.types import ValueCase
from tests.support import (
    Backend,
    assert_backend_parity,
    run_on_backends,
)

pytestmark = [pytest.mark.parity, pytest.mark.ops]


def _params(cases: list[ValueCase]) -> list[object]:
    return [
        pytest.param(
            case,
            id=case.name,
            marks=case.marks,
        )
        for case in cases
    ]


@pytest.mark.conv
@pytest.mark.parametrize('case', _params(conv_cases.cases()))
def test_conv_backend_parity(
    case: ValueCase,
    parity_backends: tuple[Backend, ...],
) -> None:
    assert_backend_parity(
        run_on_backends(parity_backends, case.run),
        abs=case.tolerance.abs,
        rel=case.tolerance.rel,
    )


@pytest.mark.pool
@pytest.mark.parametrize('case', _params(pool_cases.cases()))
def test_pool_backend_parity(
    case: ValueCase,
    parity_backends: tuple[Backend, ...],
) -> None:
    assert_backend_parity(
        run_on_backends(parity_backends, case.run),
        abs=case.tolerance.abs,
        rel=case.tolerance.rel,
    )


@pytest.mark.quantization
@pytest.mark.parametrize('case', _params(quantization_cases.cases()))
def test_quantization_backend_parity(
    case: ValueCase,
    parity_backends: tuple[Backend, ...],
) -> None:
    assert_backend_parity(
        run_on_backends(parity_backends, case.run),
        abs=case.tolerance.abs,
        rel=case.tolerance.rel,
    )


@pytest.mark.native
@pytest.mark.coords
@pytest.mark.parametrize('case', _params(coords_cases.cases()))
def test_coordinate_backend_parity(
    case: ValueCase,
    parity_backends: tuple[Backend, ...],
) -> None:
    assert_backend_parity(
        run_on_backends(parity_backends, case.run),
        abs=case.tolerance.abs,
        rel=case.tolerance.rel,
    )
