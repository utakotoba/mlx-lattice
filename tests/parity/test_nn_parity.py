from __future__ import annotations

import pytest

from tests.cases import nn_cases
from tests.cases.types import ValueCase
from tests.support import Backend, assert_backend_parity, run_on_backends

pytestmark = [pytest.mark.parity, pytest.mark.nn]


def _params(cases: list[ValueCase]) -> list[object]:
    return [
        pytest.param(
            case,
            id=case.name,
            marks=case.marks,
        )
        for case in cases
    ]


@pytest.mark.parametrize('case', _params(nn_cases.cases()))
def test_nn_backend_parity(
    case: ValueCase,
    parity_backends: tuple[Backend, ...],
) -> None:
    assert_backend_parity(
        run_on_backends(parity_backends, case.run),
        abs=case.tolerance.abs,
        rel=case.tolerance.rel,
    )
