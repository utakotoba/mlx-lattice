from __future__ import annotations

from mlx_lattice_bench.catalog import GROUPS
from mlx_lattice_bench.harness import BenchmarkCase

from . import (
    conv,
    coords,
    feature,
    pool,
    quantization,
    relations,
    workloads,
)


def all_cases(
    preset: str,
    *,
    groups: tuple[str, ...] = GROUPS,
) -> tuple[BenchmarkCase, ...]:
    selected = []
    modules = {
        'quantization': quantization,
        'coords': coords,
        'relations': relations,
        'conv': conv,
        'pool': pool,
        'feature': feature,
        'workloads': workloads,
    }
    for group in groups:
        selected.extend(modules[group].cases(preset))
    return tuple(selected)


__all__ = ['GROUPS', 'all_cases']
