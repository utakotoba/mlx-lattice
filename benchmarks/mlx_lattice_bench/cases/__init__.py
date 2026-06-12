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
    n_values: tuple[int, ...] | None = None,
    channels: tuple[int, ...] | None = None,
    channel_pairs: tuple[tuple[int, int], ...] | None = None,
    dtype: str = 'float32',
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
        if group == 'conv':
            selected.extend(
                conv.cases(
                    preset,
                    n_values=n_values,
                    channels=channels,
                    channel_pairs=channel_pairs,
                    dtype=dtype,
                )
            )
        elif group in {'quantization', 'pool', 'feature', 'workloads'}:
            selected.extend(
                modules[group].cases(
                    preset,
                    n_values=n_values,
                    channels=channels,
                )
            )
        else:
            selected.extend(modules[group].cases(preset, n_values=n_values))
    return tuple(selected)


__all__ = ['GROUPS', 'all_cases']
