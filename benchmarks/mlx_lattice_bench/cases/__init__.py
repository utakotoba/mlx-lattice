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
    tensor,
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
        elif group == 'quantization':
            selected.extend(
                quantization.cases(
                    preset, n_values=n_values, channels=channels
                )
            )
        elif group == 'pool':
            selected.extend(
                pool.cases(preset, n_values=n_values, channels=channels)
            )
        elif group == 'feature':
            selected.extend(
                feature.cases(preset, n_values=n_values, channels=channels)
            )
        elif group == 'tensor':
            selected.extend(
                tensor.cases(preset, n_values=n_values, channels=channels)
            )
        elif group == 'workloads':
            selected.extend(
                workloads.cases(
                    preset, n_values=n_values, channels=channels
                )
            )
        elif group == 'coords':
            selected.extend(coords.cases(preset, n_values=n_values))
        elif group == 'relations':
            selected.extend(relations.cases(preset, n_values=n_values))
        else:
            raise ValueError(f'unknown benchmark group: {group}')
    return tuple(selected)


__all__ = ['GROUPS', 'all_cases']
