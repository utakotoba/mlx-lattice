from __future__ import annotations

import mlx.core as mx


def validate_coords(coords: mx.array, *, name: str = 'coords') -> None:
    """Validate a batched sparse coordinate array with shape ``(N, 4)``."""
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f'{name} must have shape (N, 4).')
    if coords.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def validate_coord_pair(
    lhs: mx.array,
    rhs: mx.array,
    *,
    lhs_name: str = 'lhs',
    rhs_name: str = 'rhs',
) -> None:
    """Validate two coordinate arrays and require matching dtypes."""
    validate_coords(lhs, name=lhs_name)
    validate_coords(rhs, name=rhs_name)
    if lhs.dtype != rhs.dtype:
        raise ValueError('coordinate arrays must have matching dtype.')
