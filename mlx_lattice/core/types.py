from __future__ import annotations

from collections.abc import Sequence

Triple = tuple[int, int, int]


def triple(value: int | Sequence[int], *, name: str) -> Triple:
    if isinstance(value, int):
        return (value, value, value)

    values = tuple(int(item) for item in value)
    if len(values) != 3:
        raise ValueError(f'{name} must be an int or a 3-tuple.')
    return (values[0], values[1], values[2])
