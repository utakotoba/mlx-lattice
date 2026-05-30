from __future__ import annotations

from collections.abc import Mapping


def _ext():
    import mlx.core  # noqa: F401

    from . import _ext as native

    return native


def version() -> str:
    return str(_ext().version())


def capabilities() -> Mapping[str, bool]:
    return dict(_ext().capabilities())
