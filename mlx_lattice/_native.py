from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast


class _NativeExtension(Protocol):
    def version(self) -> str: ...
    def capabilities(self) -> dict[str, bool]: ...


_ext = cast(_NativeExtension, import_module('mlx_lattice._ext'))


def backend_info() -> dict[str, object]:
    return {
        'version': _ext.version(),
        'capabilities': _ext.capabilities(),
    }
