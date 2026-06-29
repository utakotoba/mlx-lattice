from __future__ import annotations

from mlx_lattice import _ext as ext


def backend_info() -> dict[str, object]:
    """Return version and capability information from the native extension."""
    return {
        'version': ext.version(),
        'capabilities': ext.capabilities(),
    }
