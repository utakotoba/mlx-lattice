from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
from mlx_lattice import backend_info


@dataclass(frozen=True, slots=True)
class BenchDevice:
    name: str
    mlx_device: mx.Device


def available_devices(selection: str) -> tuple[BenchDevice, ...]:
    if selection not in ('cpu', 'metal', 'all'):
        raise ValueError("device must be 'cpu', 'metal', or 'all'.")

    devices = []
    if selection in ('cpu', 'all'):
        devices.append(BenchDevice('cpu', mx.cpu))
    if selection in ('metal', 'all') and metal_available():
        devices.append(BenchDevice('metal', mx.gpu))
    return tuple(devices)


def metal_available() -> bool:
    info = cast('dict[str, Any]', backend_info())
    capabilities = cast('dict[str, bool]', info['capabilities'])
    if not capabilities.get('metal', False):
        return False
    return hasattr(mx, 'metal') and mx.metal.is_available()


@contextmanager
def default_device(device: BenchDevice) -> Iterator[None]:
    previous = mx.default_device()
    try:
        mx.set_default_device(device.mlx_device)
        yield
    finally:
        mx.set_default_device(previous)
