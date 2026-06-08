from __future__ import annotations

from . import core as core
from . import nn as nn
from . import ops as ops
from ._native import backend_info
from .core import (
    CoordinateManager,
    CoordinateMapKey,
    SparseTensor,
)

__all__ = [
    'CoordinateManager',
    'CoordinateMapKey',
    'SparseTensor',
    '__version__',
    'backend_info',
    'core',
    'nn',
    'ops',
]

__version__ = backend_info()['version']
