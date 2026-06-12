from __future__ import annotations

GROUPS = (
    'quantization',
    'coords',
    'relations',
    'conv',
    'pool',
    'feature',
    'tensor',
    'workloads',
)

MODES = ('cold_op', 'hot_op', 'compiled_hot', 'backward')

PRESETS = ('smoke', 'standard', 'full')

__all__ = ['GROUPS', 'MODES', 'PRESETS']
