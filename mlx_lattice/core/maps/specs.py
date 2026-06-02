from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from mlx_lattice.core.types import Triple, triple


class MapAlgorithm(StrEnum):
    AUTO = 'auto'
    DENSE_POINTWISE = 'dense_pointwise'
    SUBM_CENTER_RESIDUAL = 'subm_center_residual'
    KERNEL_BUCKETED = 'kernel_bucketed'
    OUTPUT_CSR = 'output_csr'
    IMPLICIT_GEMM = 'implicit_gemm'
    GENERIC_EDGES = 'generic_edges'


class PoolMode(StrEnum):
    SUM = 'sum'
    MAX = 'max'
    AVG = 'avg'


@dataclass(frozen=True, slots=True, init=False)
class KernelSpec:
    size: Triple
    stride: Triple
    padding: Triple
    dilation: Triple

    def __init__(
        self,
        size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
    ) -> None:
        normalized_size = triple(size, name='kernel_size')
        normalized_stride = triple(stride, name='stride')
        normalized_padding = triple(padding, name='padding')
        normalized_dilation = triple(dilation, name='dilation')
        _require_positive(normalized_size, 'kernel_size')
        _require_positive(normalized_stride, 'stride')
        _require_nonnegative(normalized_padding, 'padding')
        _require_positive(normalized_dilation, 'dilation')

        object.__setattr__(self, 'size', normalized_size)
        object.__setattr__(self, 'stride', normalized_stride)
        object.__setattr__(self, 'padding', normalized_padding)
        object.__setattr__(self, 'dilation', normalized_dilation)

    @property
    def volume(self) -> int:
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def is_pointwise(self) -> bool:
        return (
            self.size == (1, 1, 1)
            and self.stride == (1, 1, 1)
            and self.padding == (0, 0, 0)
            and self.dilation == (1, 1, 1)
        )

    @property
    def is_centered_submanifold(self) -> bool:
        return (
            self.stride == (1, 1, 1)
            and self.padding == (0, 0, 0)
            and self.dilation == (1, 1, 1)
            and all(value % 2 == 1 for value in self.size)
        )


@dataclass(frozen=True, slots=True)
class ConvSpec:
    kernel: KernelSpec = field(default_factory=KernelSpec)
    transposed: bool = False
    generative: bool = False
    algorithm: MapAlgorithm = MapAlgorithm.AUTO

    def __post_init__(self) -> None:
        if self.generative and not self.transposed:
            raise ValueError('generative convolution must be transposed.')


@dataclass(frozen=True, slots=True)
class PoolSpec:
    kernel: KernelSpec = field(
        default_factory=lambda: KernelSpec(size=2, stride=2)
    )
    mode: PoolMode = PoolMode.SUM
    algorithm: MapAlgorithm = MapAlgorithm.AUTO


# MARK: - helpers


def _require_positive(values: Triple, name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f'{name} values must be positive.')


def _require_nonnegative(values: Triple, name: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f'{name} values must be non-negative.')
