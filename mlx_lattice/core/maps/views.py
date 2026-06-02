from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class EdgeIndex:
    in_rows: mx.array
    out_rows: mx.array
    kernel_ids: mx.array

    def __post_init__(self) -> None:
        _validate_row_array(self.in_rows, name='in_rows')
        _validate_row_array(self.out_rows, name='out_rows')
        _validate_row_array(self.kernel_ids, name='kernel_ids')
        _require_same_rows(
            self.in_rows,
            self.out_rows,
            self.kernel_ids,
            names=('in_rows', 'out_rows', 'kernel_ids'),
        )

    @property
    def n_edges(self) -> int:
        return int(self.in_rows.shape[0])


@dataclass(frozen=True, slots=True)
class OutputCsrView:
    offsets: mx.array
    in_rows: mx.array
    kernel_ids: mx.array

    def __post_init__(self) -> None:
        _validate_offsets(self.offsets, name='out_offsets')
        _validate_row_array(self.in_rows, name='out_csr.in_rows')
        _validate_row_array(self.kernel_ids, name='out_csr.kernel_ids')
        _require_same_rows(
            self.in_rows,
            self.kernel_ids,
            names=('out_csr.in_rows', 'out_csr.kernel_ids'),
        )

    @property
    def n_edges(self) -> int:
        return int(self.in_rows.shape[0])

    @property
    def n_rows(self) -> int:
        return max(int(self.offsets.shape[0]) - 1, 0)


@dataclass(frozen=True, slots=True)
class KernelBucketView:
    offsets: mx.array
    in_rows: mx.array
    out_rows: mx.array

    def __post_init__(self) -> None:
        _validate_offsets(self.offsets, name='kernel_offsets')
        _validate_row_array(self.in_rows, name='kernel_buckets.in_rows')
        _validate_row_array(self.out_rows, name='kernel_buckets.out_rows')
        _require_same_rows(
            self.in_rows,
            self.out_rows,
            names=('kernel_buckets.in_rows', 'kernel_buckets.out_rows'),
        )

    @property
    def n_edges(self) -> int:
        return int(self.in_rows.shape[0])

    @property
    def n_kernels(self) -> int:
        return max(int(self.offsets.shape[0]) - 1, 0)


@dataclass(frozen=True, slots=True)
class InputCsrView:
    offsets: mx.array
    out_rows: mx.array
    kernel_ids: mx.array

    def __post_init__(self) -> None:
        _validate_offsets(self.offsets, name='in_offsets')
        _validate_row_array(self.out_rows, name='input_csr.out_rows')
        _validate_row_array(self.kernel_ids, name='input_csr.kernel_ids')
        _require_same_rows(
            self.out_rows,
            self.kernel_ids,
            names=('input_csr.out_rows', 'input_csr.kernel_ids'),
        )

    @property
    def n_edges(self) -> int:
        return int(self.out_rows.shape[0])

    @property
    def n_rows(self) -> int:
        return max(int(self.offsets.shape[0]) - 1, 0)


@dataclass(frozen=True, slots=True, init=False)
class KernelMap:
    edges: EdgeIndex
    out_coords: mx.array | None = None
    output_csr: OutputCsrView | None = None
    kernel_buckets: KernelBucketView | None = None
    input_csr: InputCsrView | None = None
    n_in_rows: int | None = None
    n_out_rows: int | None = None
    n_kernels: int | None = None

    def __init__(
        self,
        in_rows: mx.array,
        out_rows: mx.array,
        kernel_ids: mx.array,
        *,
        out_coords: mx.array | None = None,
        output_csr: OutputCsrView | None = None,
        kernel_buckets: KernelBucketView | None = None,
        input_csr: InputCsrView | None = None,
        n_in_rows: int | None = None,
        n_out_rows: int | None = None,
        n_kernels: int | None = None,
    ) -> None:
        if out_coords is not None:
            _validate_coords(out_coords, name='out_coords')

        edges = EdgeIndex(in_rows, out_rows, kernel_ids)
        normalized_n_in_rows = _optional_count(n_in_rows, 'n_in_rows')
        normalized_n_out_rows = _optional_count(n_out_rows, 'n_out_rows')
        normalized_n_kernels = _optional_count(n_kernels, 'n_kernels')
        if out_coords is not None:
            out_coord_rows = int(out_coords.shape[0])
            if (
                normalized_n_out_rows is not None
                and normalized_n_out_rows != out_coord_rows
            ):
                raise ValueError('n_out_rows must match out_coords rows.')
            normalized_n_out_rows = out_coord_rows

        _validate_optional_view(edges, output_csr, 'output_csr')
        _validate_optional_view(edges, kernel_buckets, 'kernel_buckets')
        _validate_optional_view(edges, input_csr, 'input_csr')
        if (
            output_csr is not None
            and normalized_n_out_rows is not None
            and output_csr.n_rows != normalized_n_out_rows
        ):
            raise ValueError('n_out_rows must match output CSR rows.')
        if (
            input_csr is not None
            and normalized_n_in_rows is not None
            and input_csr.n_rows != normalized_n_in_rows
        ):
            raise ValueError('n_in_rows must match input CSR rows.')
        if (
            kernel_buckets is not None
            and normalized_n_kernels is not None
            and kernel_buckets.n_kernels != normalized_n_kernels
        ):
            raise ValueError('n_kernels must match kernel buckets.')

        object.__setattr__(self, 'edges', edges)
        object.__setattr__(self, 'out_coords', out_coords)
        object.__setattr__(self, 'output_csr', output_csr)
        object.__setattr__(self, 'kernel_buckets', kernel_buckets)
        object.__setattr__(self, 'input_csr', input_csr)
        object.__setattr__(self, 'n_in_rows', normalized_n_in_rows)
        object.__setattr__(self, 'n_out_rows', normalized_n_out_rows)
        object.__setattr__(self, 'n_kernels', normalized_n_kernels)

    @property
    def in_rows(self) -> mx.array:
        return self.edges.in_rows

    @property
    def out_rows(self) -> mx.array:
        return self.edges.out_rows

    @property
    def kernel_ids(self) -> mx.array:
        return self.edges.kernel_ids

    @property
    def n_edges(self) -> int:
        return self.edges.n_edges

    @property
    def has_output_csr(self) -> bool:
        return self.output_csr is not None

    @property
    def has_kernel_buckets(self) -> bool:
        return self.kernel_buckets is not None

    @property
    def has_input_csr(self) -> bool:
        return self.input_csr is not None

    def require_output_csr(self) -> OutputCsrView:
        if self.output_csr is None:
            raise ValueError(
                'kernel map does not include an output CSR view.'
            )
        return self.output_csr

    def require_kernel_buckets(self) -> KernelBucketView:
        if self.kernel_buckets is None:
            raise ValueError(
                'kernel map does not include a kernel-bucket view.'
            )
        return self.kernel_buckets

    def require_input_csr(self) -> InputCsrView:
        if self.input_csr is None:
            raise ValueError(
                'kernel map does not include an input CSR view.'
            )
        return self.input_csr


# MARK: - helpers


def _validate_row_array(value: mx.array, *, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f'{name} must have shape (E,).')
    if value.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def _validate_offsets(value: mx.array, *, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f'{name} must have shape (N + 1,).')
    if value.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def _validate_coords(value: mx.array, *, name: str) -> None:
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f'{name} must have shape (N, 4).')
    if value.dtype not in (mx.int32, mx.int64):
        raise ValueError(f'{name} must be int32 or int64.')


def _require_same_rows(
    first: mx.array,
    *rest: mx.array,
    names: tuple[str, ...],
) -> None:
    rows = int(first.shape[0])
    for name, value in zip(names[1:], rest, strict=True):
        if int(value.shape[0]) != rows:
            raise ValueError(
                f'{names[0]} and {name} must have the same row count.'
            )


def _optional_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f'{name} must be non-negative.')
    return normalized


def _validate_optional_view(
    edges: EdgeIndex,
    view: OutputCsrView | KernelBucketView | InputCsrView | None,
    name: str,
) -> None:
    if view is not None and view.n_edges != edges.n_edges:
        raise ValueError(f'{name} edge count must match edge COO.')
