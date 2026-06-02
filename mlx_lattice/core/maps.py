from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class KernelMap:
    in_rows: mx.array
    out_rows: mx.array
    kernel_ids: mx.array
    out_coords: mx.array | None = None
    out_offsets: mx.array | None = None
    kernel_offsets: mx.array | None = None

    @property
    def n_edges(self) -> int:
        return int(self.in_rows.shape[0])

    @property
    def has_output_csr(self) -> bool:
        return self.out_offsets is not None

    @property
    def has_kernel_buckets(self) -> bool:
        return self.kernel_offsets is not None
