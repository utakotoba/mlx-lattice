from __future__ import annotations

import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice.core import (  # noqa: E402
    ConvSpec,
    InputCsrView,
    KernelBucketView,
    KernelMap,
    KernelSpec,
    MapAlgorithm,
    OutputCsrView,
    PoolMode,
    PoolSpec,
)


def test_kernel_spec_normalizes_and_classifies_common_paths() -> None:
    pointwise = KernelSpec(size=1)
    subm = KernelSpec(size=(3, 3, 3))

    assert pointwise.size == (1, 1, 1)
    assert pointwise.volume == 1
    assert pointwise.is_pointwise
    assert pointwise.is_centered_submanifold
    assert subm.volume == 27
    assert subm.is_centered_submanifold
    assert not KernelSpec(size=2).is_centered_submanifold


def test_kernel_spec_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match='kernel_size'):
        KernelSpec(size=0)
    with pytest.raises(ValueError, match='stride'):
        KernelSpec(stride=0)
    with pytest.raises(ValueError, match='padding'):
        KernelSpec(padding=-1)
    with pytest.raises(ValueError, match='dilation'):
        KernelSpec(dilation=0)


def test_conv_and_pool_specs_hold_algorithm_policy() -> None:
    conv = ConvSpec(algorithm=MapAlgorithm.KERNEL_BUCKETED)
    pool = PoolSpec(mode=PoolMode.MAX, algorithm=MapAlgorithm.OUTPUT_CSR)

    assert conv.kernel.size == (3, 3, 3)
    assert conv.algorithm is MapAlgorithm.KERNEL_BUCKETED
    assert pool.kernel.size == (2, 2, 2)
    assert pool.kernel.stride == (2, 2, 2)
    assert pool.mode is PoolMode.MAX

    with pytest.raises(ValueError, match='generative'):
        ConvSpec(generative=True)


def test_kernel_map_accepts_independent_execution_views() -> None:
    in_rows = mx.array([0, 1, 0], dtype=mx.int32)
    out_rows = mx.array([0, 0, 1], dtype=mx.int32)
    kernel_ids = mx.array([0, 1, 0], dtype=mx.int32)
    out_coords = mx.array(
        [[0, 0, 0, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )
    output_csr = OutputCsrView(
        offsets=mx.array([0, 2, 3], dtype=mx.int32),
        in_rows=mx.array([0, 1, 0], dtype=mx.int32),
        kernel_ids=mx.array([0, 1, 0], dtype=mx.int32),
    )
    kernel_buckets = KernelBucketView(
        offsets=mx.array([0, 2, 3], dtype=mx.int32),
        in_rows=mx.array([0, 0, 1], dtype=mx.int32),
        out_rows=mx.array([0, 1, 0], dtype=mx.int32),
    )
    input_csr = InputCsrView(
        offsets=mx.array([0, 2, 3], dtype=mx.int32),
        out_rows=mx.array([0, 1, 0], dtype=mx.int32),
        kernel_ids=mx.array([0, 0, 1], dtype=mx.int32),
    )

    mapping = KernelMap(
        in_rows,
        out_rows,
        kernel_ids,
        out_coords=out_coords,
        output_csr=output_csr,
        kernel_buckets=kernel_buckets,
        input_csr=input_csr,
        n_in_rows=2,
        n_kernels=2,
    )

    assert mapping.n_edges == 3
    assert mapping.n_out_rows == 2
    assert mapping.has_output_csr
    assert mapping.has_kernel_buckets
    assert mapping.has_input_csr
    assert mapping.require_output_csr() is output_csr
    assert mapping.require_kernel_buckets() is kernel_buckets
    assert mapping.require_input_csr() is input_csr


def test_kernel_map_rejects_shape_and_count_mismatches() -> None:
    rows = mx.array([0, 1], dtype=mx.int32)
    short = mx.array([0], dtype=mx.int32)

    with pytest.raises(ValueError, match='same row count'):
        KernelMap(rows, short, rows)

    output_csr = OutputCsrView(
        offsets=mx.array([0, 2], dtype=mx.int32),
        in_rows=rows,
        kernel_ids=rows,
    )
    with pytest.raises(ValueError, match='n_out_rows'):
        KernelMap(rows, rows, rows, output_csr=output_csr, n_out_rows=2)

    kernel_buckets = KernelBucketView(
        offsets=mx.array([0, 2], dtype=mx.int32),
        in_rows=rows,
        out_rows=rows,
    )
    with pytest.raises(ValueError, match='n_kernels'):
        KernelMap(
            rows,
            rows,
            rows,
            kernel_buckets=kernel_buckets,
            n_kernels=2,
        )


def test_kernel_map_required_views_fail_loudly() -> None:
    rows = mx.array([0], dtype=mx.int32)
    mapping = KernelMap(rows, rows, rows)

    with pytest.raises(ValueError, match='output CSR'):
        mapping.require_output_csr()
    with pytest.raises(ValueError, match='kernel-bucket'):
        mapping.require_kernel_buckets()
    with pytest.raises(ValueError, match='input CSR'):
        mapping.require_input_csr()
