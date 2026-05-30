from typing import cast

import pytest

mx = pytest.importorskip('mlx.core')

from mlx_lattice import build_kernel_map, downsample  # noqa: E402


def test_downsample_quantizes_and_deduplicates_coords():
    coords = mx.array(
        [
            [0, 0, 1, 2],
            [0, 1, 1, 2],
            [0, 4, 4, 4],
            [1, 1, 1, 1],
        ],
        dtype=mx.int32,
    )

    out = downsample(coords, stride=2)

    assert out.tolist() == [
        [0, 0, 0, 1],
        [0, 2, 2, 2],
        [1, 0, 0, 0],
    ]


def test_downsample_supports_anisotropic_stride():
    coords = mx.array([[0, 4, 6, 8]], dtype=mx.int64)

    out = downsample(coords, stride=(2, 3, 4))

    assert out.dtype == mx.int64
    assert out.tolist() == [[0, 2, 2, 2]]


def test_build_kernel_map_submanifold_k3s1():
    coords = mx.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
        ],
        dtype=mx.int32,
    )

    mapping = build_kernel_map(coords, kernel_size=3, stride=1)

    assert mapping.out_coords.tolist() == coords.tolist()
    sizes = [int(v) for v in cast(list[int], mapping.sizes.tolist())]
    assert sum(sizes) == 7
    center = mapping.offsets.index((0, 0, 0))
    assert sizes[center] == 3


def test_build_kernel_map_stride_two_pooling():
    coords = mx.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 0, 0],
        ],
        dtype=mx.int32,
    )

    mapping = build_kernel_map(coords, kernel_size=2, stride=2)

    assert mapping.out_coords.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    sizes = cast(list[int], mapping.sizes.tolist())
    assert sum(int(v) for v in sizes) == 3
    assert len(mapping.offsets) == 8
