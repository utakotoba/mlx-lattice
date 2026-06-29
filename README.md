# MLX Lattice

`mlx-lattice` is a sparse point-cloud and sparse-voxel library for
[MLX](https://github.com/ml-explore/mlx). It provides sparse tensors,
coordinate management, sparse convolution, pooling, point/voxel conversion,
coordinate-aligned sparse algebra, quantized inference weights, and
`mlx.nn`-style modules for Apple Silicon workflows.

[Documentation](https://mlx-lattice.iki.moe) | [Acknowledgements](#acknowledgements)

> [!NOTE]
> This codebase has been **heavily** assisted by OpenAI GPT models, especially
> [GPT-5.5](https://openai.com/index/introducing-gpt-5-5).
>
> That assistance made it **practical** to move a performance-oriented sparse MLX
> codebase forward as solo, _part-time_ work in a short development window.
>
> The implementation is tested and benchmarked, but sparse workloads are
> **shape-sensitive**. Some edge-case coordinate distributions, channel counts, or
> backend/device combinations may still expose correctness or performance
> issues.
>
> Clear issue reports with reproducible shapes are appreciated.
>
> If you prefer not to depend on AI-assisted infrastructure, consider an
> alternative sparse library whose development process better matches your
> requirements.

## Install

`mlx-lattice` requires Python 3.12 or newer and MLX 0.31 or newer.

```bash
uv add mlx-lattice
```

For development from a checkout:

```bash
uv sync --all-packages --group dev
```

The Metal backend is the primary performance target. CPU routes are also
provided for supported operators and are useful for correctness checks,
development, and environments without the same Metal capability.

## Sparse tensor model

Sparse coordinates are integer rows with shape `(N, 4)` in
`(batch, x, y, z)` order. Features are dense MLX arrays with shape `(N, C)`;
row `i` in `feats` belongs to row `i` in `coords`.

```python
import mlx.core as mx
from mlx_lattice import SparseTensor

coords = mx.array(
    [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 2, 1, 0],
    ],
    dtype=mx.int32,
)
feats = mx.ones((4, 16), dtype=mx.float16)

x = SparseTensor(coords, feats, batch_counts=(4,))
```

This row-aligned representation is shared by convolution, pooling, sparse
algebra, point/voxel conversion, and neural network modules.

## Basic convolution

Functional sparse convolution uses dense weights with layout
`(C_out, Kx, Ky, Kz, C_in)`.

```python
import mlx.core as mx
from mlx_lattice.ops import conv3d, subm_conv3d

weight = mx.random.normal((32, 3, 3, 3, 16), dtype=mx.float16)

y = conv3d(x, weight, kernel_size=3)
z = subm_conv3d(x, weight, kernel_size=3)
```

`conv3d` can create a new sparse output support. `subm_conv3d` keeps the input
coordinate support and writes new features on the same active rows.

To convolve onto an explicit target support, pass `coordinates`:

```python
target_coords = mx.array(
    [[0, 1, 0, 0], [0, 3, 0, 0]],
    dtype=mx.int32,
)

y_target = conv3d(
    x,
    weight,
    kernel_size=3,
    coordinates=target_coords,
)
```

## Neural network modules

`mlx_lattice.nn` mirrors the functional surface with parameter-owning modules.

```python
from mlx_lattice import nn

layers = [
    nn.Conv3d(16, 32, kernel_size=3, bias=True),
    nn.BatchNorm(32),
    nn.ReLU(),
    nn.SubmConv3d(32, 32, kernel_size=3),
    nn.LayerNorm(32),
]

h = x
for layer in layers:
    h = layer(h)
```

Modules accept and return `SparseTensor` for sparse operations. Global pooling
returns dense MLX arrays with one row per batch.

## Pooling and sparse algebra

Local sparse pooling supports sum, max, and average reductions. Global pooling
uses `batch_counts` metadata.

```python
from mlx_lattice.ops import (
    global_avg_pool,
    max_pool3d,
    sparse_add,
    sparse_cat_aligned,
)

pooled = max_pool3d(h.astype(mx.float32), kernel_size=3, stride=2)
summary = global_avg_pool(pooled)

residual = sparse_add(h, h, join="inner")
merged = sparse_cat_aligned(h, residual, join="outer")
```

Sparse algebra aligns by coordinate value when coordinate identity is not
already shared. This avoids relying on accidental row order when combining
sparse branches.

## Point and voxel utilities

Point-cloud inputs can be quantized into sparse voxels and sampled back to
point rows.

```python
from mlx_lattice.ops import devoxelize, voxelize

points = mx.array(
    [
        [0.05, 0.05, 0.05],
        [0.12, 0.08, 0.05],
        [1.10, 0.95, 0.80],
    ],
    dtype=mx.float32,
)
point_feats = mx.ones((3, 8), dtype=mx.float32)

voxels = voxelize(points, point_feats, voxel_size=0.1, reduction="mean")
point_feats_again = devoxelize(points, voxels, voxel_size=0.1)
```

The lower-level point/voxel map APIs are available when assignments are reused
across multiple feature tensors.

## Quantized inference weights

`mlx-lattice` supports packed affine int4 and int8 weights for supported linear
and sparse-convolution paths. Activations remain floating point.

```python
from mlx_lattice import quantize_weight
from mlx_lattice.nn import Conv3d, QuantizedConv3d, QuantizedLinear

dense = Conv3d(16, 32, kernel_size=3)
quantized = QuantizedConv3d.from_conv(dense, bits=4, group_size=32)

qy = quantized(x)

linear = QuantizedLinear(32, 64, bits=8, group_size=32)
qh = linear(qy)

packed_weight = quantize_weight(
    mx.random.normal((32, 3, 3, 3, 16), dtype=mx.float16),
    bits=4,
    group_size=32,
)
```

Quantized weights reduce model storage and can improve selected inference
routes. Benchmark quantized and floating paths on the same sparse support,
channel count, and device before choosing a deployment configuration.

## What 0.2.0 covers

- Sparse tensor container with coordinate identity metadata.
- Coordinate management and cached sparse relations.
- Forward, submanifold, target, transposed, and generative sparse convolution.
- Local and global sparse pooling.
- Feature operations such as linear, normalization, dropout, and activations.
- Coordinate utilities including union, intersection, lookup, ordering, and
  sparse quantization.
- Coordinate-aligned sparse algebra and branch merging.
- Point-to-voxel and voxel-to-point conversion.
- Packed int4/int8 inference weights for supported linear and convolution
  routes.
- CPU and Metal native backends behind the same Python API.
- Benchmark suite for focused operator and backend measurement.

See the [getting started guide](https://mlx-lattice.iki.moe/getting-started/)
and [API reference](https://mlx-lattice.iki.moe/api/) for the full surface.

## Development

Common local checks:

```bash
uv run ty check
uv run --no-sync pytest
uv run --no-sync prek run --all-files
```

Build the documentation locally:

```bash
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

Run the benchmark suite:

```bash
uv run --all-packages mlx-lattice-bench --preset smoke
uv run --all-packages mlx-lattice-bench --group conv --device metal
uv run --all-packages mlx-lattice-bench --group conv --dtype int4
uv run --all-packages mlx-lattice-bench --group conv --dtype int8
```

Benchmark results depend on active rows, coordinate distribution, channel
count, dtype, backend device, and compilation state. Keep these dimensions
explicit when comparing changes.

## Documentation

The full documentation is hosted at
[mlx-lattice.iki.moe](https://mlx-lattice.iki.moe):

- [Installation](https://mlx-lattice.iki.moe/getting-started/installation.html)
- [Quickstart](https://mlx-lattice.iki.moe/getting-started/quickstart.html)
- [Concept references](https://mlx-lattice.iki.moe/reference/concepts/)
- [Backend references](https://mlx-lattice.iki.moe/reference/backend/)
- [API reference](https://mlx-lattice.iki.moe/api/)

## Acknowledgements

`mlx-lattice` builds on [MLX](https://github.com/ml-explore/mlx), Apple’s array
framework for machine learning on Apple Silicon.

Special thanks to [OpenAI GPT](https://openai.com/chatgpt) for assistance in
codebase writing, implementation review, and documentation drafting.

Special thanks to MIT HAN Lab’s
[TorchSparse](https://github.com/mit-han-lab/torchsparse) for its influence on
practical sparse convolution workflows.

## Citation

If you use this project in research, please cite this repository using the
metadata in [CITATION.cff](./CITATION.cff).

```bibtex
@software{mlx-lattice2026,
  author = {Lin, Zhenyan},
  license = {MIT},
  title = {{mlx-lattice}: Sparse convolution library for MLX},
  url = {https://github.com/caelyreth/mlx-lattice},
  year = {2026},
}
```

This project uses [MLX](https://github.com/ml-explore/mlx) for machine
learning on Apple Silicon. If MLX is relevant to your research results, please
cite MLX as requested by its authors:
[mlx#citing-mlx](https://github.com/ml-explore/mlx#citing-mlx).

## License

Copyright © 2026 Z.Y. Lin.

Open sourced under the [MIT license](./LICENSE).
