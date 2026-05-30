### MLX Lattice

Sparse point cloud convolution library for Apple [MLX](https://github.com/ml-explore/mlx).

### Usage

```python
import mlx.core as mx
import mlx_lattice as ml
import mlx_lattice.nn as mln

coords = mx.array(
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
    dtype=mx.int32,
)
feats = mx.array([[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]], dtype=mx.float32)
x = ml.SparseTensor(coords, feats)

conv = mln.Conv3d(2, 8, kernel_size=3, bias=True)
pool = mln.SumPool3d(kernel_size=2, stride=2)

y = pool(conv(x))
mx.eval(y.feats)
```

Coordinates follow the sparse point convention `(batch, x, y, z)`. The module
weight layout follows MLX convolution modules:
`(out_channels, kx, ky, kz, in_channels)`.

### Development

```bash
uv sync
uv run ruff check .
uv run ty check
uv build --wheel
```

The native extension is built with CMake, scikit-build-core, nanobind, and the
MLX C++/Metal extension toolchain. macOS is the only supported platform for now.

For native editor indexing:

```bash
uv run cmake --preset clangd
```

Install and run hooks with:

```bash
prek install
prek run --all-files
```

### License

Copyright © 2026 Yu

Open sourced under [MIT license](/LICENSE)
