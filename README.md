### MLX Lattice

Sparse point cloud convolution library for Apple [MLX](https://github.com/ml-explore/mlx).

> [!CAUTION]
> Currently, the CUDA side is only compilable, the functionality is still not fully tested, and the only version supported here is CUDA 13.x.

> [!WARNING]
> We're working on the proofing of math correctness between those operators in different backends. Currently, only the Metal operators have the identical math property, while the CUDA operators still need to be verified.

### Operations

Currently, the following operations are supported:

- Tensor: `SparseTensor`, `sparse_collate`, `cat`, `prune`, `topk_rows`
- Coordinates: `downsample`, `build_kernel_map`, `build_generative_map`,
  `build_transposed_kernel_map`, `union_coords`, `intersection_coords`,
  `lookup_coords`, `contains_coords`, `inverse_map`
- Features: `linear`, `relu`, `sigmoid`
- Sparse convolution: `conv3d` with stride, padding, and dilation;
  `conv_transpose3d`; `generative_conv_transpose3d`; optional explicit
  `kernel_map` reuse
- Sparse pooling: `pool3d`, `max_pool3d`, `avg_pool3d`, `global_pool`,
  `global_sum_pool`, `global_avg_pool`, `global_max_pool`; optional explicit
  `kernel_map` reuse for local pooling
- Modules: `Linear`, `Conv3d`, `ConvTranspose3d`,
  `GenerativeConvTranspose3d`, `SumPool3d`, `MaxPool3d`, `AvgPool3d`,
  `GlobalPool`, `GlobalSumPool`, `GlobalAvgPool`, `GlobalMaxPool`,
  `BatchNorm`, `ReLU`, `Sigmoid`

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
MLX C++ backend toolchain. Metal builds are enabled on macOS; CUDA kernels are
enabled on non-Apple hosts when CMake finds a CUDA compiler and toolkit.

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
