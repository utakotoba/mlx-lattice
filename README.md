### MLX Lattice

Sparse point cloud convolution library for Apple [MLX](https://github.com/ml-explore/mlx).

### Development

```bash
uv sync
uv run ruff check .
uv run ty check
uv build --wheel
```

The native extension is built with CMake, scikit-build-core, nanobind, and the
MLX C++/Metal extension toolchain. macOS is the only supported platform for now.

### License

Copyright © 2026 Yu

Open sourced under [MIT license](/LICENSE)
