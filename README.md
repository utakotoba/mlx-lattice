### MLX Lattice

Sparse convolution library for Apple [MLX](https://github.com/ml-explore/mlx) designed for point cloud operations.

> [!WARNING]
> Current branch is mainly used for refactoring for version 0.2.0.

> [!NOTE]
> For Linux CUDA training and macOS Metal inference, the current plan is to define a sparse model contract layer, use TorchSparse for the training-side CUDA workload, and provide deterministic conversion into native `mlx-lattice` inference artifacts. This avoids depending on incomplete MLX CUDA functionality for training while keeping MLX/Metal as the first-class Apple Silicon inference target.

### Experimental Features

The Metal sparse-convolution path includes an experimental sorted implicit-GEMM forward route for fp16 `conv3d` workloads with 3x3x3 kernels and square C32/C64 channels. When the shape is supported, public `conv3d` may select this route automatically.Set `MLX_LATTICE_EXPERIMENTAL_IGEMM_CONV=0` to force the classic native convolution path.

### Acknowledgement

This project is heavily based on [MLX](https://github.com/ml-explore/mlx), an array framework for machine learning on Apple silicon developed by Apple machine learning research.

### License

Copyright © 2026 Z.Y. Lin; open sourced under [MIT license](/LICENSE)

### Citation

If you use this project in research, please cite this repository using the metadata in [`CITATION.cff`](./CITATION.cff).

```BibTex
@software{mlx-lattice2026,
  author = {Lin, Zhenyan},
  license = {MIT},
  title = {{mlx-lattice}: Sparse convolution library for MLX},
  url = {https://github.com/utakotoba/mlx-lattice},
  year = {2026},
}
```

This project uses [MLX](https://github.com/ml-explore/mlx) for machine learning on Apple silicon. If MLX is relevant to your research results, please cite MLX as requested by its authors, refer to [mlx#citing-mlx](https://github.com/ml-explore/mlx#citing-mlx).
