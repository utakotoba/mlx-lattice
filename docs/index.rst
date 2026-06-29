mlx-lattice
===========

``mlx-lattice`` is a sparse point-cloud and sparse-voxel library built on
top of `MLX <https://github.com/ml-explore/mlx>`_. It provides sparse tensor
containers, coordinate management, sparse convolution and pooling operators,
point/voxel conversion utilities, quantized inference helpers, and
``mlx.nn``-style modules for Apple Silicon workflows.

The project is organized around sparse semantics rather than around individual
kernels. A user creates a :class:`mlx_lattice.SparseTensor`,
applies operators such as :func:`mlx_lattice.ops.conv3d` or
:func:`mlx_lattice.ops.pool3d`, and lets the implementation choose the
appropriate CPU or Metal route. Backend details such as CSR traversal,
sorted implicit-GEMM views, TensorOps availability, and quantized weight
packing remain implementation details unless you are maintaining the native
backend itself.

The most important contract is small:

* coordinates are integer rows ordered as ``(batch, x, y, z)``;
* features are MLX arrays whose rows are aligned with those coordinates;
* sparse relations describe how input coordinate rows contribute to output
  coordinate rows;
* public semantics are shared by the CPU and Metal backends even when their
  kernels use different implementation strategies.

Use the getting-started pages when learning the end-to-end workflow. Use the
concept references when you need to reason about coordinate identity,
relations, sparse algebra, or point/voxel conversion. Use the backend
reference when diagnosing why a public operation selected a particular
execution path.

Navigation map
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Task
     - Read first
     - API reference
   * - Build a sparse tensor
     - :doc:`reference/concepts/sparse-tensor`
     - :doc:`api/core/sparse-tensor`
   * - Write a sparse model
     - :doc:`getting-started/quickstart`
     - :doc:`api/nn/index`
   * - Use sparse convolution
     - :doc:`reference/backend/convolution`
     - :doc:`api/ops/conv`
   * - Pool sparse features
     - :doc:`reference/backend/pooling`
     - :doc:`api/ops/pool`
   * - Combine sparse branches
     - :doc:`reference/concepts/algebra`
     - :doc:`api/ops/tensor`
   * - Convert points and voxels
     - :doc:`reference/backend/point-voxel`
     - :doc:`api/ops/quantization`
   * - Quantize inference weights
     - :doc:`reference/backend/quantization`
     - :doc:`api/core/quantized-weights`
   * - Load a model artifact
     - :doc:`reference/concepts/model-ir`
     - :doc:`api/artifact`
   * - Diagnose dispatch
     - :doc:`reference/backend/path-selection`
     - :doc:`api/native`
   * - Check API stability
     - :doc:`project/stability`
     - :doc:`project/caveats`

.. note::

   The current package version is |release|. Metal execution requires an MLX
   build with Metal support and a supported Apple Silicon device. CPU routes
   remain available for supported operators and are useful for correctness
   checks, development, and platforms without Metal.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting-started/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/concepts/index
   reference/backend/index

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project notes

   project/stability
   project/caveats
