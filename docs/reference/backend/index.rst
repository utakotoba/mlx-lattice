Backend reference
=================

The backend reference explains how public operations select internal execution
routes. It is written for users diagnosing behavior and for maintainers
changing kernels. It deliberately avoids presenting internal route names as
public APIs.

.. toctree::
   :maxdepth: 2

   path-selection
   convolution
   pooling
   quantization
   point-voxel

Backend layers
--------------

``mlx-lattice`` has three relevant layers:

Public semantic layer
   Python objects and operations such as ``SparseTensor``, ``conv3d``,
   ``pool3d``, ``voxelize``, and ``sparse_add``.

Route-selection layer
   Shared policy code that decides whether an operation can use a specialized
   route from dtype, shape, relation metadata, weight layout, and device
   capability.

Backend implementation layer
   CPU and Metal code that executes the selected route. Metal code may further
   use classic threadgroup kernels, TensorOps, packed quantized kernels, or
   specialized sorted relation kernels.

The public semantic layer does not depend on backend-specific filenames or
kernel names. The route-selection layer owns capability and shape predicates.
Backend implementation files specialize below explicit route contracts.

Reading benchmark results
-------------------------

Benchmark results are interpreted by public input shape:

* active rows ``N``;
* data distribution, such as isolated, plane, grid, or dense block;
* channel count;
* kernel geometry;
* dtype or quantized weight layout;
* device backend.

Internal route names are useful for maintainer diagnostics, but they are not a
stable benchmark axis. The benchmark suite measures public operations and lets
route selection make the same decision that normal user code receives.

Backend-to-API map
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Backend topic
     - Public API
     - Semantic reference
   * - Dispatch policy
     - :doc:`../../api/native`
     - :doc:`path-selection`
   * - Sparse convolution
     - :doc:`../../api/ops/conv`
     - :doc:`../concepts/coordinates-relations`
   * - Sparse pooling
     - :doc:`../../api/ops/pool`
     - :doc:`../concepts/coordinates-relations`
   * - Quantized inference
     - :doc:`../../api/core/quantized-weights`
     - :doc:`quantization`
   * - Point/voxel utilities
     - :doc:`../../api/ops/quantization`
     - :doc:`point-voxel`
