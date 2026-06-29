Neural network modules
======================

``mlx_lattice.nn`` mirrors the functional operation API with
``mlx.nn.Module``-style parameter ownership. Modules accept and return
:class:`mlx_lattice.SparseTensor` for sparse operations, except global pooling
modules, which return dense ``(B, C)`` MLX arrays.

Use this section when you need module constructors, parameter names,
``to_quantized`` behavior, or the exact relationship between module wrappers
and functional operations.

Module map
----------

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Feature set
     - Module API
     - Functional/reference pages
   * - Sparse convolution
     - :doc:`convolution`
     - :doc:`../ops/conv`, :doc:`../../reference/backend/convolution`
   * - Quantized sparse convolution
     - :doc:`quantized-convolution`
     - :doc:`../core/quantized-weights`,
       :doc:`../../reference/backend/quantization`
   * - Sparse feature modules
     - :doc:`feature`
     - :doc:`../ops/feature`,
       :doc:`../../reference/concepts/sparse-tensor`
   * - Quantized sparse feature modules
     - :doc:`quantized-feature`
     - :doc:`../ops/feature`, :doc:`../core/quantized-weights`
   * - Sparse pooling
     - :doc:`pooling`
     - :doc:`../ops/pool`, :doc:`../../reference/backend/pooling`

Coordinate behavior
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Module family
     - Coordinate effect
     - Output type
   * - ``Conv3d``
     - Generates forward or explicit target support.
     - ``SparseTensor``
   * - ``SubmConv3d``
     - Preserves input coordinate identity.
     - ``SparseTensor``
   * - Transpose convolution modules
     - Generate expanded sparse support.
     - ``SparseTensor``
   * - Feature modules
     - Preserve coordinate identity.
     - ``SparseTensor``
   * - Local pooling modules
     - Generate pooled support from a kernel relation.
     - ``SparseTensor``
   * - Global pooling modules
     - Reduce by ``batch_counts``.
     - Dense MLX array

.. toctree::
   :maxdepth: 2

   convolution
   quantized-convolution
   feature
   quantized-feature
   pooling
