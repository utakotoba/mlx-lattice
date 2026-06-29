Convolution operations
======================

Sparse convolution functions operate on :class:`mlx_lattice.SparseTensor`
objects and dense or packed weights. They build or reuse kernel relations and
return a new sparse tensor whose coordinate support is determined by the
operation:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Function
     - Relation kind
     - Output coordinates
   * - ``conv3d``
     - ``forward`` or ``target``
     - Generated from input geometry, or taken from explicit target
       coordinates.
   * - ``subm_conv3d``
     - ``forward``
     - Reuses input coordinate identity.
   * - ``conv_transpose3d``
     - ``transposed``
     - Uses the transpose-convolution relation support.
   * - ``generative_conv_transpose3d``
     - ``generative``
     - Generates support from input rows and stride.

Floating weights accept dense 5D layout ``(C_out, Kx, Ky, Kz, C_in)`` and
mapped kernel-major layout ``(K, C_in, C_out)``. Packed quantized weights use
``QuantizedWeight``.

Related pages
-------------

* Route predicates and backend details: :doc:`../../reference/backend/convolution`
* Relation model: :doc:`../../reference/concepts/coordinates-relations`
* Sparse tensor output metadata: :doc:`../../reference/concepts/sparse-tensor`
* Module wrappers: :doc:`../nn/convolution`
* Quantized weight storage: :doc:`../core/quantized-weights`

.. automodule:: mlx_lattice.ops.conv
   :members:
