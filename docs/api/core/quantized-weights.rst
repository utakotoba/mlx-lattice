Quantized weights
=================

``QuantizedWeight`` stores packed affine int4/int8 weights for inference. The
object is a logical weight plus the metadata required to execute it without
guessing shape from storage:

.. math::

   w_{g,j} = s_g q_{g,j} + b_g.

Supported layouts map to the public operations:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Layout
     - Logical source shape
     - Used by
   * - ``linear``
     - ``(C_out, C_in)``
     - Sparse-feature linear projections.
   * - ``kernel_major``
     - ``(K, C_in, C_out)``
     - Relation convolution with mapped kernel rows.
   * - ``dense_5d``
     - ``(C_out, Kx, Ky, Kz, C_in)``
     - Public sparse convolution modules and functions.

Related pages
-------------

* Quantized backend routes: :doc:`../../reference/backend/quantization`
* Sparse convolution routes: :doc:`../../reference/backend/convolution`
* Quantized module wrappers: :doc:`../nn/quantized-convolution` and
  :doc:`../nn/quantized-feature`
* Feature linear API: :doc:`../ops/feature`

.. automodule:: mlx_lattice.core.quantized
   :members:
