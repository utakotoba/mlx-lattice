Quantized convolution modules
=============================

Quantized convolution modules store packed affine int4/int8 weights and keep
floating-point sparse activations. Coordinate semantics match the floating
module with the same geometry.

Related pages
-------------

* Packed weight container: :doc:`../core/quantized-weights`
* Quantized backend routes: :doc:`../../reference/backend/quantization`
* Floating convolution modules: :doc:`convolution`
* Functional convolution API: :doc:`../ops/conv`

Module summary
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Module
     - Floating source
     - Coordinate support
   * - ``QuantizedConv3d``
     - ``Conv3d``
     - Forward or explicit target support.
   * - ``QuantizedSubmConv3d``
     - ``SubmConv3d``
     - Input coordinate identity.
   * - ``QuantizedConvTranspose3d``
     - ``ConvTranspose3d``
     - Transposed relation support.
   * - ``QuantizedGenerativeConvTranspose3d``
     - ``GenerativeConvTranspose3d``
     - Generated transpose-convolution support.

.. automodule:: mlx_lattice.nn.quantized_conv
   :members:
