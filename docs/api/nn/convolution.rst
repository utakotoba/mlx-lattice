Convolution modules
===================

Convolution modules own dense floating weights and delegate execution to the
functional sparse convolution API. Use them for model composition when weights
are learned or stored as module parameters.

Related pages
-------------

* Functional convolution API: :doc:`../ops/conv`
* Convolution backend routes: :doc:`../../reference/backend/convolution`
* Relation model: :doc:`../../reference/concepts/coordinates-relations`
* Quantized variants: :doc:`quantized-convolution`

Module summary
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Module
     - Coordinate support
     - Functional equivalent
   * - ``Conv3d``
     - Forward support or explicit target support.
     - :func:`mlx_lattice.ops.conv3d`
   * - ``SubmConv3d``
     - Input coordinate identity.
     - :func:`mlx_lattice.ops.subm_conv3d`
   * - ``ConvTranspose3d``
     - Transposed relation support.
     - :func:`mlx_lattice.ops.conv_transpose3d`
   * - ``GenerativeConvTranspose3d``
     - Generated transpose-convolution support.
     - :func:`mlx_lattice.ops.generative_conv_transpose3d`

.. automodule:: mlx_lattice.nn.conv
   :members:
