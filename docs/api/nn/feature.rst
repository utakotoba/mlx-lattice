Feature modules
===============

Feature modules are sparse-aware wrappers around row-local or channel-local
``mlx.nn`` style operations. They replace ``x.feats`` and preserve sparse
coordinate identity.

Related pages
-------------

* Functional feature API: :doc:`../ops/feature`
* Sparse tensor identity: :doc:`../../reference/concepts/sparse-tensor`
* Quantized feature modules: :doc:`quantized-feature`
* Sparse workflow: :doc:`../../getting-started/workflow`

Module summary
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Module family
     - Examples
     - Coordinate effect
   * - Projection
     - ``Linear``
     - Preserves coordinate identity.
   * - Activations
     - ``ReLU``, ``GELU``, ``SiLU``, ``Sigmoid``, ``Tanh``
     - Preserves coordinate identity.
   * - Normalization
     - ``BatchNorm``, ``LayerNorm``, ``RMSNorm``
     - Preserves coordinate identity.
   * - Regularization
     - ``Dropout``
     - Preserves coordinate identity.

.. automodule:: mlx_lattice.nn.feature
   :members:
