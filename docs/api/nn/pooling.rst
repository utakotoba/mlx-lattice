Pooling modules
===============

Pooling modules wrap local relation pooling and batch-wise global pooling.
Local pooling returns sparse tensors; global pooling returns dense batch rows.

Related pages
-------------

* Functional pooling API: :doc:`../ops/pool`
* Pooling backend routes: :doc:`../../reference/backend/pooling`
* Relation model: :doc:`../../reference/concepts/coordinates-relations`
* Batch metadata: :doc:`../../reference/concepts/sparse-tensor`

Module summary
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Module
     - Reduction
     - Output type
   * - ``Pool3d``
     - Configurable ``sum``, ``max``, or ``avg``.
     - ``SparseTensor``
   * - ``SumPool3d`` / ``MaxPool3d`` / ``AvgPool3d``
     - Fixed local reduction mode.
     - ``SparseTensor``
   * - ``GlobalSumPool`` / ``GlobalAvgPool`` / ``GlobalMaxPool``
     - Batch-wise dense reduction.
     - MLX array

.. automodule:: mlx_lattice.nn.pool
   :members:
