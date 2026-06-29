Pooling operations
==================

Pooling operations reduce sparse feature rows through a kernel relation or
through batch metadata.

Local pooling computes:

.. math::

   y_{o,c} = \operatorname{reduce}_{e:o_e=o} x_{i_e,c}.

``sum`` and ``avg`` accept empty output rows as zero-valued reductions. ``max``
requires at least one contributing row for every output row. Global pooling
uses ``batch_counts`` metadata from the input sparse tensor and returns a dense
``(B, C)`` MLX array.

Related pages
-------------

* Backend reduction routes: :doc:`../../reference/backend/pooling`
* Relation model: :doc:`../../reference/concepts/coordinates-relations`
* Batch metadata: :doc:`../../reference/concepts/sparse-tensor`
* Module wrappers: :doc:`../nn/pooling`

.. automodule:: mlx_lattice.ops.pool
   :members:
