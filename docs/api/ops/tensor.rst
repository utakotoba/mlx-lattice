Sparse tensor algebra
=====================

Sparse algebra functions combine tensors by coordinate identity or by
value-aligned joins. If two tensors share coordinate identity, operations use a
direct feature path. Otherwise the operation builds row maps:

.. math::

   L_j, R_j \in \{-1, 0, \ldots, N-1\},

where ``-1`` means the coordinate is missing from that side of the join.

Join modes follow database-style sparse support semantics: ``inner`` keeps
coordinates present on both sides, ``left`` and ``right`` preserve one side,
and ``outer`` keeps the union.

Related pages
-------------

* Algebra concept reference: :doc:`../../reference/concepts/algebra`
* Sparse tensor identity: :doc:`../../reference/concepts/sparse-tensor`
* Coordinate alignment core: :doc:`../core/coordinate-utilities`
* Workflow guidance for branch joins: :doc:`../../getting-started/workflow`

.. automodule:: mlx_lattice.ops.tensor
   :members:
