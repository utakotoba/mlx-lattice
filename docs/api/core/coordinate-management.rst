Coordinate management
=====================

Coordinate managers own coordinate arrays, active-row scalars, and relation
caches. A ``CoordinateMapKey`` is only valid inside the manager that created it.
This lets operations compare coordinate identity without comparing every row.

The cache key for a kernel relation includes:

* input coordinate key;
* optional explicit target coordinate key;
* normalized kernel geometry;
* relation kind, such as ``forward`` or ``target``.

This page documents the canonical manager API. Higher-level users usually
construct managers indirectly by creating :class:`mlx_lattice.SparseTensor`
objects.

Related pages
-------------

* Sparse tensor identity: :doc:`../../reference/concepts/sparse-tensor`
* Relation views: :doc:`../../reference/concepts/coordinates-relations`
* Relation classes: :doc:`relations`
* Functional relation builders: :doc:`../ops/relations`

.. automodule:: mlx_lattice.core.coords.manager
   :members:
