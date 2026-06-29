Relation operations
===================

Relation operations expose the same relation builders used internally by
convolution, pooling, and neighbor queries. They are useful when an application
needs to inspect sparse connectivity, reuse relation metadata, or gather
neighbor features explicitly.

Kernel relations are lattice relations over integer coordinates. Neighbor
relations are geometric query/source relations over sparse rows and carry
distances in edge order.

Related pages
-------------

* Concept reference: :doc:`../../reference/concepts/coordinates-relations`
* Core relation classes: :doc:`../core/relations`
* Coordinate manager cache API: :doc:`../core/coordinate-management`
* Convolution route consumers: :doc:`../../reference/backend/convolution`

.. automodule:: mlx_lattice.ops.relations
   :members:
