Core API
========

The core API contains sparse tensor containers, coordinate-management
primitives, relation data structures, quantized weight containers, and small
shared type helpers. Pages in this section document canonical implementation
modules rather than convenience re-export modules. That keeps duplicated entries
out of the API reference while preserving the normal public import surface.

Core objects are the metadata layer shared by CPU and Metal execution. They do
not select kernels by themselves; they describe coordinates, active capacity,
relation edges, relation views, and packed weight storage in a backend-neutral
form.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Page
     - Main objects
     - Contract
   * - :doc:`sparse-tensor`
     - ``SparseTensor``
     - Coordinates are ``(N, 4)`` integer rows; features are ``(N, C)`` MLX
       arrays with matching row order.
   * - :doc:`coordinate-management`
     - ``CoordinateManager``, ``CoordinateMapKey``
     - Coordinate identity and cached sparse relations are keyed by manager
       ownership, stride, and active-row metadata.
   * - :doc:`relations`
     - ``KernelSpec``, ``KernelRelation``, ``NeighborRelation``
     - Relation objects store semantic edge arrays plus execution views used by
       backend kernels.
   * - :doc:`quantized-weights`
     - ``QuantizedWeight``
     - Packed affine int4/int8 weights keep logical shape, layout, group size,
       scales, and biases together.

.. toctree::
   :maxdepth: 2

   top-level
   sparse-tensor
   coordinate-management
   coordinate-utilities
   point-voxel
   relations
   quantized-weights
   types
