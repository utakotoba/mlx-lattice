Concepts
========

This section documents the semantic model shared by the Python layer, the CPU
backend, and the Metal backend. The goal is to make the codebase predictable:
features own the public semantics, and backends implement those semantics with
specialized kernels where useful.

.. toctree::
   :maxdepth: 2

   sparse-tensor
   coordinates-relations
   algebra

Design center
-------------

The library is feature-first from the user's perspective. "Sparse convolution,"
"point/voxel conversion," "coordinate-aligned algebra," and "quantized
inference" are features with backend implementations underneath them. This is
different from a backend-first API where users choose ``cpu_conv`` or
``metal_conv`` directly.

That design has two consequences:

* public operations define their result without reference to a specific kernel;
* backend routes are replaceable as long as they preserve the operation's sparse
  semantics.

This separation is especially important in Metal code, where high-performance
kernels may require specialized packing, tiling, and launch policies. Those
details belong below the semantic layer. The Python API and benchmark suite
describe the sparse operation, not the internal route.

Concept-to-API map
------------------

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Concept
     - API page
     - Backend page
   * - Sparse tensor identity
     - :doc:`../../api/core/sparse-tensor`
     - :doc:`../backend/path-selection`
   * - Coordinate manager and relations
     - :doc:`../../api/core/coordinate-management`
     - :doc:`../backend/convolution`
   * - Relation views
     - :doc:`../../api/core/relations`
     - :doc:`../backend/convolution`
   * - Coordinate-aligned algebra
     - :doc:`../../api/ops/tensor`
     - :doc:`../backend/path-selection`
