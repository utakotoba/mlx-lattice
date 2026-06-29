Coordinates and relations
=========================

Sparse operators are expressed through relation edges. A relation separates
geometry from arithmetic: coordinate code determines which rows are connected;
backend kernels determine how features are accumulated or reduced.

Kernel offsets
--------------

A ``KernelSpec`` normalizes ``size``, ``stride``, ``padding``, and ``dilation``
to 3-tuples. Kernel volume is:

.. math::

   K = K_x K_y K_z.

For ``3x3x3``, ``K=27``. For pointwise ``1x1x1``, ``K=1`` and convolution can
often become a feature matrix multiplication.

Relation edge arrays
--------------------

``KernelRelation`` stores three equal-length arrays:

.. math::

   e = (i_e, o_e, k_e),

where ``i_e`` is an input row, ``o_e`` is an output row, and ``k_e`` is a kernel
offset id. The public edge arrays are useful for diagnostics; execution uses
additional views built from the same edges.

Execution views
---------------

.. list-table::
   :header-rows: 1
   :widths: 26 38 36

   * - View
     - Shape
     - Primary consumer
   * - Output CSR
     - ``row_offsets`` length ``N_out + 1``
     - Forward accumulation and pooling by output row.
   * - Input CSR
     - input-row offsets plus edge ids
     - Input-gradient traversal.
   * - Kernel CSR
     - kernel-id offsets plus edge ids
     - Weight-gradient traversal.
   * - Implicit-GEMM map
     - ``(N_out, K)`` input-row map plus row masks
     - Matrix-like convolution views.
   * - Sorted implicit-GEMM map
     - sorted ``(N_out, K)``, K-major transpose, row reorder, tile masks
     - Sorted Metal convolution routes.

Sorted view construction
------------------------

The sorted view starts from an implicit-GEMM map:

.. math::

   M \in \mathbb{Z}^{N_{out}\times K}.

Rows are sorted by their occupancy mask. The K-major view is:

.. math::

   M^{T}_{sorted} \in \mathbb{Z}^{K\times N_{out}}.

Tile masks summarize 64 sorted output rows in four 16-row words. The Metal
sorted routes use those masks to skip missing kernel positions without changing
the public output order.

Neighbor relations
------------------

``NeighborRelation`` is used for radius and k-nearest-neighbor queries. Its
edge tuple is:

.. math::

   e = (q_e, s_e, n_e),

where ``q_e`` is a query row, ``s_e`` is a source row, and ``n_e`` is the
neighbor rank/id. Distances are stored alongside the edge arrays.

Relation caching
----------------

``CoordinateManager`` caches kernel relations by:

* input coordinate key;
* optional target coordinate key;
* normalized kernel specification;
* relation kind.

This cache is valid because coordinate identity is explicit. Reusing equal
coordinate values without shared identity does not imply a cached relation can
be reused safely.

Related API and references
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Page
     - Notes
   * - Relation data structures
     - :doc:`../../api/core/relations`
     - ``KernelSpec``, ``KernelRelation``, CSR views, and neighbor relations.
   * - Coordinate manager methods
     - :doc:`../../api/core/coordinate-management`
     - Cached forward, target, transposed, and generative relation builders.
   * - Functional relation helpers
     - :doc:`../../api/ops/relations`
     - Operation-oriented wrappers and neighbor feature gathering.
   * - Convolution route use
     - :doc:`../backend/convolution`
     - Which relation views are consumed by convolution kernels.
   * - Pooling route use
     - :doc:`../backend/pooling`
     - How output CSR views drive sparse reduction.
