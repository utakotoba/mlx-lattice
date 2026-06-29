Sparse tensor model
===================

``SparseTensor`` is the value object that binds sparse coordinates and dense row
features. It also carries coordinate identity, stride metadata, batch metadata,
and an active-row scalar.

Core invariant
--------------

The primary invariant is row alignment:

.. math::

   \operatorname{coords}\in\mathbb{Z}^{N\times4},
   \quad
   \operatorname{feats}\in\mathbb{R}^{N\times C},
   \quad
   \operatorname{coords}_i \leftrightarrow \operatorname{feats}_i.

The coordinate columns are ``(batch, x, y, z)``. The batch column is part of the
coordinate value, so ``(0, 1, 2, 3)`` and ``(1, 1, 2, 3)`` are different sparse
sites.

Capacity and active rows
------------------------

Native coordinate builders often allocate a capacity and store a scalar active
row count. ``SparseTensor.capacity`` is the static buffer length. ``active_rows``
is an ``int32`` scalar MLX array containing the number of valid rows.

This distinction matters for kernels that produce an upper-bound buffer and a
lazy count. Operators pass both the coordinate buffer and active-row scalar
rather than assuming every allocated row is active.

Stride metadata
---------------

``stride`` describes the lattice scale of the coordinate set. If the input
stride is :math:`s` and a forward convolution or pooling uses stride
:math:`r`, the output stride is:

.. math::

   s_{out} = s \odot r.

Transposed and generative transpose convolution divide by their stride:

.. math::

   s_{out} = s \oslash r.

The code validates exact divisibility for transposed stride updates.

Coordinate identity
-------------------

``CoordinateManager`` assigns a ``CoordinateMapKey`` to a coordinate array,
stride, and active-row scalar. Identity is stronger than value equality:

.. math::

   \text{same identity} \Rightarrow \text{same coordinate values},
   \qquad
   \text{same coordinate values} \nRightarrow \text{same identity}.

Identity lets relations be cached by key. If two tensors were built from
separate arrays, use sparse alignment before assuming rows match.

Batch metadata
--------------

``batch_counts`` stores the number of rows for each batch item. It is used by:

* ``batch_rows``;
* ``decomposed_coordinates``;
* ``decomposed_features``;
* global pooling.

Local convolution and pooling use the batch column directly in coordinate
relations. Global pooling needs ``batch_counts`` because it reduces to a dense
``(B, C)`` array.

Operation effects
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Operation family
     - Coordinate effect
     - Feature effect
   * - Activations/norm/linear
     - Preserve coordinate identity
     - Replace feature matrix.
   * - Submanifold convolution
     - Preserve input support
     - Relation convolution on same key.
   * - Forward convolution/pooling
     - Build output support from kernel geometry
     - Relation accumulation or reduction.
   * - Transposed/generative convolution
     - Expand or generate support
     - Relation accumulation.
   * - Prune/crop/top-k
     - Select row subset
     - Select matching feature rows.
   * - Sparse algebra
     - Preserve identity or build value-aligned support
     - Elementwise/combine feature rows.
   * - Point/voxel conversion
     - Convert continuous points to integer voxel coordinates
     - Aggregate or interpolate features.

Manual array edits
------------------

If coordinates or features are edited manually, preserve all linked fields:
coordinate rows, feature rows, active-row scalar, stride, coordinate key, and
batch counts. For most model code, prefer library operations that update these
fields together.

Related API and references
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Page
     - Notes
   * - Construct and inspect tensors
     - :doc:`../../api/core/sparse-tensor`
     - Public ``SparseTensor`` properties and replacement helpers.
   * - Coordinate ownership
     - :doc:`../../api/core/coordinate-management`
     - Manager/key semantics and relation cache entry points.
   * - Feature-only transforms
     - :doc:`../../api/ops/feature`
     - Operations that preserve coordinate identity.
   * - Row-changing transforms
     - :doc:`../../api/ops/tensor`
     - Prune, crop, collation, and sparse joins.
   * - Backend dispatch
     - :doc:`../backend/path-selection`
     - How tensor metadata participates in route selection.
