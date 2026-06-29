Coordinate-aligned sparse algebra
=================================

Sparse algebra combines tensors by coordinate values. The row index is an
ordering detail; the coordinate row is the key.

Join model
----------

Let :math:`A` and :math:`B` be coordinate sets. The supported joins are:

.. math::

   A_{\text{inner}} = A \cap B,\qquad
   A_{\text{outer}} = A \cup B,

.. math::

   A_{\text{left}} = A,\qquad
   A_{\text{right}} = B.

The alignment result stores output coordinates plus gather rows for both
operands. Missing rows are encoded as ``-1`` and filled by the operation.

Operation defaults
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Operation
     - Default join
     - Reason
   * - ``sparse_add``
     - ``outer``
     - Preserve contributions from either side.
   * - ``sparse_sub``
     - ``outer``
     - Preserve left-only and right-only support with fill values.
   * - ``sparse_mul``
     - ``inner``
     - Multiplication is usually meaningful only where both sides exist.
   * - ``sparse_maximum`` / ``sparse_minimum``
     - ``inner``
     - Avoid introducing filled comparison values by default.
   * - ``cat``
     - ``inner``
     - Concatenate feature channels on shared support unless a join is supplied.

Identity fast path
------------------

If two tensors share coordinate identity, algebra can operate directly on
features:

.. math::

   Z_i = f(X_i, Y_i).

If identity differs, alignment builds gather maps:

.. math::

   Z_j = f(X_{\ell_j}, Y_{r_j}),

where :math:`\ell_j` and :math:`r_j` may be missing-row sentinels.

Row-changing utilities
----------------------

``prune``, ``prune_mask``, ``crop``, and ``topk_rows`` produce row subsets.
They must transform coordinates, features, active-row metadata, and coordinate
identity together. ``replace_feature`` is the feature-only counterpart and
preserves coordinate identity when row count is unchanged.

Collation
---------

``sparse_collate`` accepts unbatched ``(N,3)`` coordinate arrays, prepends a
batch column, concatenates features, and records ``batch_counts``. This is the
preferred entry point for building a batched sparse tensor from individual
samples.

Related API and references
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Page
     - Notes
   * - Sparse algebra functions
     - :doc:`../../api/ops/tensor`
     - Join modes, collation, row selection, crop, and concatenation.
   * - Coordinate alignment core
     - :doc:`../../api/core/coordinate-utilities`
     - ``SparseAlignment`` and lower-level alignment builder.
   * - Sparse tensor identity
     - :doc:`sparse-tensor`
     - Why identity fast paths are stronger than row equality.
   * - Model workflow
     - :doc:`../../getting-started/workflow`
     - Where joins belong in residual and skip connections.
