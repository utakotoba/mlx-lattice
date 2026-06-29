Pooling routes
==============

Pooling is relation reduction. Local pooling builds a ``KernelRelation`` and
reduces each output row's input neighbors. Global pooling ignores kernel
geometry and reduces rows by batch metadata.

Local pooling contract
----------------------

For a relation edge set :math:`\mathcal{E}`, local pooling computes:

.. math::

   Y_{o,c}^{sum} = \sum_{e_o=o} X_{e_i,c}

.. math::

   Y_{o,c}^{avg} =
   \frac{1}{|\{e : e_o=o\}|}
   \sum_{e_o=o} X_{e_i,c}

.. math::

   Y_{o,c}^{max} = \max_{e_o=o} X_{e_i,c}.

The denominator in average pooling is the sparse neighbor count for the output
row. It is not the dense kernel volume unless every dense kernel position is
active.

Backend routes
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Route
     - Predicate
     - Implementation
   * - CPU local pooling
     - Valid ``float32`` features and kernel relation
     - CPU relation reduction over edge arrays.
   * - Metal local pooling
     - Valid ``float32`` features, ``int32`` coordinates, Metal device
     - ``sparse_pool_relation_f32_i32`` over output rows and channels.
   * - Local pooling VJP
     - Differentiating through local pooling
     - Sum/avg use direct gradient scatter; max uses max-tie policy.
   * - Local pooling JVP
     - Forward-mode transform
     - ``sparse_pool_relation_jvp_f32_i32``.
   * - Global pooling
     - ``batch_counts`` metadata present
     - MLX dense reductions or scatter reductions over batch ids.

Input-exclusive gradient path
-----------------------------

The pooling backend carries an ``input_exclusive`` flag derived from kernel
geometry. When each input row contributes to at most one output row, the
gradient path can use an exclusive input-gradient kernel. Otherwise it uses the
sum/avg or max relation-gradient route.

Validation boundaries
---------------------

Local pooling currently validates:

* feature dtype is ``float32``;
* Metal coordinates are ``int32``;
* mode is ``sum``, ``max``, or ``avg``;
* relation metadata includes output coordinates, counts, kernel count, and
  output capacity.

Global pooling validates:

* ``batch_counts`` is present;
* empty batches are allowed for sum and average;
* empty batches are rejected for max pooling.

Global pooling formulas
-----------------------

For batch :math:`b` with row set :math:`R_b`:

.. math::

   G^{sum}_{b,c} = \sum_{i \in R_b} X_{i,c},
   \qquad
   G^{avg}_{b,c} =
   \frac{G^{sum}_{b,c}}{\max(|R_b|, 1)}.

``global_max_pool`` requires :math:`|R_b| > 0` for every batch because there is
no finite feature value that represents the maximum of an empty sparse set.

Related API and references
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Page
     - Notes
   * - Functional pooling signatures
     - :doc:`../../api/ops/pool`
     - Local and global pooling functions.
   * - Module wrappers
     - :doc:`../../api/nn/pooling`
     - ``Pool3d`` variants and global pooling modules.
   * - Relation model
     - :doc:`../concepts/coordinates-relations`
     - Output CSR view used by relation reduction.
   * - Batch metadata
     - :doc:`../concepts/sparse-tensor`
     - ``batch_counts`` requirement for global pooling.
