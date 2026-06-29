Point/voxel backend routes
==========================

Point/voxel utilities convert continuous point rows to sparse voxel tensors and
back. They are backed by native CPU and Metal primitives because the expensive
work is coordinate hashing, stable compaction, scatter/gather, and interpolation
map construction.

Quantization
------------

For point :math:`p=(p_x,p_y,p_z)`, origin :math:`o`, and voxel size
:math:`s`, voxel coordinates are:

.. math::

   v_a = \left\lfloor \frac{p_a - o_a}{s_a} \right\rfloor,
   \qquad a \in \{x,y,z\}.

The native quantization primitive returns:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Output
     - Meaning
   * - ``coords``
     - Unique batched voxel coordinates.
   * - ``active_rows``
     - Lazy scalar row count for the voxel coordinate buffer.
   * - ``inverse_rows``
     - Point-row to voxel-row map.
   * - ``voxel_counts``
     - Number of points assigned to each voxel.

Metal quantization uses an integer coordinate hash table, stable compaction,
and a final point-to-voxel map pass. CPU quantization uses the same semantic
contract with host data structures.

Voxel feature aggregation
-------------------------

``voxelize_features`` supports ``sum`` and ``mean`` reductions. For voxel row
:math:`v` and point set :math:`P_v`:

.. math::

   F^{sum}_{v,c} = \sum_{p\in P_v} f_{p,c},
   \qquad
   F^{mean}_{v,c} =
   \frac{1}{\max(|P_v|,1)}
   \sum_{p\in P_v} f_{p,c}.

The Metal route clears the output feature matrix and scatters point
contributions into voxel rows. The gradient route gathers the voxel cotangent
back to contributing point rows with the same reduction scale.

Point-to-voxel maps
-------------------

``build_point_voxel_map`` supports nearest and linear interpolation. The map
contains voxel row ids and interpolation weights for each point. Devoxelization
then computes:

.. math::

   f_p = \sum_{j=1}^{M} \alpha_{p,j} V_{r_{p,j}},

where :math:`r_{p,j}` is a voxel row id and :math:`\alpha_{p,j}` is its
interpolation weight.

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Operation
     - Native work
     - Dtype boundary
   * - ``sparse_quantize``
     - Quantize points, hash unique voxels, compact active rows
     - ``float32`` points, ``int32`` batch/active rows.
   * - ``voxelize``
     - Quantize + feature scatter
     - ``float32`` point features.
   * - ``build_point_voxel_map``
     - Hash voxel coordinates and lookup interpolation stencil
     - ``float32`` points, ``int32`` voxel coordinates.
   * - ``devoxelize``
     - Weighted gather from voxel features to point rows
     - ``float32`` voxel features.

Batch handling
--------------

Batch ids are part of the voxel coordinate key. Two points with the same
spatial voxel and different batch ids map to different voxel rows. Passing
``batch_indices`` is therefore required whenever a point array contains more
than one sample.

Backend thresholds
------------------

Coordinate set operations use a compact/scatter split on Metal. Large selected
sets use stable parallel compaction; small sets use a single compact kernel.
The threshold is defined in the native coordinate runtime detail and applies to
set-style operations such as downsample, union, intersection, and quantization
planning. This keeps public semantics deterministic while allowing the backend
to choose the cheaper compaction mechanism for the row count.

Related API and references
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Page
     - Notes
   * - High-level point/voxel operations
     - :doc:`../../api/ops/quantization`
     - ``voxelize``, ``voxelize_with_quantization``, and ``devoxelize``.
   * - Core metadata classes
     - :doc:`../../api/core/point-voxel`
     - ``SparseQuantization`` and ``PointVoxelMap``.
   * - Sparse tensor output contract
     - :doc:`../concepts/sparse-tensor`
     - How voxel coordinates become ``SparseTensor`` support.
   * - Coordinate utilities
     - :doc:`../../api/core/coordinate-utilities`
     - Set operations and occupancy helpers used near voxel workflows.
