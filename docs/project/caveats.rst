Caveats and stability notes
===========================

This page records user-visible constraints and stability boundaries.

Coordinate value equality is not identity
-----------------------------------------

Two tensors can contain equal coordinate rows but have different
``CoordinateMapKey`` values. Cached relations are keyed by coordinate identity,
not by a late comparison of array contents. Use sparse alignment when combining
independently constructed tensors.

Active rows and capacity differ
-------------------------------

Native coordinate builders may allocate a buffer larger than the active set.
Use ``active_rows`` to determine how many rows are valid. Treating every
allocated row as active can include uninitialized or padded coordinate rows in
later operations.

Dtype boundaries
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Surface
     - Supported dtype boundary
     - Notes
   * - Metal coordinates
     - ``int32``
     - Sparse convolution and pooling validate this before launch.
   * - Floating convolution
     - ``float16`` or ``float32`` features/weights
     - Specialized Metal routes depend on dtype and channel count.
   * - Local pooling
     - ``float32`` features
     - Sum, max, and average local pooling share this boundary.
   * - Point/voxel
     - ``float32`` points/features, ``int32`` maps
     - Native quantization and interpolation routes use this contract.
   * - Quantized weights
     - int4/int8 packed in ``uint32``
     - Scales/biases match feature dtype at execution.

Global pooling requires batch metadata
--------------------------------------

Global pooling reduces to a dense ``(B, C)`` array. It requires
``batch_counts`` because a sparse coordinate buffer alone does not encode empty
batches. Sum and average define empty-batch behavior; max rejects empty batches.

Internal routes are not public APIs
-----------------------------------

Kernel names, CSR view names, sorted implicit-GEMM views, TensorOps variants,
and diagnostic reference routes are backend implementation details. They can be
useful for debugging a failing run, but application code calls public
operations and modules.

Quantization is storage-real, not fake quantization
---------------------------------------------------

``QuantizedWeight`` stores packed int4/int8 data plus affine metadata. Supported
native routes consume packed storage. If you want the floating contract, call
``dequantize_weight`` explicitly and use the floating operation.

Sparse performance depends on geometry
--------------------------------------

Sparse runtime is not determined by active row count alone. Important factors
include edge count, kernel volume, channel count, coordinate pattern, sorted
view availability, and whether the operation is relation-bound or
arithmetic-bound. Report those fields with benchmark results.

Point/voxel maps are geometry-specific
--------------------------------------

A point-to-voxel map is tied to points, batch indices, voxel size, origin,
voxel coordinates, and interpolation mode. Reusing a map after changing any of
those inputs is a semantic error.
