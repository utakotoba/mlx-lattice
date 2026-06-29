Backend path selection
======================

Path selection starts from the MLX device and the public operation. The Python
layer builds semantic objects first: sparse tensors, coordinate keys, kernel
relations, neighbor relations, point/voxel maps, or packed quantized weights.
The native layer then evaluates those objects on CPU or Metal.

Decision order
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Layer
     - Predicate source
     - Effect
   * - Device
     - ``mx.default_device()``
     - Selects CPU primitive evaluation or Metal primitive evaluation.
   * - Operation family
     - Public function/module call
     - Chooses convolution, pooling, coordinates, point/voxel, entropy, or
       feature execution.
   * - Semantic relation
     - ``KernelRelation`` / ``NeighborRelation`` / point-voxel metadata
     - Defines row connectivity before backend kernels are considered.
   * - Storage
     - Dense floating arrays or ``QuantizedWeight``
     - Selects floating or packed int4/int8 convolution/linear execution.
   * - Shape and dtype
     - Channel count, kernel volume, layout, feature dtype, coordinate dtype
     - Selects a specialized route, a generic route, or a validation error.
   * - Metal capability
     - TensorOps capability tier
     - Allows TensorOps routes on devices reporting neural acceleration.

The public operation owns semantics. A backend route can change how the sum or
reduction is evaluated, but it must evaluate the same relation contract.

Capability tiers
----------------

Metal TensorOps capability is computed by the native capability helper:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Tier
     - Native predicate
     - Routes enabled
   * - ``unavailable``
     - macOS availability/device family check fails
     - Classic CPU/Metal routes only.
   * - ``gpu``
     - Apple GPU family supports the baseline tensor API but not neural
       acceleration tier
     - Some sorted quantized implicit-GEMM routes may be considered; neural
       accelerator-only TensorOps routes are not preferred.
   * - ``neural_accelerator``
     - Device reports the neural-acceleration family
     - TensorOps sorted fp16 convolution and TensorOps quantized contraction can
       be selected when shape predicates also match.

The capability check is a route predicate, not a user-visible device API. Use
``backend_info()`` for compiled backend availability; use benchmarks to observe
selected performance behavior for a public input.

Route predicates by operation
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 42 38

   * - Operation
     - Fast-path predicates
     - Fallback or alternate route
   * - ``conv3d`` pointwise
     - ``KernelSpec.is_pointwise`` and no explicit target coordinates
     - Feature matrix multiplication; quantized weights use quantized matmul.
   * - ``conv3d`` relation
     - Non-pointwise or explicit target support
     - Builds ``forward`` or ``target`` kernel relation and dispatches sparse
       convolution primitive.
   * - ``subm_conv3d``
     - Odd kernel, stride 1, no coordinate expansion
     - Reuses input coordinate identity and sparse relation.
   * - Transposed convolution
     - ``conv_transpose3d`` or ``generative_conv_transpose3d``
     - Builds transposed or generative relation; sorted implicit-GEMM is not
       selected because the relation kind is not ``forward``/``target``.
   * - Local pooling
     - ``float32`` features, kernel relation
     - Metal/CPU sparse reduction kernels.
   * - Global pooling
     - ``batch_counts`` metadata
     - MLX scatter/reduction over batch row groups.
   * - Point/voxel utilities
     - ``float32`` point geometry and supported map/reduction type
     - Native coordinate kernels for quantization, maps, and interpolation.
   * - Sparse algebra
     - Shared coordinate identity or explicit join policy
     - Native alignment helpers plus MLX feature arithmetic.

Convolution route equations
---------------------------

Relation convolution evaluates:

.. math::

   Y_{o,c_o} =
   \sum_{e \in \mathcal{E}: e_o=o}
   X_{e_i,c_i} W_{e_k,c_i,c_o}.

Pooling evaluates:

.. math::

   Y_{o,c} =
   \operatorname{reduce}_{e \in \mathcal{E}: e_o=o}
   X_{e_i,c},

where ``reduce`` is sum, max, or average. Average pooling divides by the sparse
edge count for the output row.

Validation boundaries
---------------------

The route selector validates public contracts before native execution:

* Metal sparse convolution and pooling require ``int32`` coordinates.
* Floating convolution supports ``float16`` and ``float32`` feature/weight
  matrices on native routes.
* Local pooling currently accepts ``float32`` features.
* Packed quantized convolution requires int4 or int8 packed ``uint32`` weights,
  scale/bias arrays matching feature dtype, and group size in ``{32, 64, 128}``.
* Sorted fp16 implicit-GEMM routes require relation metadata that can produce
  the sorted implicit-GEMM view.

When a specialized route predicate fails, execution uses the more general route
for the same public operation if the public input is otherwise valid.

Related API and references
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Need
     - Page
     - Why it matters
   * - Inspect native metadata
     - :doc:`../../api/native`
     - Confirms compiled backend and capability information.
   * - Sparse tensor inputs
     - :doc:`../../api/core/sparse-tensor`
     - Documents coordinate dtype, active rows, stride, and batch metadata.
   * - Relation metadata
     - :doc:`../../api/core/relations`
     - Documents relation counts, CSR views, and implicit-GEMM views.
   * - Convolution routes
     - :doc:`convolution`
     - Expands route predicates for forward, backward, and quantized paths.
   * - Quantized storage
     - :doc:`quantization`
     - Explains packed weight metadata used by route selection.
