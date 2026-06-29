Quantization routes
===================

Quantized inference uses packed affine weights. The storage object is
:class:`mlx_lattice.core.QuantizedWeight`; operations select quantized execution
when that object is passed as the weight.

Packed affine layout
--------------------

Each group stores unsigned integer values plus an affine scale and bias. The
logical value used by the kernel is:

.. math::

   w_{g,j} = s_g q_{g,j} + b_g,

where :math:`q` is the packed int4/int8 code, :math:`s_g` is the group scale,
and :math:`b_g` is the group bias.

Supported metadata:

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Field
     - Supported values
     - Validation
   * - ``bits``
     - ``4`` or ``8``
     - Other bit widths are rejected.
   * - ``group_size``
     - ``32``, ``64``, or ``128``
     - Storage channels must be divisible by group size.
   * - Packed dtype
     - ``uint32``
     - Weight tensor is 3D packed storage.
   * - Scale/bias dtype
     - ``float16`` or ``float32``
     - Must match each other and feature dtype at execution.
   * - Layout
     - ``linear``, ``kernel_major``, ``dense_5d``
     - Determines how logical weights are reconstructed.

Linear route
------------

``mlx_lattice.ops.linear`` detects ``QuantizedWeight`` and uses quantized MLX
matmul for sparse features. The coordinate set is preserved because linear is a
feature-only operation.

Convolution route matrix
------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - Route
     - Predicate
     - Output order
   * - Direct packed relation conv
     - Valid packed metadata, feature dtype ``float16`` or ``float32``
     - Public output order.
   * - Sorted quantized implicit-GEMM
     - fp16 features, sorted plan present, ``K=27``, ``C_in,C_out`` in
       ``{32,64}``, storage channels equal logical channels, group size no
       larger than ``C_in``, TensorOps capability not unavailable
     - Computes sorted temporary then reorders.
   * - TensorOps quantized contraction
     - fp16 features, sorted plan, ``K=27``, ``C_in,C_out`` in ``{32,64}``,
       storage channels equal logical channels, ``group_size == C_in``,
       neural acceleration
     - Public output order.

The direct packed Metal kernels dispatch by feature dtype and bit width:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature dtype
     - int4 kernel
     - int8 kernel
   * - ``float16``
     - fp16 × int4
     - fp16 × int8
   * - ``float32``
     - fp32 × int4
     - fp32 × int8

Performance interpretation
--------------------------

Quantization reduces weight storage, but sparse convolution cost is:

.. math::

   T \approx T_{\text{relation}} + T_{\text{gather}} +
   T_{\text{dequant}} + T_{\text{multiply-accumulate}}.

If relation traversal dominates, packed weights may not improve runtime. The
largest quantization benefit appears when the operation has enough channel work
or matrix-like structure for weight bandwidth/arithmetic to matter.

Numerical comparison
--------------------

Packed affine quantization is approximate. Compare quantized convolution to the
dequantized dense contract when validating correctness:

.. math::

   \hat{Y} = \operatorname{Conv}(X, \operatorname{dequantize}(Q)).

The tests use this contract for pointwise, generic relation, submanifold,
target, transposed, generative, and sorted implicit-GEMM quantized routes.

Related API and references
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Page
     - Notes
   * - Packed weight container
     - :doc:`../../api/core/quantized-weights`
     - ``QuantizedWeight``, ``quantize_weight``, and ``dequantize_weight``.
   * - Quantized modules
     - :doc:`../../api/nn/quantized-convolution`
     - Quantized convolution and linear wrappers.
   * - Convolution route matrix
     - :doc:`convolution`
     - How quantized weights interact with sparse relation convolution.
   * - Feature linear route
     - :doc:`../../api/ops/feature`
     - Quantized sparse-feature linear projection.
