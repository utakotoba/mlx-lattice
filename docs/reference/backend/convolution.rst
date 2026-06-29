Convolution routes
==================

Sparse convolution has the richest backend selection matrix in the project. The
public operations are:

* :func:`mlx_lattice.ops.conv3d`;
* :func:`mlx_lattice.ops.subm_conv3d`;
* :func:`mlx_lattice.ops.conv_transpose3d`;
* :func:`mlx_lattice.ops.generative_conv_transpose3d`.

All four routes reduce over relation edges. The difference is how the output
coordinate support is produced.

Semantic map kinds
------------------

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Map kind
     - Builder
     - Output support
   * - ``forward``
     - ``CoordinateManager.kernel_relation``
     - Produced from input coordinates, stride, padding, and dilation.
   * - ``target``
     - ``CoordinateManager.target_kernel_relation``
     - Explicit target coordinates supplied to ``conv3d``.
   * - ``transposed``
     - ``CoordinateManager.transposed_kernel_relation``
     - Expanded support for transpose convolution.
   * - ``generative``
     - ``CoordinateManager.generative_relation``
     - Generated support from a transpose-convolution rule.

Only ``forward`` and ``target`` map kinds are considered by the sorted
implicit-GEMM forward route. Transposed and generative convolutions use relation
traversal for their current public path.

Floating forward routes
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Route
     - Predicate
     - Notes
   * - Pointwise matmul
     - ``1x1x1`` kernel, stride 1, no padding/dilation, no explicit target
       support
     - Computes ``x.feats @ weight.T`` and preserves coordinates.
   * - Generic relation traversal
     - Any valid relation convolution not captured by a more specific route
     - Consumes edge arrays plus output/input/kernel CSR views.
   * - Dense-channel Metal kernels
     - 5D dense weight layout, ``C_in`` and ``C_out`` in ``{16, 32, 64}``,
       ``K >= 16``, output capacity at least ``4096``
     - Specialized forward kernels for common channel-aligned 3D convolutions.
   * - ``cout16`` Metal kernels
     - ``C_out == 16`` and either ``K >= 16`` with output capacity at least
       ``4096`` or output capacity at least ``50000``
     - Optimizes the small-output-channel case.
   * - ``vec4`` Metal kernel
     - fp32 features and ``C_out`` divisible by 4
     - Vectorized output-channel traversal.
   * - fp16 gather kernel
     - fp16 features
     - Uses gather-style traversal instead of fp32 atomic fallback.
   * - fp32 atomic kernel
     - fp32 features when no gather/vector route is selected
     - Accumulates by relation edge.

Sorted fp16 implicit-GEMM
-------------------------

The Python predicate for the sorted floating route is:

.. math::

   \operatorname{kind} \in \{\text{forward}, \text{target}\},
   \quad X,W \in \mathrm{fp16},
   \quad K=27,
   \quad C_{in}=C_{out}\in\{32,64\}.

For a 5D dense weight, the Python layer maps
``(C, 3, 3, 3, C)`` into a contiguous ``(27, C, C)`` tensor before dispatch.

The Metal runtime then chooses between:

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Route
     - Additional predicate
     - Kernel family
   * - TensorOps sorted contraction
     - Neural-accelerator capability, contiguous fp16 features/weights,
       sorted relation view
     - Row-stationary TensorOps kernels with 64-row tiles.
   * - Direct sorted reference route
     - Same shape/layout predicate but TensorOps is not preferred
     - Row-stationary direct Metal kernels for C32/C64.

The sorted view stores:

.. math::

   \texttt{sorted\_out\_in\_map} \in \mathbb{Z}^{N_{out}\times 27},
   \qquad
   \texttt{sorted\_kv\_out\_in\_map} \in \mathbb{Z}^{27\times N_{out}}.

``reorder_rows`` maps sorted output rows back to public output order, and
``tile_masks`` stores occupancy masks for 64-row tiles.

Quantized forward routes
------------------------

Packed quantized convolution is selected by passing ``QuantizedWeight``.
Supported bit widths are 4 and 8. Packed weights use ``uint32`` storage, affine
scales, affine biases, and group size ``32``, ``64``, or ``128``.

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Route
     - Predicate
     - Notes
   * - Direct packed convolution
     - Any valid quantized relation convolution
     - Metal kernels dispatch by feature dtype and bit width:
       fp16/fp32 × int4/int8.
   * - Sorted quantized implicit-GEMM
     - Sorted plan present, fp16 features, ``K=27``, ``C_in`` and ``C_out`` in
       ``{32,64}``, storage channels equal logical channels, group size no
       larger than ``C_in``, TensorOps tier not unavailable
     - Contracts in sorted order and reorders output rows.
   * - TensorOps quantized contraction
     - fp16 features, ``K=27``, ``C_in`` and ``C_out`` in ``{32,64}``, storage
       channels equal logical channels, ``group_size == C_in``, neural
       acceleration
     - Dequantizes a temporary fp16 weight tile and runs TensorOps contraction.

The direct packed route computes the affine reconstruction per group:

.. math::

   w_{q} = s_g\,q + b_g,
   \qquad
   y = \sum_g \sum_{c \in g} x_c\,w_{q,c}.

Autodiff routes
---------------

Floating sparse convolution defines JVP and VJP for features and weights.
Backward execution has its own Metal route selection:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Backward path
     - Predicate
     - Route
   * - Input gradient TensorOps
     - ``C_in = C_out = 16``, ``K >= 16``, mapped weight layout, input capacity
       at least ``32768``, neural acceleration, fp32 path
     - TensorOps input-gradient contraction.
   * - Input gradient dense-channel kernels
     - Dense 5D weight, input capacity at least ``4096``, supported channel
       pairs
     - Specialized dense-channel kernels; grouped dense route is selected at
       larger input capacities.
   * - Weight gradient TensorOps
     - ``C_in`` and ``C_out`` in ``{16,32,64}``, ``K >= 16``, input capacity at
       least ``32768``, neural acceleration
     - Partitioned TensorOps contraction followed by reduction.
   * - Weight gradient classic kernels
     - fp16, block4-compatible channels, ``C_out=16`` cases, dense square
       channel cases, or generic fallback
     - Classic Metal kernels selected by channel, kernel volume, edge count,
       and input capacity.

Quantized convolution is inference-oriented in the public surface. If a
training path requires gradients through packed weights, dequantize explicitly
and use the floating route.

Weight layouts
--------------

Floating convolution accepts:

* dense 5D layout ``(C_out, Kx, Ky, Kz, C_in)``;
* mapped kernel-major layout ``(K, C_in, C_out)`` for internal sorted routes.

The sorted floating route requires mapped ``(27, C, C)`` storage. The public
``conv3d`` call can still receive 5D dense weights; Python maps and caches the
contiguous internal view.

Validation checklist
--------------------

When diagnosing a convolution route, record:

* map kind: forward, target, transposed, or generative;
* feature dtype and coordinate dtype;
* dense versus quantized weight;
* kernel volume ``K``;
* ``C_in`` and ``C_out``;
* relation output capacity and edge count;
* Metal capability tier when running on GPU.

Related API and references
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - Page
     - Notes
   * - Functional convolution signatures
     - :doc:`../../api/ops/conv`
     - ``conv3d``, ``subm_conv3d``, transpose, and generative transpose.
   * - Module wrappers
     - :doc:`../../api/nn/convolution`
     - ``Conv3d``/``SubmConv3d`` and quantized module variants.
   * - Relation model
     - :doc:`../concepts/coordinates-relations`
     - Edge arrays, CSR views, implicit-GEMM maps, and sorted views.
   * - Sparse tensor metadata
     - :doc:`../concepts/sparse-tensor`
     - Coordinate identity and stride updates for output tensors.
   * - Packed weights
     - :doc:`../../api/core/quantized-weights`
     - ``QuantizedWeight`` layout and dequantized validation contract.
