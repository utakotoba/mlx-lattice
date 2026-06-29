Quickstart
==========

This walkthrough builds a small sparse pipeline using both functional
operations and ``mlx_lattice.nn`` modules. It covers the main user-facing
surface: sparse tensors, convolution, pooling, feature transforms, sparse
algebra, point/voxel conversion, quantized inference, relation builders, and
global reductions.

The examples are small enough to read in one pass. The same APIs are used for
large point clouds; only the coordinate count, channel count, and backend route
change.

Walkthrough map
---------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Section
     - Main API page
     - Deeper reference
   * - Sparse tensor construction
     - :doc:`../api/core/sparse-tensor`
     - :doc:`../reference/concepts/sparse-tensor`
   * - Functional convolution
     - :doc:`../api/ops/conv`
     - :doc:`../reference/backend/convolution`
   * - ``mlx_lattice.nn`` modules
     - :doc:`../api/nn/index`
     - :doc:`workflow`
   * - Pooling
     - :doc:`../api/ops/pool`
     - :doc:`../reference/backend/pooling`
   * - Feature transforms
     - :doc:`../api/ops/feature`
     - :doc:`../api/nn/index`
   * - Sparse algebra
     - :doc:`../api/ops/tensor`
     - :doc:`../reference/concepts/algebra`
   * - Point/voxel conversion
     - :doc:`../api/ops/quantization`
     - :doc:`../reference/backend/point-voxel`
   * - Quantized inference
     - :doc:`../api/core/quantized-weights`
     - :doc:`../reference/backend/quantization`

Create a sparse tensor
----------------------

Sparse coordinates are integer rows with shape ``(N, 4)`` and column order
``(batch, x, y, z)``. Features are dense row features with shape ``(N, C)``.
The row contract is:

.. math::

   \text{feature row } i \longleftrightarrow
   \text{coordinate row } i = (b_i, x_i, y_i, z_i).

.. code-block:: python

   import mlx.core as mx
   from mlx_lattice import SparseTensor

   coords = mx.array(
       [
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 1, 0],
           [0, 2, 1, 0],
       ],
       dtype=mx.int32,
   )
   feats = mx.ones((4, 16), dtype=mx.float16)

   x = SparseTensor(coords, feats, batch_counts=(4,))

``batch_counts`` is optional for local convolution and pooling, but it is
required by global pooling because global pooling must know which rows belong
to each batch item.

Functional convolution
----------------------

The functional convolution API accepts either dense floating weights or packed
``QuantizedWeight`` objects. Dense 3D convolution weights use shape
``(C_out, Kx, Ky, Kz, C_in)``.

.. code-block:: python

   from mlx_lattice.ops import conv3d, subm_conv3d

   weight = mx.ones((32, 3, 3, 3, 16), dtype=mx.float16)
   y = conv3d(x, weight, kernel_size=3)

   subm_weight = mx.ones((32, 3, 3, 3, 16), dtype=mx.float16)
   z = subm_conv3d(x, subm_weight, kernel_size=3)

``conv3d`` builds a forward sparse kernel relation and may create an output
coordinate set. ``subm_conv3d`` reuses the input coordinate identity and writes
new features on the same active support.

Mathematically, relation convolution is:

.. math::

   Y[o, c_o] =
   \sum_{(i,o,k)\in \mathcal{E}}
   X[i, c_i]\,W[k, c_i, c_o],

where :math:`\mathcal{E}` is the sparse edge set for the selected kernel
relation. The backend may evaluate this edge set with CSR traversal, direct
Metal kernels, or a sorted implicit-GEMM view; the public result is the same
relation sum.

Target coordinates
------------------

``conv3d`` can compute onto an explicit output support. Pass another
``SparseTensor``, a ``CoordinateMapKey``, or a coordinate array through
``coordinates``:

.. code-block:: python

   target = SparseTensor(coords, mx.zeros((4, 32), dtype=mx.float16))
   y_target = conv3d(x, weight, kernel_size=3, coordinates=target)

This builds a target relation from ``x`` to the target support. If the target
coordinate key is the same as the input key, the operation can preserve the
input coordinate identity.

``mlx_lattice.nn`` modules
--------------------------

The module API mirrors the functional API and composes with ``mlx.nn`` style
code. Modules accept and return ``SparseTensor`` where the operation is sparse,
or dense MLX arrays where the operation is global pooling.

.. code-block:: python

   from mlx_lattice import nn

   block = [
       nn.Conv3d(16, 32, kernel_size=3, bias=True),
       nn.BatchNorm(32),
       nn.ReLU(),
       nn.SubmConv3d(32, 32, kernel_size=3),
       nn.LayerNorm(32),
   ]

   h = x
   for layer in block:
       h = layer(h)

Module classes are thin semantic wrappers around the same operations documented
under :mod:`mlx_lattice.ops`. Use modules when you want parameter ownership and
model composition. Use operations when you already own the weights or are
building custom layers.

Pooling
-------

Local pooling uses a sparse kernel relation and reduces neighbor features.
``sum``, ``max``, and ``avg`` are supported. The current local pooling kernels
expect ``float32`` features, so cast if your convolution block uses fp16:

.. code-block:: python

   from mlx_lattice.ops import avg_pool3d, global_avg_pool, max_pool3d

   pooled = max_pool3d(h.astype(mx.float32), kernel_size=3, stride=2)
   pooled_avg = avg_pool3d(h.astype(mx.float32), kernel_size=3, stride=2)
   summary = global_avg_pool(pooled)

Average pooling divides by the sparse contribution count for each output row,
not by dense kernel volume. Global pooling returns one dense feature row per
batch.

Feature operations
------------------

Feature operations preserve coordinates and transform only ``feats``:

.. code-block:: python

   from mlx_lattice.ops import gelu, layer_norm, linear, relu, rms_norm

   h = relu(x)
   h = gelu(h, approximate='tanh')
   h = layer_norm(h, weight=mx.ones((16,)), bias=mx.zeros((16,)))
   h = rms_norm(h, weight=mx.ones((16,)))
   h = linear(h, mx.ones((32, 16), dtype=h.dtype))

Because these operations do not create or remove rows, they preserve coordinate
identity. This makes them safe inside residual blocks when both branches share
the same support.

Sparse tensor algebra
---------------------

Sparse algebra aligns by coordinate value when identity is not shared. The join
policy defines the output support:

.. list-table::
   :header-rows: 1
   :widths: 18 34 48

   * - Join
     - Output support
     - Typical use
   * - ``inner``
     - :math:`A \cap B`
     - Intersection residuals or comparisons.
   * - ``left``
     - :math:`A`
     - Preserve the left branch support.
   * - ``right``
     - :math:`B`
     - Preserve the right branch support.
   * - ``outer``
     - :math:`A \cup B`
     - Merge or accumulate two sparse supports.

.. code-block:: python

   from mlx_lattice.ops import sparse_add, sparse_cat_aligned

   residual = sparse_add(h, h, join='inner')
   merged = sparse_cat_aligned(h, residual, join='outer')

Avoid adding ``h.feats`` arrays directly unless you know the coordinate
identity and row ordering match.

Point/voxel conversion
----------------------

Point-cloud data usually starts in continuous coordinates. Voxelization maps
points into integer lattice coordinates and aggregates point features:

.. code-block:: python

   from mlx_lattice.ops import devoxelize, voxelize

   points = mx.array(
       [
           [0.05, 0.05, 0.05],
           [0.12, 0.08, 0.05],
           [1.10, 0.95, 0.80],
       ],
       dtype=mx.float32,
   )
   point_features = mx.ones((3, 8), dtype=mx.float32)

   voxels = voxelize(points, point_features, voxel_size=0.1, reduction='mean')
   point_features_again = devoxelize(points, voxels, voxel_size=0.1)

The coordinate transform is:

.. math::

   v_a = \left\lfloor \frac{p_a - o_a}{s_a} \right\rfloor,
   \qquad a \in \{x,y,z\},

where :math:`p` is the point position, :math:`o` is the origin, and
:math:`s` is the voxel size.

Quantized inference
-------------------

Quantized weights are packed storage objects. They are selected by passing
``QuantizedWeight`` to a supported operation or by using quantized modules.

.. code-block:: python

   from mlx_lattice import quantize_weight
   from mlx_lattice.nn import Conv3d, QuantizedConv3d, QuantizedLinear

   dense_weight = mx.random.normal((32, 16), dtype=mx.float16)
   packed_linear_weight = quantize_weight(
       dense_weight,
       bits=8,
       group_size=32,
   )

   floating_conv = Conv3d(16, 32, kernel_size=3)
   quantized_conv = QuantizedConv3d.from_conv(
       floating_conv,
       bits=4,
       group_size=32,
   )
   qy = quantized_conv(x)

   qlinear = QuantizedLinear(16, 32, bits=8, group_size=32)
   qh = qlinear(x)

Packed int4/int8 convolution supports both pointwise and relation convolution
paths. For a general sparse pattern the relation traversal can dominate runtime;
benchmark quantized and floating routes on the same sparse support before
assuming one is faster.

Relation builders and neighbor queries
--------------------------------------

Most users let convolution and pooling build relations automatically. Relation
builders are available when writing custom sparse operations:

.. code-block:: python

   from mlx_lattice.ops import (
       build_kernel_relation,
       build_radius_relation,
       gather_neighbor_features,
   )

   relation = build_kernel_relation(x.coords, kernel_size=3)
   neighbors = build_radius_relation(
       x.coords,
       x.coords,
       radius=2.0,
       max_neighbors=16,
   )
   gathered = gather_neighbor_features(x.feats, neighbors)

Kernel relations carry ``(in_row, out_row, kernel_id)`` edges. Neighbor
relations carry ``(query_row, source_row, neighbor_id)`` edges plus distances.
Both are sparse connectivity descriptions, but they serve different operation
families.

Recommended next reads
----------------------

Read :doc:`../reference/concepts/sparse-tensor` for coordinate identity and
batch metadata, :doc:`../reference/concepts/coordinates-relations` for relation
views, and :doc:`../reference/backend/convolution` for concrete convolution
route predicates. For signatures and public objects, use
:doc:`../api/core/index`, :doc:`../api/ops/index`, and :doc:`../api/nn/index`.

If a benchmark result is surprising, first identify the public input shape:
active rows, channel count, kernel volume, dtype, quantized layout, and device.
Then read :doc:`../reference/backend/path-selection` before inspecting native
kernel names.
