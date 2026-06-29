End-to-end sparse workflow
==========================

This page describes the lifecycle of data in a typical ``mlx-lattice`` model.
It focuses on how user-level operations connect to coordinate identity,
relations, backend dispatch, and module composition.

Workflow cross-reference
------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Workflow concern
     - Concept/backend page
     - API page
   * - Sparse support and identity
     - :doc:`../reference/concepts/sparse-tensor`
     - :doc:`../api/core/sparse-tensor`
   * - Kernel relations
     - :doc:`../reference/concepts/coordinates-relations`
     - :doc:`../api/core/relations`
   * - Modules and feature blocks
     - :doc:`quickstart`
     - :doc:`../api/nn/index`
   * - Branch joins
     - :doc:`../reference/concepts/algebra`
     - :doc:`../api/ops/tensor`
   * - Backend route diagnosis
     - :doc:`../reference/backend/path-selection`
     - :doc:`../api/native`

Stage 1: choose the sparse support
----------------------------------

An active coordinate has one meaning throughout a pipeline. Common
choices are:

.. list-table::
   :header-rows: 1
   :widths: 26 38 36

   * - Support type
     - Active coordinate means
     - Typical source
   * - Voxelized point cloud
     - At least one point fell into the voxel
     - ``voxelize`` or ``sparse_quantize``.
   * - Submanifold network
     - Site remains active through feature transforms
     - ``subm_conv3d`` blocks.
   * - Strided encoder
     - Downsampled support from a previous level
     - ``conv3d`` or pooling with stride.
   * - Sparse decoder
     - Generated/expanded site
     - Transposed or generative convolution.
   * - Algebra output
     - Coordinate selected by join policy
     - ``sparse_add``, ``sparse_mul``, ``cat``.

The library can align coordinates, but it cannot infer that two branches mean
the same physical quantity. Make the support semantics explicit in model code.

Stage 2: build tensors with metadata
------------------------------------

Construct ``SparseTensor`` with coordinates, features, stride, and
``batch_counts`` when global pooling or decomposition is needed:

.. code-block:: python

   x = SparseTensor(coords, feats, stride=1, batch_counts=(len(coords),))

For batched data, prefer ``sparse_collate``:

.. code-block:: python

   from mlx_lattice.ops import sparse_collate

   batch = sparse_collate([coords0, coords1], [feats0, feats1])

``sparse_collate`` records batch counts and prepends the batch column. This
avoids a common bug where feature rows are concatenated but batch ownership is
lost.

Stage 3: compose feature-preserving blocks
------------------------------------------

Feature-only operations preserve coordinate identity. This includes
activations, normalization, linear layers, and most ``mlx_lattice.nn`` feature
modules:

.. code-block:: python

   from mlx_lattice import nn

   block = [
       nn.Linear(32, 64),
       nn.BatchNorm(64),
       nn.SiLU(),
       nn.LayerNorm(64),
   ]

   h = x
   for op in block:
       h = op(h)

These modules call functional operations that replace ``feats`` but keep the
same coordinate key when row count and stride are unchanged.

Stage 4: insert relation-changing blocks
----------------------------------------

Convolution and pooling build kernel relations. A relation contains the sparse
edge set used by the backend:

.. math::

   \mathcal{E} = \{(i,o,k)\}.

The relation belongs to the coordinate manager and is cached by coordinate key,
target key, relation kind, and kernel specification. Model code does not need
to manually cache edge arrays unless it is implementing a custom operator.

.. code-block:: python

   encoder = nn.Conv3d(32, 64, kernel_size=3, stride=2)
   subm = nn.SubmConv3d(64, 64, kernel_size=3)
   pooled = nn.MaxPool3d(kernel_size=2, stride=2)

   h = encoder(x)
   h = subm(h)
   h = pooled(h.astype(mx.float32))

Stage 5: align branches before combining
----------------------------------------

Residual and skip connections are safe only when support is known. If both
branches preserve identity, direct sparse addition uses the identity fast path.
If supports may differ, choose a join:

.. code-block:: python

   from mlx_lattice.ops import sparse_add

   skip = sparse_add(decoder_branch, encoder_branch, join='inner')

``inner`` is common for strict skip connections. ``outer`` is common for
occupancy-style merging. ``left`` and ``right`` are useful when one branch owns
the desired output support.

Stage 6: bridge point and voxel domains
---------------------------------------

Point/voxel maps are geometry metadata. The same points,
voxel size, origin, batch indices, and interpolation mode must be used when
reusing a map.

.. code-block:: python

   voxels = voxelize(points, point_feats, voxel_size=0.05)
   voxel_feats = model(voxels)
   point_feats = devoxelize(points, voxel_feats, voxel_size=0.05)

For point heads that need both raw point features and voxel features, keep the
point rows as the dense domain and use devoxelization as the bridge back from
the sparse domain.

Stage 7: quantize after the floating contract is correct
--------------------------------------------------------

Quantized modules are drop-in inference modules for supported layers. A typical
workflow is:

1. build and validate the floating module;
2. construct the quantized module from the floating module;
3. compare against the dequantized/floating contract on representative sparse
   supports;
4. benchmark the public operation on the target backend.

Quantized convolution keeps the same sparse relation semantics. The storage and
execution route change; the coordinate support does not.

Stage 8: benchmark by public shape
----------------------------------

Record:

* active rows and pattern;
* coordinate dtype and feature dtype;
* channel count;
* kernel volume and relation kind;
* packed quantized bit width and group size;
* backend device;
* batch metadata when global pooling is involved.

Internal kernel names are useful for maintainers during diagnosis, but public
benchmark reports are reproducible from public inputs.
