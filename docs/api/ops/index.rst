Operations API
==============

The operations API is split by ``mlx_lattice.ops`` module. This mirrors the
semantic grouping used by the runtime package and avoids one large page mixing
convolution, pooling, feature transforms, sparse algebra, relation builders,
coordinate utilities, point/voxel conversion, and entropy coding.

Operation map
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Operation family
     - API page
     - Reference page
   * - Convolution
     - :doc:`conv`
     - :doc:`../../reference/backend/convolution`
   * - Pooling
     - :doc:`pool`
     - :doc:`../../reference/backend/pooling`
   * - Feature transforms
     - :doc:`feature`
     - :doc:`../../reference/concepts/sparse-tensor`
   * - Sparse algebra
     - :doc:`tensor`
     - :doc:`../../reference/concepts/algebra`
   * - Relations and neighbors
     - :doc:`relations`
     - :doc:`../../reference/concepts/coordinates-relations`
   * - Point/voxel conversion
     - :doc:`quantization`
     - :doc:`../../reference/backend/point-voxel`
   * - Entropy coding
     - :doc:`entropy`
     - :doc:`../../project/caveats`

.. toctree::
   :maxdepth: 2

   conv
   pool
   feature
   tensor
   relations
   coords
   quantization
   entropy
   internals
