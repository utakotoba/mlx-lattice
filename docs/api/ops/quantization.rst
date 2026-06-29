Point/voxel operations
======================

These operation wrappers expose the point/voxel bridge as high-level sparse
tensor functions. ``voxelize`` creates sparse voxel tensors from dense point
features, while ``devoxelize`` samples sparse voxel features back to point
rows.

The functions validate shape and dtype before entering the native backend:
points are ``float32`` ``(N, 3)``, optional batch indices are ``int32`` ``(N,)``,
and voxel feature aggregation currently uses ``float32`` features.

Related pages
-------------

* Core metadata classes: :doc:`../core/point-voxel`
* Backend implementation details: :doc:`../../reference/backend/point-voxel`
* Sparse tensor output contract: :doc:`../../reference/concepts/sparse-tensor`
* Quickstart example: :doc:`../../getting-started/quickstart`

.. automodule:: mlx_lattice.ops.quantization
   :members:
