Point/voxel core
================

Point/voxel helpers convert dense point rows into sparse voxel coordinates and
back. The core module exposes the metadata classes and lower-level builders
used by the operation wrappers.

The quantization contract is:

.. math::

   v = \left\lfloor \frac{p - o}{s} \right\rfloor,

where ``p`` is a point coordinate, ``o`` is the origin, and ``s`` is the voxel
size. The resulting sparse coordinate row is ``(batch, v_x, v_y, v_z)``.

Voxel feature aggregation supports ``sum`` and ``mean`` reductions over point
rows assigned to the same voxel. Devoxelization uses a fixed-width map with
eight voxel rows per point for linear interpolation and one effective row for
nearest interpolation.

Related pages
-------------

* High-level operation wrappers: :doc:`../ops/quantization`
* Backend route details: :doc:`../../reference/backend/point-voxel`
* Sparse tensor output contract: :doc:`../../reference/concepts/sparse-tensor`

.. automodule:: mlx_lattice.core.coords.quantization
   :members:
