Coordinate operations
=====================

``mlx_lattice.ops.coords`` is a convenience re-export module for coordinate
helpers whose canonical implementations live under ``mlx_lattice.core.coords``.
The canonical API pages are:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Operation group
     - Canonical API page
   * - Coordinate set operations
     - :doc:`../core/coordinate-utilities`
   * - Morton ordering
     - :doc:`../core/coordinate-utilities`
   * - Occupancy downsample/expand
     - :doc:`../core/coordinate-utilities`
   * - Sparse alignment
     - :doc:`../core/coordinate-utilities`
   * - Point/voxel quantization helpers
     - :doc:`../core/point-voxel`

The re-export exists so user code can write operation-oriented imports such as
``from mlx_lattice.ops import downsample_coords``. The docs avoid rendering the
same functions here a second time.
