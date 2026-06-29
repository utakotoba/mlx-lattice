API reference
=============

The API reference is generated from public Python modules and is kept separate
from the conceptual documentation. Use these pages when you need signatures,
parameter names, class members, shape contracts, dtype contracts, and error
boundaries. Use the concept and backend references when you need architectural
context or route-selection details.

The public surface is grouped by semantic feature:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Section
     - Use it for
     - Coordinate effect
   * - :doc:`core/index`
     - Containers, coordinate identity, relation metadata, quantized weights,
       and shared type helpers.
     - Defines coordinate ownership and metadata used by every operation.
   * - :doc:`ops/index`
     - Functional sparse operators such as convolution, pooling, feature
       transforms, sparse algebra, relation builders, and point/voxel helpers.
     - Each operation documents whether it preserves, changes, aligns, or
       generates sparse coordinates.
   * - :doc:`nn/index`
     - ``mlx.nn.Module`` wrappers over sparse operations.
     - Mirrors the corresponding functional operation.
   * - :doc:`ir`
     - Backend-neutral manifest dataclasses, schema validation, and annotated
       IR operation registry from ``lattice_contract``.
     - Defines the stable sparse model artifact contract without importing
       the MLX artifact consumer.
   * - :doc:`artifact`
     - Lattice artifact loading/saving and in-memory graph execution.
     - Reconstructs graph semantics and dispatches through public operations.
   * - :doc:`native`
     - Stable backend diagnostics through ``mlx_lattice.backend_info``.
     - Does not mutate sparse tensors.

.. toctree::
   :maxdepth: 2

   core/index
   ops/index
   nn/index
   ir
   artifact
   native
