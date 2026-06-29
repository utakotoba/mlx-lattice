Installation
============

``mlx-lattice`` is designed for MLX workflows, so the most common target is
Apple Silicon with a recent Python and an MLX build that supports the local
device. The package also contains CPU routes for supported operators. CPU routes
are useful for development and correctness checks, but the performance-oriented
paths are Metal-oriented.

Installing from a checkout
--------------------------

The repository uses ``uv`` for dependency management. From a fresh checkout,
install the project in editable mode with the default dependency set:

.. code-block:: bash

   uv sync

For documentation work, include the docs dependency group:

.. code-block:: bash

   uv sync --group docs

The native extension is part of the package. If you change native sources, use a
normal editable rebuild path rather than assuming Python import reload is
enough. MLX extension loading happens at import time, and stale build artifacts
can otherwise make a local run appear inconsistent with the source tree.

Verifying the install
---------------------

The smallest smoke check is importing the package and asking the native layer
for backend metadata:

.. code-block:: python

   import mlx_lattice as lattice

   print(lattice.__version__)
   print(lattice.backend_info())

``backend_info`` is diagnostic. It can tell you which native version and
capability metadata are visible, but it is not a route-selection API.
Sparse operations still select their implementation from the active MLX device,
the sparse relation, the dtype, and the available native kernels.

Choosing CPU or Metal during development
----------------------------------------

Device selection follows MLX conventions. ``mlx-lattice`` does not ask users to
call ``conv3d_metal`` or ``conv3d_cpu``. You set or use the MLX device context,
then call the public operation. The sparse operator builds or reuses the
relation it needs and dispatches through the native layer.

This distinction matters when reading benchmarks or debugging performance:

* choosing Metal is a device decision;
* choosing a specialized convolution kernel is an implementation decision;
* choosing a quantized route requires passing a packed
  :class:`mlx_lattice.core.QuantizedWeight` rather than a floating weight array;
* choosing a value-aligned sparse algebra join is a semantic decision made by
  the user.

Development checks
------------------

Run the focused check that matches your change first, then run broader checks
before handing work off:

.. code-block:: bash

   uv run ty check
   uv run --no-sync pytest
   uv run --no-sync prek run --all-files

Documentation builds use Sphinx and are kept warning-free:

.. code-block:: bash

   uv run --group docs sphinx-build -W -b html docs docs/_build/html

If you see a native packaging message during a docs build, check the exit code
before treating it as a documentation failure. The docs gate is the Sphinx
command returning success with warnings treated as errors.

Platform notes
--------------

The Metal backend is the primary performance target. Tensor operation support
depends on the local Apple GPU capability and on the shape selected by the
operation. The public API is written so unsupported specialized routes fall back
to a more general route instead of changing user-visible semantics.

CPU support is a semantic backend, not a mock backend. It preserves the sparse
tensor contract, relation contract, and quantization contract. Some performance
features are Metal-only, but CPU behavior remains important because it anchors
correctness and makes tests possible on machines without the same GPU feature
set.

Troubleshooting checklist
-------------------------

When an operation fails at import or launch time, narrow the problem in layers:

1. Confirm that ``import mlx_lattice`` succeeds.
2. Confirm that ``backend_info`` returns native metadata.
3. Confirm that the MLX device you expect is active.
4. Reduce the operation to a small ``SparseTensor`` and one public operator.
5. Check whether the inputs use supported dtypes and layouts.
6. If the failure is Metal-only, compare against the CPU route where possible.

Avoid debugging from an internal kernel name first. Kernel names are useful for
maintainers once the failing public operation is known, but they are not stable
API and can change during backend refactors.
