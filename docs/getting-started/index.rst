Getting started
===============

The getting-started section explains how to install ``mlx-lattice`` and how to
think about a sparse pipeline before reading individual API entries. The API is
compact, but sparse tensor code has a few concepts that are easier to learn in
order: coordinate rows, feature rows, coordinate identity, relations, and
backend path selection.

Most user code follows the same shape:

1. build or receive a set of batched integer coordinates;
2. attach feature rows to those coordinates with :class:`mlx_lattice.SparseTensor`;
3. apply sparse operators or ``mlx_lattice.nn`` modules;
4. optionally pool, voxelize, devoxelize, quantize weights, or align multiple
   sparse tensors.

The public API does not require choosing CPU kernels, Metal kernels, TensorOps
kernels, or implicit-GEMM kernels by hand. Those are internal execution choices.
The user-facing responsibility is to keep the sparse tensor contract valid:
coordinates must be integer ``(N, 4)`` rows, features must have ``N`` rows, and
operators that combine tensors must either share coordinate identity or ask for
a value-aligned join.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   workflow

Recommended reading order
-------------------------

Start with :doc:`installation` if you are setting up a checkout or a
development environment. Then read :doc:`quickstart` for a small executable
walkthrough. :doc:`workflow` is more conceptual: it explains how individual
building blocks connect in a real model or data-processing pipeline.

After that, the reference sections become more useful:

* :doc:`../reference/concepts/sparse-tensor` describes the sparse tensor value
  model and the coordinate manager.
* :doc:`../reference/concepts/coordinates-relations` describes relation objects,
  kernel geometry, and neighbor views.
* :doc:`../reference/concepts/algebra` describes value-aligned sparse algebra.
* :doc:`../reference/backend/path-selection` explains how the backend dispatch
  chooses CPU, classic Metal, TensorOps, and quantized routes.

Development expectations
------------------------

``mlx-lattice`` contains Python code, MLX integration code, and native CPU/Metal
extensions. A source checkout is therefore more involved than a pure Python
package. When contributing, keep three checks separate in your head:

``ty``
   Verifies Python typing and catches API-level inconsistencies.

``pytest``
   Verifies semantic behavior and backend parity where the local environment can
   execute the requested backend.

``prek``
   Runs the repository's configured formatting, linting, and native static
   checks. This is the broadest local hygiene gate and is the closest match for
   CI style checks.

Documentation describes public semantics and stable maintainer contracts.
Temporary environment variables, internal kernel names, and benchmark switches
are not presented as user APIs.
