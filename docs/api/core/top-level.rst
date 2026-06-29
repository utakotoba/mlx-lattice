Top-level package
=================

``mlx_lattice`` re-exports the most common entry points for convenience:
``SparseTensor``, coordinate manager types, quantized weight helpers,
``backend_info``, and the ``core``, ``ops``, and ``nn`` namespaces.

The canonical class and function documentation lives on the feature-specific
pages in this section. This page uses links instead of a full ``automodule``
member listing so objects such as ``SparseTensor`` are not rendered both here
and on their canonical page.

.. currentmodule:: mlx_lattice

Canonical pages
---------------

Use these pages for the actual object documentation:

* :doc:`sparse-tensor` for ``SparseTensor``.
* :doc:`coordinate-management` for ``CoordinateManager`` and
  ``CoordinateMapKey``.
* :doc:`quantized-weights` for ``QuantizedWeight``, ``quantize_weight``, and
  ``dequantize_weight``.
* :doc:`../native` for ``backend_info``.
