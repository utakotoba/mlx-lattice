Feature operations
==================

Feature operations preserve sparse coordinate identity. They replace only the
``feats`` matrix and keep the coordinate manager, coordinate key, stride,
active rows, and batch metadata unchanged.

Use these functions when the operation is row-local or channel-local:
activations, normalization, dropout, and linear projections. A quantized linear
projection is selected by passing :class:`mlx_lattice.core.QuantizedWeight`.

Related pages
-------------

* Sparse tensor identity: :doc:`../../reference/concepts/sparse-tensor`
* Feature module wrappers: :doc:`../nn/feature`
* Quantized weight storage: :doc:`../core/quantized-weights`
* Quantized route details: :doc:`../../reference/backend/quantization`

.. automodule:: mlx_lattice.ops.feature
   :members:
