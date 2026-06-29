IR API
======

``mlx_lattice.ir`` contains the manifest dataclasses, schema validation helpers,
and annotated operation registry used by the lattice artifact loader. It is a
small semantic layer: it describes graph structure and operation contracts, but
does not execute sparse kernels directly.

For the conceptual model, read :doc:`../reference/concepts/model-ir`.

Manifest model
--------------

.. automodule:: mlx_lattice.ir.manifest
   :members:

Operation registry
------------------

``lattice_op_hints`` is the public annotation used by operation definitions
when type annotations are not enough to classify an argument for artifact
export. Typical examples are persisted dense or packed quantized weights,
optional affine vectors, and graph-carried optional values. The export runtime
uses these hints together with function signatures, so new public ops can join
the artifact surface without adding a bespoke runtime handler.

.. automodule:: mlx_lattice.ir.ops
   :members:
