IR API
======

``lattice_contract`` contains the backend-neutral manifest dataclasses, schema
validation helpers, and operation-contract annotations used at the artifact
boundary. It is a small semantic package: it describes graph structure and
operation contracts, but does not import MLX, Torch, native kernels, or sparse
runtime objects.

For the conceptual model, read :doc:`../reference/concepts/model-ir`.

Manifest model
--------------

.. automodule:: lattice_contract.manifest
   :members:

Operation registry
------------------

``lattice_op_hints`` is the public annotation used by operation definitions
when type annotations are not enough to classify an argument for artifact
export. Typical examples are persisted dense or packed quantized weights,
optional affine vectors, and graph-carried optional values. The export runtime
uses these hints together with function signatures, so new public ops can join
the artifact surface without adding a bespoke runtime handler.

.. automodule:: lattice_contract.ops
   :members:
