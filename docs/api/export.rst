Artifact API
============

``mlx_lattice.export`` loads and saves lattice model artifacts. The deployment
entry point is :func:`mlx_lattice.export.load_lattice_model`, which reads a
manifest and ``safetensors`` weight file, validates the graph, and returns an
in-memory :class:`mlx_lattice.export.LatticeModel`.

The name ``export`` is kept for the artifact boundary as a whole. The current
package implements the MLX-side loader/runtime and a strict low-level writer
for tests and future exporters.

The MLX runtime has three layers:

* :mod:`mlx_lattice.export.artifact` handles the on-disk artifact directory;
* :mod:`mlx_lattice.export.graph` executes a validated in-memory manifest;
* :mod:`mlx_lattice.export.registry` maps manifest operations and module
  annotations to approved public ``mlx_lattice.ops`` calls.

Module export helpers under :mod:`mlx_lattice.export.modules` build manifests
and weight dictionaries from serializable sparse NN modules or explicit graph
builders. Shared runtime binding primitives live in
:mod:`mlx_lattice.export.runtime`.

Explicit graph builders infer output value types from registered operation
return annotations. Callers can still override the type with
:class:`mlx_lattice.export.GraphOutput` when a custom graph value needs a more
specific public contract. Structured lattice values such as sparse occupancy
or coordinate ordering objects expose approved tensor fields through
``LatticeGraphBuilder.field()``.
The builder and loader also validate registered input and value-attribute
types, which catches manifest wiring errors before an operation implementation
is called.

Use ``LatticeGraphBuilder.call()`` for most explicit graphs. It reads the
registered operation contract and separates graph-value inputs, value
attributes, JSON attributes, and parameters automatically. Parameter arguments
may be existing artifact key strings, dense ``mx.array`` tensors, or packed
``QuantizedWeight`` objects for quantized operation bindings. Use the
lower-level ``add_op()`` only when constructing a manifest with exact port
dictionaries.

The runtime honors manifest ``dtype_policy`` for floating dense arrays and
sparse feature matrices while preserving coordinates, integer arrays, byte
streams, and packed quantized payloads.

Artifact manifests also carry runtime compatibility metadata. The loader
accepts artifacts targeted at ``mlx-lattice`` whose version specifier matches
the installed native runtime, and rejects incompatible runtime names or version
windows before dispatching any graph node.

.. automodule:: mlx_lattice.export
   :members:

Artifact I/O
------------

.. automodule:: mlx_lattice.export.artifact
   :members:

Graph runtime
-------------

.. automodule:: mlx_lattice.export.graph
   :members:

Runtime registry
----------------

.. automodule:: mlx_lattice.export.registry
   :members:

Runtime bindings
----------------

.. automodule:: mlx_lattice.export.runtime
   :members:

Module export
-------------

.. automodule:: mlx_lattice.export.modules
   :members:
