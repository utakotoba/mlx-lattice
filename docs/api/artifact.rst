Artifact API
============

``mlx_lattice.artifact`` loads and saves lattice model artifacts. The deployment
entry point is :func:`mlx_lattice.artifact.load_lattice_model`, which reads a
manifest and ``safetensors`` weight file, validates the graph, and returns an
in-memory :class:`mlx_lattice.artifact.LatticeModel`.

The package implements the MLX-side artifact loader, graph executor, graph
builder, and strict low-level writer used by tests and future producers.

The artifact implementation has three layers:

* :mod:`mlx_lattice.artifact.io` handles the on-disk artifact directory;
* :mod:`mlx_lattice.artifact.model` executes a validated in-memory manifest;
* :mod:`mlx_lattice.artifact.registry` maps manifest operations and module
  annotations to approved public ``mlx_lattice.ops`` calls.

Module artifact helpers under :mod:`mlx_lattice.artifact.builder` build manifests
and weight dictionaries from serializable sparse NN modules or explicit graph
builders. Shared artifact binding primitives live in
:mod:`mlx_lattice.artifact.bindings`.

Explicit graph builders infer output value types from registered operation
return annotations. Callers can still override the type with
:class:`mlx_lattice.artifact.GraphOutput` when a custom graph value needs a more
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

The artifact runner honors manifest ``dtype_policy`` for floating dense arrays and
sparse feature matrices while preserving coordinates, integer arrays, byte
streams, and packed quantized payloads.

Artifact manifests also carry runtime compatibility metadata. The loader
accepts artifacts targeted at ``mlx-lattice`` whose version specifier matches
the installed native mlx-lattice package, and rejects incompatible runtime metadata names or version
windows before dispatching any graph node.

.. automodule:: mlx_lattice.artifact
   :members:

Artifact I/O
------------

.. automodule:: mlx_lattice.artifact.io
   :members:

Artifact model
--------------

.. automodule:: mlx_lattice.artifact.model
   :members:

Artifact registry
-----------------

.. automodule:: mlx_lattice.artifact.registry
   :members:

Artifact bindings
-----------------

.. automodule:: mlx_lattice.artifact.bindings
   :members:

Module artifact
---------------

.. automodule:: mlx_lattice.artifact.builder
   :members:
