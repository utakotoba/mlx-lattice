Native and diagnostics API
==========================

The native diagnostics API reports compiled extension metadata. It is useful
for verifying that the native module imported successfully and for inspecting
backend capability strings in bug reports.

``backend_info()`` does not select routes. Public operations still dispatch
from the active MLX device, input dtype, relation metadata, shape predicates,
and available backend kernels.

Related pages
-------------

* Backend path selection: :doc:`../reference/backend/path-selection`
* Convolution routes: :doc:`../reference/backend/convolution`
* Quantized routes: :doc:`../reference/backend/quantization`

.. automodule:: mlx_lattice._native
   :members:
