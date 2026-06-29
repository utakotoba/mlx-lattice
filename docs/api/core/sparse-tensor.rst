Sparse tensor
=============

``SparseTensor`` is the central value type. It binds batched lattice
coordinates to feature rows and carries coordinate identity metadata so sparse
relations can be cached and reused.

Coordinate rows use ``(batch, x, y, z)`` order. Feature rows use the same row
order as the coordinate buffer. ``active_rows`` separates the static buffer
capacity from the dynamic number of valid rows:

.. math::

   \texttt{coords} \in \mathbb{Z}^{N\times4},\quad
   \texttt{feats} \in \mathbb{R}^{N\times C},\quad
   0 \le \texttt{active\_rows}[0] \le N.

Feature-only operations preserve coordinate identity. Row-changing operations
construct a new coordinate key and active-row scalar.

Related pages
-------------

* Concept model: :doc:`../../reference/concepts/sparse-tensor`
* Coordinate ownership: :doc:`coordinate-management`
* Feature-preserving operations: :doc:`../ops/feature`
* Row-changing sparse algebra: :doc:`../ops/tensor`
* Backend route inputs: :doc:`../../reference/backend/path-selection`

.. automodule:: mlx_lattice.core.tensor
   :members:
