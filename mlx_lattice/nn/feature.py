from __future__ import annotations

import mlx.nn as mxnn

from mlx_lattice.core import SparseTensor

__all__ = [
    'GELU',
    'BatchNorm',
    'Dropout',
    'LayerNorm',
    'LeakyReLU',
    'Linear',
    'RMSNorm',
    'ReLU',
    'SiLU',
    'Sigmoid',
    'Softplus',
    'Tanh',
]


class Linear(mxnn.Linear):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class ReLU(mxnn.ReLU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class Sigmoid(mxnn.Sigmoid):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class GELU(mxnn.GELU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class SiLU(mxnn.SiLU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class LeakyReLU(mxnn.LeakyReLU):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class Tanh(mxnn.Tanh):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class Softplus(mxnn.Softplus):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class Dropout(mxnn.Dropout):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class BatchNorm(mxnn.BatchNorm):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class LayerNorm(mxnn.LayerNorm):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))


class RMSNorm(mxnn.RMSNorm):
    def __call__(self, x: SparseTensor) -> SparseTensor:
        return x.replace(feats=super().__call__(x.feats))
