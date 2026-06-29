from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as mxnn
from lattice_contract import (
    IRManifest,
    load_manifest,
    manifest_to_dict,
)

from mlx_lattice.artifact.builder import (
    LatticeGraphBuilder,
    build_lattice_graph_artifact,
    build_lattice_module_artifact,
)
from mlx_lattice.artifact.model import LatticeModel

_MANIFEST_NAME = 'manifest.json'
_WEIGHTS_NAME = 'weights.safetensors'


@dataclass(frozen=True, slots=True)
class LatticeArtifact:
    """Loaded lattice model artifact."""

    manifest: IRManifest
    weights: dict[str, mx.array]

    def model(self) -> LatticeModel:
        """Construct an executable in-memory MLX graph."""

        return LatticeModel(self.manifest, self.weights)


def load_lattice_artifact(path: str | Path) -> LatticeArtifact:
    """Load a lattice artifact directory.

    The artifact directory must contain ``manifest.json`` and
    ``weights.safetensors``. Use :meth:`LatticeArtifact.model` or
    :func:`load_lattice_model` when an executable graph runner is needed.
    """

    return _load_artifact(path)


def load_lattice_model(path: str | Path) -> LatticeModel:
    """Load a lattice artifact directory into an executable model.

    The model-oriented name is the recommended deployment entry point.
    """

    return load_lattice_artifact(path).model()


def save_lattice_artifact(
    path: str | Path,
    manifest: IRManifest,
    weights: dict[str, mx.array],
) -> None:
    """Write a lattice artifact directory.

    This helper is intentionally strict and low level. It is useful for tests,
    fixtures, and future producers that already own a valid manifest.
    """

    LatticeModel(manifest, weights)
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / _MANIFEST_NAME
    weights_path = root / _WEIGHTS_NAME
    with manifest_path.open('w', encoding='utf-8') as file:
        json.dump(manifest_to_dict(manifest), file, indent=2)
        file.write('\n')
    mx.save_safetensors(str(weights_path), weights)


def save_lattice_model(
    path: str | Path,
    manifest: IRManifest,
    weights: dict[str, mx.array],
) -> None:
    """Alias for :func:`save_lattice_artifact`."""

    save_lattice_artifact(path, manifest, weights)


def save_lattice_module(
    path: str | Path,
    module: mxnn.Module,
    **kwargs,
) -> None:
    """Build and save a serializable sparse NN module."""

    artifact = build_lattice_module_artifact(module, **kwargs)
    save_lattice_artifact(path, artifact.manifest, artifact.weights)


def save_lattice_graph(
    path: str | Path,
    builder: LatticeGraphBuilder,
    **kwargs,
) -> None:
    """Build and save an explicitly built lattice graph."""

    artifact = build_lattice_graph_artifact(builder, **kwargs)
    save_lattice_artifact(path, artifact.manifest, artifact.weights)


def _load_artifact(path: str | Path) -> LatticeArtifact:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(
            f'lattice artifact directory does not exist: {root}'
        )
    manifest_path = root / _MANIFEST_NAME
    weights_path = root / _WEIGHTS_NAME
    if not manifest_path.is_file():
        raise ValueError(f'lattice artifact is missing {_MANIFEST_NAME}.')
    if not weights_path.is_file():
        raise ValueError(f'lattice artifact is missing {_WEIGHTS_NAME}.')
    weights = mx.load(str(weights_path))
    if not isinstance(weights, dict):
        raise ValueError(
            'weights.safetensors must load as a tensor mapping.'
        )
    return LatticeArtifact(
        load_manifest(manifest_path),
        cast(dict[str, mx.array], weights),
    )
