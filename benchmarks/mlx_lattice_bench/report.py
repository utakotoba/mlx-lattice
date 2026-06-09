from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx_lattice import __version__, backend_info

from mlx_lattice_bench.harness import BenchmarkResult


def environment() -> dict[str, Any]:
    info = cast('dict[str, Any]', backend_info())
    return {
        'git_sha': _git_sha(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'mlx_version': getattr(mx, '__version__', 'unknown'),
        'mlx_lattice_version': __version__,
        'backend_info': info,
    }


def write_json(
    path: Path,
    *,
    results: list[BenchmarkResult],
) -> None:
    payload = {
        'environment': environment(),
        'results': [result.to_json() for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def write_summary(
    path: Path,
    *,
    results: list[BenchmarkResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table(results, color=False) + '\n')


def table(results: list[BenchmarkResult], *, color: bool = False) -> str:
    if not results:
        return _style('No benchmark results.', color, '33')

    headers = [
        'case',
        'mode',
        'device',
        'params',
        'workload',
        'median_ms',
        'p95_ms',
        'throughput',
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.case,
                result.mode,
                result.device,
                _params(result.params),
                _workload(result.workload),
                f'{result.median_ms:.3f}',
                f'{result.p95_ms:.3f}',
                _throughput(result.units),
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    lines = [
        '  '.join(
            _style(headers[col].ljust(widths[col]), color, '1')
            for col in range(len(headers))
        ),
        '  '.join('-' * width for width in widths),
    ]
    for row in rows:
        lines.append(
            '  '.join(
                row[col].ljust(widths[col]) for col in range(len(row))
            )
        )
    return '\n'.join(lines)


def _params(params: dict[str, Any]) -> str:
    parts = []
    for key in ('N', 'points', 'channels', 'batches', 'kernel'):
        if key in params:
            parts.append(f'{key}={params[key]}')
    return ','.join(parts)


def _workload(workload: dict[str, int | float]) -> str:
    labels = (
        ('points', 'P'),
        ('n_in', 'Nin'),
        ('n_out', 'Nout'),
        ('n_query', 'Nq'),
        ('edges', 'E'),
        ('channels_in', 'Cin'),
        ('channels_out', 'Cout'),
        ('kernel_volume', 'K'),
        ('avg_neighbors', 'avgN'),
    )
    parts = []
    for key, label in labels:
        value = workload.get(key)
        if isinstance(value, int):
            parts.append(f'{label}={value}')
        elif isinstance(value, float):
            parts.append(f'{label}={value:.2f}')
    return ','.join(parts)


def _throughput(units: dict[str, float]) -> str:
    if not units:
        return ''
    parts = []
    for key, value in list(units.items())[:2]:
        parts.append(f'{value:.1f} {key}')
    return ', '.join(parts)


def _style(text: str, enabled: bool, *codes: str) -> str:
    if not enabled or not codes:
        return text
    return f'\033[{";".join(codes)}m{text}\033[0m'


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return result.stdout.strip()
