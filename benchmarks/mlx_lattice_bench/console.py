from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TextIO

type ColorMode = Literal['auto', 'always', 'never']


@dataclass(slots=True)
class Console:
    use_color: bool
    quiet: bool = False
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    total: int | None = None
    current: int = 0

    def set_total(self, total: int) -> None:
        self.total = total
        self.current = 0

    def heading(self, text: str) -> None:
        if self.quiet:
            return
        self.write(self.style(text, '1', '36'))

    def start(
        self,
        case: Any,
        params: Mapping[str, Any],
        mode: str,
        device: str,
    ) -> None:
        if self.quiet:
            return
        self.current += 1
        progress = self._progress()
        name = f'{case.group}/{case.name}'
        detail = _format_params(params)
        label = self.style('run', '36')
        self.write(
            f'{progress}{label} {device:<5} {mode:<12} {name} {detail}'.rstrip()
        )

    def done(self, result: Any, *_: Any) -> None:
        if self.quiet:
            return
        label = self.style('ok ', '32')
        workload = _format_workload(result.workload)
        self.write(
            f'     {label} median={result.median_ms:.3f}ms '
            f'p95={result.p95_ms:.3f}ms {workload}'.rstrip()
        )

    def failed(
        self,
        case: Any,
        params: Mapping[str, Any],
        mode: str,
        device: str,
        error: BaseException,
    ) -> None:
        if self.quiet:
            return
        name = f'{case.group}/{case.name}'
        detail = _format_params(params)
        label = self.style('fail', '31')
        self.write(
            f'     {label} {device:<5} {mode:<12} {name} {detail}: {error}'
        )

    def skipped(
        self,
        case: Any,
        params: Mapping[str, Any],
        mode: str,
        device: str,
    ) -> None:
        if self.quiet:
            return
        name = f'{case.group}/{case.name}'
        detail = _format_params(params)
        label = self.style('skip', '2')
        self.write(
            f'     {label} {device:<5} {mode:<12} {name} {detail}'.rstrip()
        )

    def write(self, text: str = '') -> None:
        print(text, file=self.stream, flush=True)

    def style(self, text: str, *codes: str) -> str:
        if not self.use_color or not codes:
            return text
        return f'\033[{";".join(codes)}m{text}\033[0m'

    def _progress(self) -> str:
        if self.total is None:
            return ''
        width = len(str(max(self.total, 1)))
        return self.style(
            f'[{self.current:{width}d}/{self.total:{width}d}] ',
            '2',
        )


def make_console(color: ColorMode, *, quiet: bool = False) -> Console:
    if color == 'always':
        use_color = True
    elif color == 'never':
        use_color = False
    else:
        use_color = sys.stdout.isatty()
    return Console(use_color=use_color, quiet=quiet)


def _format_params(params: Mapping[str, Any]) -> str:
    preferred = (
        'rows',
        'points',
        'channels',
        'batches',
        'kernel',
        'stride',
        'dilation',
        'neighbors',
        'radius',
        'k',
    )
    parts = [f'{key}={params[key]}' for key in preferred if key in params]
    extras = sorted(key for key in params if key not in preferred)
    parts.extend(f'{key}={params[key]}' for key in extras)
    return ' '.join(parts)


def _format_workload(workload: Mapping[str, Any]) -> str:
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
    return ' '.join(parts)


__all__ = ['ColorMode', 'Console', 'make_console']
