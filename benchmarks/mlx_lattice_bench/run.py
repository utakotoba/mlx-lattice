from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from mlx_lattice_bench.catalog import GROUPS, MODES, PRESETS
from mlx_lattice_bench.console import make_console

_RESULTS_DIR = Path('benchmarks/results')


@dataclass(frozen=True, slots=True)
class _Runtime:
    all_cases: Any
    available_devices: Any
    default_device: Any
    run_cases: Any
    write_json: Any
    write_summary: Any


def main() -> None:
    args = _parser().parse_args()
    groups = tuple(args.group) if args.group else GROUPS
    modes = tuple(args.mode) if args.mode else ('cold_op', 'hot_op')
    n_values = tuple(args.n_values) if args.n_values else None
    channels = tuple(args.channels) if args.channels else None
    channel_pairs = tuple(args.channel_pair) if args.channel_pair else None
    dtype = _dtype_name(args.dtype)

    runtime = _load_runtime(
        show_build_log=args.show_build_log,
        preload_native=not args.list,
    )
    console = make_console(args.color, quiet=args.quiet)
    cases = runtime.all_cases(
        args.preset,
        groups=groups,
        n_values=n_values,
        channels=channels,
        channel_pairs=channel_pairs,
        dtype=dtype,
    )

    if args.list:
        for case in cases:
            print(f'{case.group}/{case.name}')
        return

    devices = runtime.available_devices(args.device)
    total = _count_runs(cases, modes, devices, args.case_filter)
    console.set_total(total)
    json_path, summary_path = _report_paths(args)

    results = []
    for device in devices:
        console.heading(f'device {device.name}')
        with runtime.default_device(device):
            results.extend(
                runtime.run_cases(
                    cases,
                    modes=modes,
                    device=device.name,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    include=args.case_filter,
                    on_start=console.start,
                    on_result=console.done,
                    on_error=console.failed,
                )
            )

    runtime.write_json(json_path, results=results)
    runtime.write_summary(summary_path, results=results)
    console.report(json_path, summary_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='mlx-lattice-bench',
        description='Benchmark mlx-lattice public Python operator surfaces.',
    )
    parser.add_argument(
        '--preset',
        choices=PRESETS,
        default='smoke',
        help='parameter matrix size',
    )
    parser.add_argument(
        '--device',
        choices=('cpu', 'metal', 'cuda', 'all'),
        default='cpu',
        help='backend device selection',
    )
    parser.add_argument(
        '--mode',
        action='append',
        choices=MODES,
        help='benchmark mode; repeat flag to run multiple modes',
    )
    parser.add_argument(
        '--group',
        action='append',
        choices=GROUPS,
        help='case group; repeat flag to run multiple groups',
    )
    parser.add_argument(
        '--case-filter',
        help='substring filter applied to case names',
    )
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument(
        '--size',
        dest='n_values',
        action='append',
        type=_positive_int,
        help='planned input size N; repeat to sweep multiple values',
    )
    parser.add_argument(
        '--channels',
        action='append',
        type=_positive_int,
        help='channel count for channel-aware cases; repeat to sweep values',
    )
    parser.add_argument(
        '--channel-pair',
        action='append',
        type=_channel_pair,
        metavar='CIN:COUT',
        help=(
            'conv-only input/output channel pair; repeat to sweep values '
            '(for example 16:32)'
        ),
    )
    parser.add_argument(
        '--dtype',
        choices=('float32', 'float16'),
        default='float32',
        help='feature/weight dtype for dtype-aware cases',
    )
    parser.add_argument(
        '--output',
        help=(
            'JSON report filename/path; relative paths resolve under '
            'benchmarks/results'
        ),
    )
    parser.add_argument(
        '--color',
        choices=('auto', 'always', 'never'),
        default='auto',
        help='colorize status output',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='hide progress output; reports are still written to files',
    )
    parser.add_argument(
        '--show-build-log',
        action='store_true',
        help='show editable native-extension rebuild output during startup',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='list selected cases without running benchmarks',
    )
    return parser


def _load_runtime(
    *, show_build_log: bool, preload_native: bool
) -> _Runtime:
    if show_build_log:
        with _editable_verbose():
            return _import_runtime(preload_native=preload_native)

    capture: _CapturedOutput | None = None
    try:
        with _CapturedOutput() as capture:
            return _import_runtime(preload_native=preload_native)
    except Exception:
        if capture is not None:
            capture.replay()
        raise


def _import_runtime(*, preload_native: bool) -> _Runtime:
    from mlx_lattice_bench.cases import all_cases
    from mlx_lattice_bench.devices import available_devices, default_device
    from mlx_lattice_bench.harness import run_cases
    from mlx_lattice_bench.report import write_json, write_summary

    if preload_native:
        _preload_native_extension()

    return _Runtime(
        all_cases=all_cases,
        available_devices=available_devices,
        default_device=default_device,
        run_cases=run_cases,
        write_json=write_json,
        write_summary=write_summary,
    )


def _preload_native_extension() -> None:
    from mlx_lattice._native import ext

    ext.version()


@contextmanager
def _editable_verbose() -> Iterator[None]:
    old = os.environ.get('SKBUILD_EDITABLE_VERBOSE')
    os.environ['SKBUILD_EDITABLE_VERBOSE'] = '1'
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('SKBUILD_EDITABLE_VERBOSE', None)
        else:
            os.environ['SKBUILD_EDITABLE_VERBOSE'] = old


def _count_runs(
    cases: Sequence[Any],
    modes: Sequence[str],
    devices: Sequence[Any],
    include: str | None,
) -> int:
    total = 0
    for _device in devices:
        for case in cases:
            if include is not None and include not in case.name:
                continue
            for _params in case.params:
                total += sum(1 for mode in modes if case.supports(mode))
    return total


def _report_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    json_path = _json_report_path(args)
    summary_path = json_path.with_suffix('.summary.txt')
    return json_path, summary_path


def _json_report_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        path = Path(args.output)
        if not path.is_absolute():
            path = _RESULTS_DIR / path
        if path.suffix == '':
            path = path.with_suffix('.json')
        return path

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    groups = _slug(args.group or GROUPS)
    modes = _slug(args.mode or ('cold_op', 'hot_op'))
    n_part = _n_slug(args.n_values)
    return (
        _RESULTS_DIR
        / f'{stamp}-{args.preset}-{args.device}-{n_part}-{groups}-{modes}.json'
    )


def _slug(values: Sequence[str]) -> str:
    return '-'.join(value.replace('_', '-') for value in values)


def _n_slug(values: Sequence[int] | None) -> str:
    if not values:
        return 'preset-N'
    return 'N' + '-'.join(str(value) for value in values)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def _channel_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(':', 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'channel pair must have form CIN:COUT'
        ) from exc
    return (_positive_int(left), _positive_int(right))


def _dtype_name(value: str) -> str:
    if value not in ('float32', 'float16'):
        raise argparse.ArgumentTypeError('must be float32 or float16')
    return value


class _CapturedOutput(AbstractContextManager['_CapturedOutput']):
    def __init__(self) -> None:
        self.stdout = ''
        self.stderr = ''
        self._stdout_file: Any | None = None
        self._stderr_file: Any | None = None
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._fallback: ExitStack | None = None

    def __enter__(self) -> _CapturedOutput:
        try:
            self._saved_stdout_fd = os.dup(1)
            self._saved_stderr_fd = os.dup(2)
        except OSError:
            self._fallback = ExitStack()
            stdout = self._fallback.enter_context(
                _redirect_text(sys.stdout)
            )
            stderr = self._fallback.enter_context(
                _redirect_text(sys.stderr)
            )
            self._stdout_file = stdout
            self._stderr_file = stderr
            return self

        sys.stdout.flush()
        sys.stderr.flush()
        self._stdout_file = tempfile.TemporaryFile(mode='w+b')
        self._stderr_file = tempfile.TemporaryFile(mode='w+b')
        os.dup2(self._stdout_file.fileno(), 1)
        os.dup2(self._stderr_file.fileno(), 2)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        del exc_type, exc_value, traceback
        self._restore()
        self.stdout, self.stderr = self._read()
        self._close()
        return None

    def replay(self) -> None:
        if self.stdout:
            print(self.stdout, file=sys.stdout, end='')
        if self.stderr:
            print(self.stderr, file=sys.stderr, end='')

    def _restore(self) -> None:
        if self._fallback is not None:
            self._fallback.close()
            return
        if self._saved_stdout_fd is None or self._saved_stderr_fd is None:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None

    def _read(self) -> tuple[str, str]:
        if isinstance(self._stdout_file, io.StringIO):
            stdout = self._stdout_file.getvalue()
        else:
            stdout = _read_binary_temp(self._stdout_file)
        if isinstance(self._stderr_file, io.StringIO):
            stderr = self._stderr_file.getvalue()
        else:
            stderr = _read_binary_temp(self._stderr_file)
        return stdout, stderr

    def _close(self) -> None:
        if self._fallback is not None:
            self._fallback = None
            return
        if self._stdout_file is not None:
            self._stdout_file.close()
            self._stdout_file = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None


@contextmanager
def _redirect_text(stream: Any) -> Iterator[io.StringIO]:
    capture = io.StringIO()
    if stream is sys.stdout:
        old = sys.stdout
        sys.stdout = capture
        try:
            yield capture
        finally:
            sys.stdout = old
        return
    old = sys.stderr
    sys.stderr = capture
    try:
        yield capture
    finally:
        sys.stderr = old


def _read_binary_temp(handle: Any | None) -> str:
    if handle is None:
        return ''
    handle.flush()
    handle.seek(0)
    return handle.read().decode(errors='replace')


__all__ = ['main']
