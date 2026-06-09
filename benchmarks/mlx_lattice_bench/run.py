from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from mlx_lattice_bench.catalog import GROUPS, MODES, PRESETS
from mlx_lattice_bench.console import make_console


@dataclass(frozen=True, slots=True)
class _Runtime:
    all_cases: Any
    available_devices: Any
    default_device: Any
    run_cases: Any
    table: Any
    write_json: Any


def main() -> None:
    args = _parser().parse_args()
    groups = tuple(args.group) if args.group else GROUPS
    modes = tuple(args.mode) if args.mode else ('cold_op', 'hot_op')

    runtime = _load_runtime(
        show_build_log=args.show_build_log,
        preload_native=not args.list,
    )
    console = make_console(args.color, quiet=args.quiet)
    cases = runtime.all_cases(args.preset, groups=groups)

    if args.list:
        for case in cases:
            print(f'{case.group}/{case.name}')
        return

    devices = runtime.available_devices(args.device)
    total = _count_runs(cases, modes, devices, args.case_filter)
    console.set_total(total)

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

    print(runtime.table(results, color=console.use_color))
    if args.output is not None:
        runtime.write_json(Path(args.output), results=results)


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
        choices=('cpu', 'metal', 'all'),
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
    parser.add_argument('--output', help='write JSON report to this path')
    parser.add_argument(
        '--color',
        choices=('auto', 'always', 'never'),
        default='auto',
        help='colorize status output',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='hide per-case progress and print only the final report',
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
    from mlx_lattice_bench.report import table, write_json

    if preload_native:
        _preload_native_extension()

    return _Runtime(
        all_cases=all_cases,
        available_devices=available_devices,
        default_device=default_device,
        run_cases=run_cases,
        table=table,
        write_json=write_json,
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
