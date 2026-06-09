from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal

import mlx.core as mx
from mlx_lattice.core import (
    CoordinateSet,
    KernelRelation,
    NeighborRelation,
    SparseQuantization,
    SparseTensor,
)

type Mode = Literal['cold_op', 'hot_op', 'compiled_hot', 'backward']
type Params = Mapping[str, Any]
type EvalTarget = Sequence[mx.array]
type WorkloadMetrics = dict[str, int | float]
type CompiledFactory = Callable[
    [Any], tuple[Callable[..., Any], tuple[Any, ...]]
]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    group: str
    params: tuple[Params, ...]
    setup: Callable[[Params], Any]
    prepare: Callable[[Any], Any]
    run: Callable[[Any], Any]
    compiled: CompiledFactory | None = None
    backward: CompiledFactory | None = None
    units: tuple[str, ...] = ()
    modes: tuple[Mode, ...] | None = None

    def supports(self, mode: Mode) -> bool:
        if self.modes is not None and mode not in self.modes:
            return False
        if mode == 'compiled_hot':
            return self.compiled is not None
        if mode == 'backward':
            return self.backward is not None
        return True


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    case: str
    group: str
    mode: Mode
    device: str
    params: dict[str, Any]
    warmup: int
    repeats: int
    median_ms: float
    min_ms: float
    p90_ms: float
    p95_ms: float
    samples_ms: tuple[float, ...]
    workload: WorkloadMetrics
    units: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            'case': self.case,
            'group': self.group,
            'mode': self.mode,
            'device': self.device,
            'params': _jsonable(self.params),
            'warmup': self.warmup,
            'repeats': self.repeats,
            'median_ms': self.median_ms,
            'min_ms': self.min_ms,
            'p90_ms': self.p90_ms,
            'p95_ms': self.p95_ms,
            'samples_ms': list(self.samples_ms),
            'workload': self.workload,
            'units': self.units,
        }


type ProgressStart = Callable[[BenchmarkCase, Params, Mode, str], None]
type ProgressResult = Callable[
    [BenchmarkResult, BenchmarkCase, Params, Mode, str], None
]
type ProgressSkip = Callable[[BenchmarkCase, Params, Mode, str], None]
type ProgressError = Callable[
    [BenchmarkCase, Params, Mode, str, BaseException], None
]


def run_case(
    case: BenchmarkCase,
    params: Params,
    *,
    mode: Mode,
    device: str,
    warmup: int,
    repeats: int,
) -> BenchmarkResult | None:
    if not case.supports(mode):
        return None

    fixture = case.setup(params)
    _force(fixture)
    execution = _execution_for(case, fixture, mode)
    _force(execution.prepared)

    last_output: Any | None = None
    for _ in range(warmup):
        last_output = execution.action()
        _force(last_output)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        last_output = execution.action()
        _force(last_output)
        stop = time.perf_counter_ns()
        samples.append((stop - start) / 1_000_000.0)

    sample_tuple = tuple(samples)
    metric_prepared = execution.prepared
    metric_output = last_output
    if mode in ('compiled_hot', 'backward'):
        metric_prepared = case.prepare(fixture)
        _force(metric_prepared)
        metric_output = case.run(metric_prepared)
        _force(metric_output)
    workload = _derive_workload_metrics(
        params,
        fixture=fixture,
        prepared=metric_prepared,
        output=metric_output,
    )
    return BenchmarkResult(
        case=case.name,
        group=case.group,
        mode=mode,
        device=device,
        params=dict(params),
        warmup=warmup,
        repeats=repeats,
        median_ms=statistics.median(sample_tuple),
        min_ms=min(sample_tuple),
        p90_ms=_percentile(sample_tuple, 90),
        p95_ms=_percentile(sample_tuple, 95),
        samples_ms=sample_tuple,
        workload=workload,
        units=_derive_units(sample_tuple, params, workload, case.units),
    )


def run_cases(
    cases: Iterable[BenchmarkCase],
    *,
    modes: Sequence[Mode],
    device: str,
    warmup: int,
    repeats: int,
    include: str | None = None,
    on_start: ProgressStart | None = None,
    on_result: ProgressResult | None = None,
    on_skip: ProgressSkip | None = None,
    on_error: ProgressError | None = None,
) -> list[BenchmarkResult]:
    results = []
    for case in cases:
        if include is not None and include not in case.name:
            continue
        for params in case.params:
            for mode in modes:
                if not case.supports(mode):
                    if on_skip is not None:
                        on_skip(case, params, mode, device)
                    continue
                if on_start is not None:
                    on_start(case, params, mode, device)
                try:
                    result = run_case(
                        case,
                        params,
                        mode=mode,
                        device=device,
                        warmup=warmup,
                        repeats=repeats,
                    )
                except Exception as error:
                    if on_error is not None:
                        on_error(case, params, mode, device, error)
                    raise
                if result is not None:
                    if on_result is not None:
                        on_result(result, case, params, mode, device)
                    results.append(result)
    return results


@dataclass(frozen=True, slots=True)
class _Execution:
    action: Callable[[], Any]
    prepared: Any | None


def _execution_for(
    case: BenchmarkCase,
    fixture: Any,
    mode: Mode,
) -> _Execution:
    if mode == 'cold_op':
        prepared = case.prepare(fixture)
        return _Execution(
            action=lambda: case.run(case.prepare(fixture)),
            prepared=prepared,
        )

    if mode == 'hot_op':
        prepared = case.prepare(fixture)
        return _Execution(
            action=lambda: case.run(prepared), prepared=prepared
        )

    if mode == 'compiled_hot':
        if case.compiled is None:
            raise ValueError(f'{case.name} does not support compiled_hot.')
        fn, args = case.compiled(fixture)
        _force(args)
        compiled = mx.compile(fn)
        return _Execution(action=lambda: compiled(*args), prepared=args)

    if case.backward is None:
        raise ValueError(f'{case.name} does not support backward.')
    fn, args = case.backward(fixture)
    _force(args)
    return _Execution(action=lambda: fn(*args), prepared=args)


def _force(value: Any) -> None:
    arrays = tuple(_collect_arrays(value))
    if arrays:
        mx.eval(*arrays)


def _collect_arrays(value: Any) -> Iterable[mx.array]:
    if isinstance(value, mx.array):
        yield value
        return
    if isinstance(value, SparseTensor):
        yield value.coords
        yield value.feats
        yield value.active_rows
        return
    if isinstance(value, CoordinateSet):
        yield value.coords
        yield value.active_rows
        return
    if isinstance(value, SparseQuantization):
        yield value.coords
        yield value.active_rows
        yield value.inverse_rows
        yield value.counts
        return
    if isinstance(value, KernelRelation):
        yield value.edges.in_rows
        yield value.edges.out_rows
        yield value.edges.kernel_ids
        if value.out_coords is not None:
            yield value.out_coords
        yield value.counts
        return
    if isinstance(value, NeighborRelation):
        yield value.edges.query_rows
        yield value.edges.source_rows
        yield value.edges.neighbor_ids
        yield value.distances
        yield value.row_offsets
        yield value.counts
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _collect_arrays(item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            yield from _collect_arrays(item)
        return
    if _is_dataclass_instance(value):
        for field in fields(value):
            yield from _collect_arrays(getattr(value, field.name))


def _percentile(samples: Sequence[float], pct: int) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _derive_units(
    samples: Sequence[float],
    params: Params,
    workload: WorkloadMetrics,
    units: Sequence[str],
) -> dict[str, float]:
    median_seconds = statistics.median(samples) / 1000.0
    if median_seconds <= 0.0:
        return {}

    out = {}
    unit_names = units or _default_units(workload)
    for unit in unit_names:
        raw = workload.get(unit, params.get(unit))
        if isinstance(raw, int | float):
            out[f'{unit}_per_s'] = float(raw) / median_seconds
    return out


def _derive_workload_metrics(
    params: Params,
    *,
    fixture: Any,
    prepared: Any | None,
    output: Any | None,
) -> WorkloadMetrics:
    metrics: WorkloadMetrics = {}
    _apply_param_metrics(metrics, params)
    input_metric_count = len(metrics)
    _apply_input_metrics(metrics, prepared)
    if len(metrics) == input_metric_count:
        _apply_input_metrics(metrics, fixture)
    _apply_output_metrics(metrics, output)
    _derive_composite_metrics(metrics)
    return metrics


def _apply_param_metrics(metrics: WorkloadMetrics, params: Params) -> None:
    for key, metric in (
        ('N', 'N'),
        ('points', 'points'),
        ('channels', 'channels_in'),
        ('batches', 'batches'),
        ('kernel', 'kernel_volume'),
    ):
        value = params.get(key)
        if isinstance(value, int | float):
            _setdefault_metric(metrics, metric, value)


def _apply_input_metrics(metrics: WorkloadMetrics, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, SparseTensor):
        _setdefault_metric(
            metrics,
            'n_in',
            _scalar_count(value.active_rows, value.capacity),
        )
        _setdefault_metric(metrics, 'n_in_capacity', value.capacity)
        _setdefault_metric(metrics, 'channels_in', value.channels)
        if value.batch_counts is not None:
            _setdefault_metric(metrics, 'batches', len(value.batch_counts))
        return
    if _has_attrs(value, 'points', 'feats'):
        points = value.points
        feats = value.feats
        if isinstance(points, mx.array):
            _setdefault_metric(metrics, 'points', _leading_dim(points))
        if isinstance(feats, mx.array) and feats.ndim == 2:
            _setdefault_metric(metrics, 'channels_in', int(feats.shape[1]))
        _walk_input_children(metrics, value)
        return
    if _has_attrs(value, 'points', 'point_feats'):
        points = value.points
        feats = value.point_feats
        if isinstance(points, mx.array):
            _setdefault_metric(metrics, 'points', _leading_dim(points))
        if isinstance(feats, mx.array) and feats.ndim == 2:
            _setdefault_metric(metrics, 'channels_in', int(feats.shape[1]))
        _walk_input_children(metrics, value)
        return
    if _has_attrs(value, 'coords', 'feats'):
        coords = value.coords
        feats = value.feats
        if isinstance(coords, mx.array):
            _setdefault_metric(metrics, 'n_in', _leading_dim(coords))
            _setdefault_metric(
                metrics, 'n_in_capacity', _leading_dim(coords)
            )
        if isinstance(feats, mx.array) and feats.ndim == 2:
            _setdefault_metric(metrics, 'channels_in', int(feats.shape[1]))
        _walk_input_children(metrics, value)
        return
    _walk_input_children(metrics, value)


def _walk_input_children(
    metrics: WorkloadMetrics,
    value: Any,
) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _apply_input_metrics(metrics, item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            _apply_input_metrics(metrics, item)
        return
    if _is_dataclass_instance(value):
        for field in fields(value):
            _apply_input_metrics(metrics, getattr(value, field.name))


def _apply_output_metrics(metrics: WorkloadMetrics, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, SparseTensor):
        n_out = _scalar_count(value.active_rows, value.capacity)
        _set_metric(metrics, 'n_out', n_out)
        _set_metric(metrics, 'n_out_capacity', value.capacity)
        _set_metric(metrics, 'channels_out', value.channels)
        _set_metric(metrics, 'elements', n_out * value.channels)
        return
    if isinstance(value, CoordinateSet):
        n_out = _scalar_count(value.active_rows, value.capacity)
        _set_metric(metrics, 'n_out', n_out)
        _set_metric(metrics, 'n_out_capacity', value.capacity)
        return
    if isinstance(value, SparseQuantization):
        n_out = _scalar_count(value.active_rows, value.capacity)
        _set_metric(metrics, 'n_out', n_out)
        _set_metric(metrics, 'n_out_capacity', value.capacity)
        _setdefault_metric(
            metrics, 'points', _leading_dim(value.inverse_rows)
        )
        return
    if isinstance(value, KernelRelation):
        edges = _scalar_count(value.edge_count, value.edge_capacity)
        _set_metric(metrics, 'edges', edges)
        _set_metric(metrics, 'edge_capacity', value.edge_capacity)
        _set_metric(metrics, 'n_out', _scalar_count(value.out_count))
        if value.n_in_capacity is not None:
            _setdefault_metric(
                metrics, 'n_in_capacity', value.n_in_capacity
            )
        if value.n_out_capacity is not None:
            _set_metric(metrics, 'n_out_capacity', value.n_out_capacity)
        if value.n_kernels is not None:
            _set_metric(metrics, 'kernel_volume', value.n_kernels)
        return
    if isinstance(value, NeighborRelation):
        edges = _scalar_count(value.edge_count, value.edge_capacity)
        _set_metric(metrics, 'edges', edges)
        _set_metric(metrics, 'edge_capacity', value.edge_capacity)
        _set_metric(metrics, 'n_query', _scalar_count(value.query_count))
        if value.n_query_capacity is not None:
            _set_metric(metrics, 'n_query_capacity', value.n_query_capacity)
        if value.n_source_capacity is not None:
            _setdefault_metric(
                metrics, 'n_source_capacity', value.n_source_capacity
            )
        if value.max_neighbors is not None:
            _set_metric(metrics, 'max_neighbors', value.max_neighbors)
        return
    if isinstance(value, mx.array):
        _setdefault_metric(metrics, 'n_out', _leading_dim(value))
        _setdefault_metric(metrics, 'elements', _array_size(value))
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _apply_output_metrics(metrics, item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            _apply_output_metrics(metrics, item)
        return
    if _is_dataclass_instance(value):
        for field in fields(value):
            _apply_output_metrics(metrics, getattr(value, field.name))


def _derive_composite_metrics(metrics: WorkloadMetrics) -> None:
    n_out = metrics.get('n_out')
    channels_out = metrics.get('channels_out')
    if (
        'elements' not in metrics
        and isinstance(n_out, int | float)
        and isinstance(channels_out, int | float)
    ):
        _set_metric(metrics, 'elements', n_out * channels_out)

    edges = metrics.get('edges')
    neighbor_base = metrics.get('n_out', metrics.get('n_query'))
    if (
        isinstance(edges, int | float)
        and isinstance(neighbor_base, int | float)
        and neighbor_base > 0
    ):
        _set_metric(metrics, 'avg_neighbors', edges / neighbor_base)


def _default_units(workload: WorkloadMetrics) -> tuple[str, ...]:
    return tuple(
        unit
        for unit in (
            'edges',
            'elements',
            'points',
            'n_out',
            'n_in',
            'N',
        )
        if unit in workload
    )


def _setdefault_metric(
    metrics: WorkloadMetrics,
    key: str,
    value: int | float,
) -> None:
    if key not in metrics:
        _set_metric(metrics, key, value)


def _set_metric(
    metrics: WorkloadMetrics,
    key: str,
    value: int | float,
) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        metrics[key] = value
        return
    if isinstance(value, float):
        metrics[key] = value


def _scalar_count(value: mx.array, fallback: int | None = None) -> int:
    if value.shape == (1,):
        return int(value.item())
    if fallback is not None:
        return int(fallback)
    return int(value.shape[0])


def _leading_dim(value: mx.array) -> int:
    if value.ndim == 0:
        return 1
    return int(value.shape[0])


def _array_size(value: mx.array) -> int:
    size = 1
    for dim in value.shape:
        size *= int(dim)
    return size


def _has_attrs(value: Any, *names: str) -> bool:
    return all(hasattr(value, name) for name in names)


def _is_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
