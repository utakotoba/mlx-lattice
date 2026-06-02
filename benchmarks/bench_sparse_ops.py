from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mlx_lattice import (
    KernelMap,
    SparseTensor,
    cat,
    conv3d,
    downsample,
    generative_conv_transpose3d,
    max_pool3d,
    pool3d,
    prune,
    relu,
    topk_rows,
)


@dataclass(frozen=True)
class Result:
    test: str
    variant: str
    backend: str
    n: int
    times_ms: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def q1_ms(self) -> float:
        return percentile(self.times_ms, 0.25)

    @property
    def q3_ms(self) -> float:
        return percentile(self.times_ms, 0.75)


@dataclass(frozen=True)
class Case:
    name: str
    variant: str
    factory: Callable[[int, str], Callable[[], object]]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def make_coords(n: int, *, backend: str) -> mx.array:
    span = math.ceil(n ** (1 / 3)) + 1
    dtype = mx.int32 if backend in {'cuda', 'metal'} else mx.int64
    values = [
        [0, i % span, (i // span) % span, i // (span * span)]
        for i in range(n)
    ]
    return mx.array(values, dtype=dtype)


def make_feats(n: int, channels: int) -> mx.array:
    base = mx.arange(n * channels, dtype=mx.float32).reshape((n, channels))
    return (base % 17) / 17


def make_weight(
    volume: int, in_channels: int, out_channels: int
) -> mx.array:
    base = mx.arange(
        volume * in_channels * out_channels, dtype=mx.float32
    ).reshape((volume, in_channels, out_channels))
    return (base % 13) / (13 * in_channels)


def make_tensor(n: int, backend: str, channels: int = 8) -> SparseTensor:
    return SparseTensor(
        make_coords(n, backend=backend), make_feats(n, channels)
    )


def eval_output(value: object) -> None:
    if isinstance(value, SparseTensor):
        mx.eval(value.coords, value.feats)
    elif isinstance(value, KernelMap):
        mx.eval(
            value.maps,
            value.sizes,
            value.kernels,
            value.out_coords,
        )
    elif isinstance(value, mx.array):
        mx.eval(value)
    elif isinstance(value, tuple | list):
        mx.eval(*[item for item in value if isinstance(item, mx.array)])


def measure(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
) -> tuple[float, ...]:
    for _ in range(warmup):
        eval_output(fn())

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        eval_output(fn())
        times.append((time.perf_counter() - start) * 1000)
    return tuple(times)


def bench_downsample(n: int, backend: str) -> Callable[[], object]:
    coords = make_coords(n, backend=backend)
    return lambda: downsample(coords, stride=2)


def bench_pool_cold(n: int, backend: str) -> Callable[[], object]:
    coords = make_coords(n, backend=backend)
    feats = make_feats(n, 8)

    def run() -> object:
        return pool3d(SparseTensor(coords, feats), kernel_size=2, stride=2)

    return run


def bench_pool_hot(n: int, backend: str) -> Callable[[], object]:
    x = make_tensor(n, backend)
    x.kernel_map(kernel_size=2, stride=2)
    eval_output(x.kernel_map(kernel_size=2, stride=2))
    return lambda: pool3d(x, kernel_size=2, stride=2)


def bench_max_pool_cold(n: int, backend: str) -> Callable[[], object]:
    coords = make_coords(n, backend=backend)
    feats = make_feats(n, 8)

    def run() -> object:
        return max_pool3d(
            SparseTensor(coords, feats), kernel_size=2, stride=2
        )

    return run


def bench_max_pool_hot(n: int, backend: str) -> Callable[[], object]:
    x = make_tensor(n, backend)
    x.kernel_map(kernel_size=2, stride=2)
    eval_output(x.kernel_map(kernel_size=2, stride=2))
    return lambda: max_pool3d(x, kernel_size=2, stride=2)


def bench_conv_cold(n: int, backend: str) -> Callable[[], object]:
    coords = make_coords(n, backend=backend)
    feats = make_feats(n, 8)
    weight = make_weight(27, 8, 16)

    def run() -> object:
        return conv3d(
            SparseTensor(coords, feats), weight, kernel_size=3, stride=1
        )

    return run


def bench_conv_hot(n: int, backend: str) -> Callable[[], object]:
    x = make_tensor(n, backend)
    weight = make_weight(27, 8, 16)
    x.kernel_map(kernel_size=3, stride=1)
    eval_output(x.kernel_map(kernel_size=3, stride=1))
    return lambda: conv3d(x, weight, kernel_size=3, stride=1)


def bench_pointwise(n: int, backend: str) -> Callable[[], object]:
    x = make_tensor(n, backend)
    weight = make_weight(1, 8, 16)
    return lambda: conv3d(x, weight, kernel_size=1, stride=1)


def bench_feature_chain(n: int, backend: str) -> Callable[[], object]:
    x = make_tensor(n, backend)
    weight = make_weight(1, 8, 16)

    def run() -> object:
        return relu(conv3d(x, weight, kernel_size=1, stride=1))

    return run


def bench_cat_add(n: int, backend: str) -> Callable[[], object]:
    x = make_tensor(n, backend)
    y = x.replace(feats=x.feats + 1)

    def run() -> object:
        return cat([x + y, y])

    return run


def bench_prune_topk(n: int, backend: str) -> Callable[[], object]:
    x = make_tensor(n, backend, channels=1)
    keep = max(1, n // 2)

    def run() -> object:
        return prune(x, topk_rows(x, [keep]))

    return run


def bench_generative_cold(n: int, backend: str) -> Callable[[], object]:
    coords = make_coords(n, backend=backend)
    feats = make_feats(n, 8)
    weight = make_weight(8, 8, 8)

    def run() -> object:
        return generative_conv_transpose3d(
            SparseTensor(coords, feats, stride=2),
            weight,
            kernel_size=2,
            stride=2,
        )

    return run


def bench_generative_hot(n: int, backend: str) -> Callable[[], object]:
    x = SparseTensor(
        make_coords(n, backend=backend), make_feats(n, 8), stride=2
    )
    weight = make_weight(8, 8, 8)
    return lambda: generative_conv_transpose3d(
        x, weight, kernel_size=2, stride=2
    )


def bench_encoder_block(n: int, backend: str) -> Callable[[], object]:
    coords = make_coords(n, backend=backend)
    feats = make_feats(n, 8)
    w0 = make_weight(27, 8, 16)
    w1 = make_weight(8, 16, 16)

    def run() -> object:
        x = SparseTensor(coords, feats)
        return relu(
            conv3d(
                relu(conv3d(x, w0, kernel_size=3)),
                w1,
                kernel_size=2,
                stride=2,
            )
        )

    return run


def bench_decoder_block(n: int, backend: str) -> Callable[[], object]:
    coords = make_coords(n, backend=backend)
    feats = make_feats(n, 8)
    up = make_weight(8, 8, 8)
    cls = make_weight(1, 8, 1)

    def run() -> object:
        x = SparseTensor(coords, feats, stride=2)
        y = relu(generative_conv_transpose3d(x, up))
        logits = conv3d(y, cls, kernel_size=1)
        return prune(y, topk_rows(logits, [n]))

    return run


def cases() -> tuple[Case, ...]:
    return (
        Case('spdownsample', 'k2', bench_downsample),
        Case('FOG.forward', 'pool-hot', bench_pool_hot),
        Case('FOG.forward', 'pool-cold', bench_pool_cold),
        Case('max_pool3d', 'hot', bench_max_pool_hot),
        Case('max_pool3d', 'cold', bench_max_pool_cold),
        Case('conv3d_k3s1', 'hot', bench_conv_hot),
        Case('conv3d_k3s1', 'cold', bench_conv_cold),
        Case('conv3d_k1s1', 'pointwise', bench_pointwise),
        Case('feature_chain', 'linear-relu', bench_feature_chain),
        Case('tensor_algebra', 'cat-add', bench_cat_add),
        Case('prune', 'topk', bench_prune_topk),
        Case('gen_tconv_k2s2', 'hot', bench_generative_hot),
        Case('gen_tconv_k2s2', 'cold', bench_generative_cold),
        Case('encoder_block', 'cold', bench_encoder_block),
        Case('decoder_block', 'cold', bench_decoder_block),
    )


def run_case(
    case: Case,
    *,
    backend: str,
    n: int,
    warmup: int,
    repeat: int,
) -> Result:
    mx.set_default_device(
        mx.Device(mx.gpu if backend in {'cuda', 'metal'} else mx.cpu)
    )
    fn = case.factory(n, backend)
    times = measure(fn, warmup=warmup, repeat=repeat)
    return Result(case.name, case.variant, backend, n, times)


def print_results(results: Sequence[Result]) -> None:
    print('=' * 88)
    print('Summary (median ms, Q1-Q3)')
    print('=' * 88)
    print(
        f'{"Test":<24} {"Var":<14} {"Backend":<8} {"N":>8} '
        f'{"median_ms":>10} {"Q1_ms":>8} {"Q3_ms":>8}'
    )
    print('-' * 88)
    for result in results:
        print(
            f'{result.test:<24} {result.variant:<14} {result.backend:<8} '
            f'{result.n:>8} {result.median_ms:>10.4f} '
            f'{result.q1_ms:>8.4f} {result.q3_ms:>8.4f}'
        )
    print('=' * 88)


def write_csv(path: Path, results: Sequence[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'test',
                'variant',
                'backend',
                'n',
                'median_ms',
                'q1_ms',
                'q3_ms',
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    'test': result.test,
                    'variant': result.variant,
                    'backend': result.backend,
                    'n': result.n,
                    'median_ms': result.median_ms,
                    'q1_ms': result.q1_ms,
                    'q3_ms': result.q3_ms,
                }
            )


def selected_backends(value: str) -> tuple[str, ...]:
    if value == 'all':
        backends = ['cpu']
        if mx.metal.is_available():
            backends.append('metal')
        if mx.cuda.is_available():
            backends.append('cuda')
        return tuple(backends)
    if value == 'metal' and not mx.metal.is_available():
        raise RuntimeError('Metal is not available.')
    if value == 'cuda' and not mx.cuda.is_available():
        raise RuntimeError('CUDA is not available.')
    return (value,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scales',
        type=int,
        nargs='+',
        default=[1000, 5000, 25000, 100000],
    )
    parser.add_argument(
        '--backend', choices=['all', 'cpu', 'metal', 'cuda'], default='all'
    )
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=9)
    parser.add_argument('--csv', type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for backend in selected_backends(args.backend):
        for n in args.scales:
            for case in cases():
                result = run_case(
                    case,
                    backend=backend,
                    n=n,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
                results.append(result)
                print(
                    f'{result.test:<24} {result.variant:<14} '
                    f'{backend:<8} N={n:<8} median={result.median_ms:.4f} ms'
                )
    print_results(results)
    if args.csv is not None:
        write_csv(args.csv, results)


if __name__ == '__main__':
    main()
