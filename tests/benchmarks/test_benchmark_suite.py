from __future__ import annotations

import json

from mlx_lattice_bench.cases import GROUPS, all_cases
from mlx_lattice_bench.datasets import PointArrays, point_arrays
from mlx_lattice_bench.harness import BenchmarkCase, run_case, run_cases
from mlx_lattice_bench.report import write_json
from mlx_lattice_bench.run import _parser

from mlx_lattice.ops import voxelize
from tests.support import mx


def test_case_catalog_exposes_expected_public_surface_groups() -> None:
    cases = all_cases('smoke')
    names = {case.name for case in cases}

    assert set(GROUPS) == {
        'quantization',
        'coords',
        'relations',
        'conv',
        'pool',
        'feature',
        'workloads',
    }
    assert 'voxelize_mean' in names
    assert 'conv3d_generic' in names
    assert 'workload_mini_encoder' in names


def test_harness_runs_cold_and_hot_public_operator_cases() -> None:
    case = BenchmarkCase(
        name='test_voxelize',
        group='quantization',
        params=({'rows': 8, 'points': 8, 'channels': 2},),
        setup=lambda _: point_arrays(rows=8, channels=2),
        prepare=lambda fixture: fixture,
        run=_run_voxelize,
        units=('points',),
    )

    cold = run_case(
        case,
        case.params[0],
        mode='cold_op',
        device='cpu',
        warmup=1,
        repeats=2,
    )
    hot = run_case(
        case,
        case.params[0],
        mode='hot_op',
        device='cpu',
        warmup=1,
        repeats=2,
    )

    assert cold is not None
    assert hot is not None
    assert cold.repeats == 2
    assert hot.repeats == 2
    assert cold.workload['points'] == 8
    assert cold.workload['n_out'] > 0
    assert cold.units['points_per_s'] > 0.0
    assert hot.median_ms >= 0.0


def test_harness_reports_progress_on_shared_case_layer() -> None:
    case = BenchmarkCase(
        name='test_voxelize',
        group='quantization',
        params=({'rows': 4, 'points': 4, 'channels': 1},),
        setup=lambda _: point_arrays(rows=4, channels=1),
        prepare=lambda fixture: fixture,
        run=_run_voxelize,
        units=('points',),
    )
    starts = []
    finishes = []

    results = run_cases(
        [case],
        modes=('hot_op',),
        device='cpu',
        warmup=0,
        repeats=1,
        on_start=lambda item, params, mode, device: starts.append(
            (item.name, params['rows'], mode, device)
        ),
        on_result=lambda result, *_: finishes.append(result.case),
    )

    assert starts == [('test_voxelize', 4, 'hot_op', 'cpu')]
    assert finishes == ['test_voxelize']
    assert len(results) == 1


def test_harness_reports_unsupported_mode_skips() -> None:
    case = BenchmarkCase(
        name='test_voxelize',
        group='quantization',
        params=({'rows': 4, 'points': 4, 'channels': 1},),
        setup=lambda _: point_arrays(rows=4, channels=1),
        prepare=lambda fixture: fixture,
        run=_run_voxelize,
    )
    skips = []

    results = run_cases(
        [case],
        modes=('compiled_hot',),
        device='cpu',
        warmup=0,
        repeats=1,
        on_skip=lambda item, params, mode, device: skips.append(
            (item.name, params['rows'], mode, device)
        ),
    )

    assert skips == [('test_voxelize', 4, 'compiled_hot', 'cpu')]
    assert results == []


def test_cli_explicit_modes_do_not_append_default_modes() -> None:
    args = _parser().parse_args(['--mode', 'backward'])

    assert args.mode == ['backward']


def test_json_report_contains_environment_and_samples(tmp_path) -> None:
    case = BenchmarkCase(
        name='test_voxelize',
        group='quantization',
        params=({'rows': 4, 'points': 4, 'channels': 1},),
        setup=lambda _: point_arrays(rows=4, channels=1),
        prepare=lambda fixture: fixture,
        run=_run_voxelize,
        units=('points',),
    )
    result = run_case(
        case,
        case.params[0],
        mode='hot_op',
        device='cpu',
        warmup=1,
        repeats=1,
    )
    assert result is not None

    path = tmp_path / 'bench.json'
    write_json(path, results=[result])
    payload = json.loads(path.read_text())

    assert 'environment' in payload
    assert payload['results'][0]['case'] == 'test_voxelize'
    assert payload['results'][0]['workload']['points'] == 4
    assert payload['results'][0]['samples_ms']


def _run_voxelize(fixture: PointArrays) -> object:
    assert fixture.points.dtype == mx.float32
    return voxelize(
        fixture.points,
        fixture.feats,
        batch_indices=fixture.batch_indices,
    )
