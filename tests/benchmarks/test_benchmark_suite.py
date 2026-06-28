from __future__ import annotations

import json

from mlx_lattice_bench.cases import GROUPS, all_cases
from mlx_lattice_bench.datasets import (
    SPARSE_LAYOUTS,
    PointArrays,
    point_arrays,
)
from mlx_lattice_bench.harness import BenchmarkCase, run_case, run_cases
from mlx_lattice_bench.report import write_json, write_summary
from mlx_lattice_bench.run import _parser, _report_paths

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
        'tensor',
        'workloads',
    }
    assert 'voxelize_mean' in names
    assert 'voxelize_mean_fixed' in names
    assert 'conv3d_generic' in names
    assert 'conv3d_generic_density' in names
    assert 'conv3d_generic_dfeatures' in names
    assert 'conv3d_generic_dweight' in names
    assert 'prune_mask' in names
    assert 'workload_mini_encoder' in names
    assert 'workload_mini_encoder_fixed' in names
    assert cases[0].params[0]['N'] == 256


def test_conv_density_case_matches_wide_layout_vocabulary() -> None:
    cases = all_cases(
        'smoke',
        groups=('conv',),
        n_values=(32,),
        channels=(4,),
    )
    case = next(
        item for item in cases if item.name == 'conv3d_generic_density'
    )

    assert (
        tuple(params['layout'] for params in case.params) == SPARSE_LAYOUTS
    )
    assert {params['channels'] for params in case.params} == {4}


def test_conv_density_case_reports_calculated_kernel_density() -> None:
    cases = all_cases(
        'smoke',
        groups=('conv',),
        n_values=(32,),
        channels=(4,),
    )
    case = next(
        item for item in cases if item.name == 'conv3d_generic_density'
    )
    params = next(
        params for params in case.params if params['layout'] == 'line'
    )

    result = run_case(
        case,
        params,
        mode='hot_op',
        device='cpu',
        warmup=0,
        repeats=1,
    )

    assert result is not None
    assert result.params['layout'] == 'line'
    assert result.workload['edges'] > 0
    assert result.workload['target_active_kernel_positions'] > 0.0
    assert result.workload['target_kernel_density'] > 0.0
    assert result.workload['expected_active_kernel_positions'] == 3.0
    assert result.workload['expected_kernel_density'] == 3.0 / 27.0


def test_quantized_conv_cases_use_packed_inference_and_report_storage() -> (
    None
):
    cases = all_cases(
        'smoke',
        groups=('conv',),
        n_values=(8,),
        channels=(32,),
        dtype='int4',
    )
    assert not any(case.name.endswith('dweight') for case in cases)
    case = next(item for item in cases if item.name == 'conv3d_generic')

    result = run_case(
        case,
        case.params[0],
        mode='hot_op',
        device='cpu',
        warmup=0,
        repeats=1,
    )

    assert result is not None
    assert result.params['dtype'] == 'int4'
    assert result.workload['weight_storage_bytes'] > 0
    assert result.workload['weight_compression_ratio'] > 1.0
    assert case.supports('compiled_hot')
    assert not case.supports('backward')


def test_harness_runs_cold_and_hot_public_operator_cases() -> None:
    case = BenchmarkCase(
        name='test_voxelize',
        group='quantization',
        params=({'N': 8, 'channels': 2},),
        setup=lambda params: point_arrays(
            rows=int(params['N']),
            channels=int(params['channels']),
        ),
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
        params=({'N': 4, 'channels': 1},),
        setup=lambda params: point_arrays(
            rows=int(params['N']),
            channels=int(params['channels']),
        ),
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
            (item.name, params['N'], mode, device)
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
        params=({'N': 4, 'channels': 1},),
        setup=lambda params: point_arrays(
            rows=int(params['N']),
            channels=int(params['channels']),
        ),
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
            (item.name, params['N'], mode, device)
        ),
    )

    assert skips == [('test_voxelize', 4, 'compiled_hot', 'cpu')]
    assert results == []


def test_harness_can_scope_case_to_specific_modes() -> None:
    case = BenchmarkCase(
        name='test_backward_only',
        group='quantization',
        params=({'N': 4, 'channels': 1},),
        setup=lambda params: point_arrays(
            rows=int(params['N']),
            channels=int(params['channels']),
        ),
        prepare=lambda fixture: fixture,
        run=_run_voxelize,
        backward=lambda fixture: (
            lambda feats: mx.sum(feats),
            (fixture.feats,),
        ),
        modes=('backward',),
    )

    assert not case.supports('hot_op')
    assert case.supports('backward')


def test_cli_explicit_modes_do_not_append_default_modes() -> None:
    args = _parser().parse_args(['--mode', 'backward'])

    assert args.mode == ['backward']


def test_cli_accepts_packed_quantized_weight_dtypes() -> None:
    assert _parser().parse_args(['--dtype', 'int4']).dtype == 'int4'
    assert _parser().parse_args(['--dtype', 'int8']).dtype == 'int8'


def test_cli_repeated_size_values_define_custom_n_sweep() -> None:
    args = _parser().parse_args(['--size', '16', '--size', '32'])

    assert args.n_values == [16, 32]
    cases = all_cases(
        'smoke', groups=('coords',), n_values=tuple(args.n_values)
    )
    assert [params['N'] for params in cases[0].params] == [16, 32]


def test_cli_report_paths_resolve_under_results_dir() -> None:
    args = _parser().parse_args(['--output', 'smoke'])

    json_path, summary_path = _report_paths(args)

    assert json_path.as_posix() == 'benchmarks/results/smoke.json'
    assert summary_path.as_posix() == 'benchmarks/results/smoke.summary.txt'


def test_json_report_contains_environment_and_samples(tmp_path) -> None:
    case = BenchmarkCase(
        name='test_voxelize',
        group='quantization',
        params=({'N': 4, 'channels': 1},),
        setup=lambda params: point_arrays(
            rows=int(params['N']),
            channels=int(params['channels']),
        ),
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

    summary_path = tmp_path / 'bench.summary.txt'
    write_summary(summary_path, results=[result])
    summary = summary_path.read_text()
    assert 'test_voxelize' in summary
    assert 'median_ms' in summary


def _run_voxelize(fixture: PointArrays) -> object:
    assert fixture.points.dtype == mx.float32
    return voxelize(
        fixture.points,
        fixture.feats,
        batch_indices=fixture.batch_indices,
    )
