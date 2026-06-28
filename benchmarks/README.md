# mlx-lattice Benchmarks

This workspace package benchmarks the public Python surface of `mlx-lattice`.
It intentionally does not call private native bindings or backend helpers.

Run from the repository root:

```bash
uv run --all-packages mlx-lattice-bench --preset smoke
```

Useful variants:

```bash
uv run --all-packages mlx-lattice-bench --device all
uv run --all-packages mlx-lattice-bench --group conv --group pool
uv run --all-packages mlx-lattice-bench --mode compiled_hot
uv run --all-packages mlx-lattice-bench --mode backward
uv run --all-packages mlx-lattice-bench --size 1000 --size 5000
uv run --all-packages mlx-lattice-bench --group conv --dtype int4
uv run --all-packages mlx-lattice-bench --group conv --dtype int8
uv run --all-packages mlx-lattice-bench --output smoke.json
```

After `uv sync --all-packages`, the shorter form also works:

```bash
uv run mlx-lattice-bench --device all
```

## Measurement Contract

Every benchmark case uses the same layer:

- public functions from `mlx_lattice.ops`;
- public sparse objects from `mlx_lattice.core`;
- `mlx_lattice.nn` only in future module-level workload cases;
- no direct `mlx_lattice._native.ext` calls;
- no private coordinate manager or relation internals.

For single-operator cases, generated inputs are evaluated before timing. The
timed region runs the public operation and then forces MLX evaluation with
`mx.eval` over the public output arrays. This keeps lazy input construction out
of the latency sample while still charging the operator for graph construction,
dispatch, execution, and output materialization.

For workload cases, only the final public output is evaluated. The suite does
not insert barriers between internal operations, so MLX can schedule a
semi-real pipeline as it would in application code.

For `compiled_hot` and `backward`, latency samples time the transformed
function. Effective workload metadata is collected from an untimed semantic
public-op run after the samples, because transformed functions may return raw
arrays or gradients rather than the public sparse object.

## Modes

- `cold_op`: reconstruct public Python objects from pre-generated arrays each
  iteration, then run and evaluate the op. This includes Python API dispatch
  and graph-node construction, but not random data generation.
- `hot_op`: reuse prepared public objects after warmup. This measures
  steady-state public operator calls.
- `compiled_hot`: compile an array-returning public-op function with
  `mx.compile`, warm it first, then time steady-state calls.
- `backward`: run MLX gradient transforms for differentiable feature paths.

Current fused sparse convolution and pooling rebuild their logical sparse map
inside the native primitive. The suite reports that honestly; it does not
claim map-reuse timing until the public API exposes a reusable semantic plan.

## Case Groups

- `quantization`: `sparse_quantize`, `voxelize`.
- `coords`: coordinate set and lookup operations.
- `relations`: kernel, generative, transposed, KNN, and radius relations.
- `conv`: convolution public modes.
- `pool`: local and global pooling modes.
- `feature`: feature-only sparse tensor transforms.
- `workloads`: semi-real pipelines composed from public ops.

## Reports

Console output streams per-case progress, including effective sparse workload
metadata after each completed result. Final benchmark reports are written under
`benchmarks/results`, which is ignored by git. Each run writes a JSON report and
a plain text summary table. If `--output` is omitted, the suite creates a
timestamped JSON filename; if `--output` is relative, it resolves below
`benchmarks/results`. ANSI color is automatic for terminals; use
`--color always` or `--color never` to override it. Editable native-extension
rebuild output is hidden during benchmark startup by default; pass
`--show-build-log` when diagnosing build/install behavior.

Sparse workload metadata is collected after timed evaluation, so host reads of
lazy scalar counts do not affect latency samples. The key fields are:

- `N`: planned input size requested by the benchmark parameter sweep.
- `P`: raw point rows before quantization.
- `Nin`: effective sparse input rows.
- `Nout`: effective sparse output rows.
- `E`: relation/map edge count when the public result exposes it.
- `Cin` and `Cout`: input and output feature channels.
- `K`: kernel volume when known.
- `avgN`: average relation neighbors, `E / Nout` or `E / query rows`.
- `weight_storage_bytes`: packed weight plus affine scale/bias storage.
- `weight_compression_ratio`: equivalent FP16 bytes divided by packed bytes.

For convolution, `--dtype int4` and `--dtype int8` mean packed affine weights
with float16 activations. Quantization happens during fixture setup, outside
the timed region. Packed cases are inference-only and therefore do not expose
the `backward` mode.

JSON reports include:

- git SHA;
- Python, platform, MLX, and `mlx-lattice` versions;
- backend capabilities;
- case, group, mode, device, parameters;
- effective sparse workload metadata;
- warmup/repeat counts;
- all latency samples;
- median, min, p90, and p95 latency;
- derived throughput for declared `N`, point, sparse row, and edge counts.
