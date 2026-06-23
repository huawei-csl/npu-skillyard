# Benchmark Script Patterns

Reference patterns for generating BenchmarkScript that measures device-side latency
of PTO kernels with `torch.npu.Event` timing and an L2-cache flush -- the rigorous,
comparison-grade method (it matches how production/hand-tuned kernels are benchmarked,
so numbers are comparable by construction).

## Timing method (standard)

- **Timer:** `torch.npu.Event(enable_timing=True)` pairs, device-only. Record one
  start/end pair per iteration on the stream, run them all, then `torch.npu.synchronize()`
  ONCE and read `start.elapsed_time(end)` (milliseconds) for each pair. Do NOT wrap a
  host `time.perf_counter()` around `sync -> launch -> sync`: that includes Python/ctypes
  dispatch and inflates small-kernel latency.
- **L2 flush:** allocate a 256 MiB `int8` scratch ONCE and `.zero_()` it before EVERY
  timed call so reps do not benefit from cache residency. (A 256 MiB `int8` `.zero_()`
  per iteration is safe and standard -- it does not deadlock the driver.)
- **Synchronize is REQUIRED here.** Unlike the ValidationScript (which must avoid
  `torch.npu.synchronize()` so it can run under the msprof simulator), the
  BenchmarkScript NEVER runs under msprof -- it runs only on real NPU. Event timing
  needs the final `synchronize()` to read `elapsed_time`.
- **Shapes:** benchmark at the contract's PRODUCTION sweep (`shape_contract.sweep_axis`
  / `production_dimensions`), never the tiny `--sim-mode` dims. A latency number at a
  non-production shape is meaningless for comparison.
- **Parameterizable knobs:** `--timer {event,wallclock}`, `--warmup`, `--repeats`,
  `--flush-mib` so the harness can be set to MATCH an external baseline's method exactly.
  Defaults: `event`, warmup 5, repeats 15, flush 256 MiB.

## Complete BenchmarkScript Template

```python
#!/usr/bin/env python3
"""Benchmark script for {stage_name} kernel.

Measures device-side latency with torch.npu.Event + a 256 MiB L2 flush per iter.
Reports statistics in nanoseconds.
"""

import torch
import torch_npu  # noqa: F401
import ctypes
import argparse
import os
import sys
import json
import statistics

# ============================================================================
# Device setup
# ============================================================================
torch.npu.set_device(0)
device = torch.device("npu")

# ============================================================================
# Kernel launch wrapper
# ============================================================================
def setup_kernel(so_path, stage_spec):
    """Load kernel .so by ABSOLUTE path and configure ctypes interface."""
    lib = ctypes.CDLL(os.path.abspath(so_path))
    call_kernel = lib.call_kernel
    # Define argtypes from StageSpec.abi.arguments (exact order)
    call_kernel.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        # ... tensor pointers ...
        ctypes.c_int64,   # total_work
        # ... problem dimensions ...
    ]
    call_kernel.restype = None
    return call_kernel

def call_kernel_wrapper(call_kernel, input_tensors, output_tensors, stage_spec, n_seq, l_seg):
    """Call kernel with the argument ordering from StageSpec.abi."""
    stream = torch.npu.current_stream()
    stream_ptr = getattr(stream, "_as_parameter_", None)

    problem = stage_spec['stage']['problem']
    bt = problem.get('BT')
    k = problem.get('K')
    hv = problem.get('HV')

    B = 1
    prod_dims = stage_spec['stage'].get('production_dimensions', {})
    hv_val = prod_dims.get('HV', hv if hv is not None else 1)
    total_work = B * hv_val * n_seq * l_seg
    block_dim = min(total_work, 256)

    # Force contiguous immediately before taking data_ptr (kernel reads raw row-major)
    tensors = [t.contiguous() for t in (input_tensors + output_tensors)]
    tensor_ptrs = [t.data_ptr() for t in tensors]

    call_kernel(
        block_dim,
        stream_ptr,
        *tensor_ptrs,
        total_work,
        l_seg,  # bt
        k,
        # ... other problem dimensions in StageSpec.abi order ...
    )

# ============================================================================
# Shape derivation from StageSpec
# ============================================================================
def resolve_shape(symbolic_shape, stage_spec, n_seq, l_seg):
    """Resolve symbolic shape like ['B','HV','NT','BT','K'] to a numeric tuple."""
    problem = stage_spec['stage']['problem']
    dim_map = {'B': 1, 'T': n_seq * l_seg, 'NT': n_seq, 'BT': l_seg}
    dim_map.update(problem)
    dim_map['NT'] = n_seq
    dim_map['BT'] = l_seg
    return tuple(dim_map[d] if isinstance(d, str) else d for d in symbolic_shape)

def allocate_io_tensors(stage_spec, n_seq, l_seg):
    """Allocate input/output tensors from StageSpec shapes (contract dtype)."""
    inputs_spec = stage_spec['stage']['inputs']
    outputs_spec = stage_spec['stage']['outputs']
    input_npu = []
    for inp in inputs_spec:
        shape = resolve_shape(inp['shape'], stage_spec, n_seq, l_seg)
        dtype = getattr(torch, inp.get('dtype', 'float32'))
        input_npu.append(torch.randn(shape, dtype=dtype, device=device).contiguous())
    output_npu = []
    for out in outputs_spec:
        shape = resolve_shape(out['shape'], stage_spec, n_seq, l_seg)
        dtype = getattr(torch, out.get('dtype', 'float32'))
        output_npu.append(torch.zeros(shape, dtype=dtype, device=device).contiguous())
    return input_npu, output_npu

# ============================================================================
# Device-event timing with L2 flush (standard)
# ============================================================================
def _stats_ns(samples_ns):
    s = sorted(samples_ns)
    n = len(s)
    return {
        'mean_ns': int(statistics.mean(s)),
        'min_ns': int(s[0]),
        'max_ns': int(s[-1]),
        'median_ns': int(statistics.median(s)),
        'p95_ns': int(s[min(n - 1, int(0.95 * n))]),
        'stddev_ns': int(statistics.stdev(s)) if n > 1 else 0,
    }

def benchmark_kernel(call_kernel, stage_spec, n_seq, l_seg,
                     warmup=5, repeats=15, timer='event', flush_mib=256):
    """Measure kernel latency. timer='event' (default) uses torch.npu.Event;
    timer='wallclock' is a per-iteration perf_counter fallback. Both flush a
    `flush_mib` MiB L2 scratch before every timed call. Returns stats in ns."""
    input_npu, output_npu = allocate_io_tensors(stage_spec, n_seq, l_seg)
    # Inputs/workspaces must be ready before the first launch (avoid a copy/launch race)
    torch.npu.synchronize()

    def run():
        call_kernel_wrapper(call_kernel, input_npu, output_npu, stage_spec, n_seq, l_seg)

    cache = (torch.empty(flush_mib * 1024 * 1024, dtype=torch.int8, device=device)
             if flush_mib else None)

    for _ in range(warmup):
        run()
    torch.npu.synchronize()

    if timer == 'event':
        starts = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
        for i in range(repeats):
            if cache is not None:
                cache.zero_()
            starts[i].record()
            run()
            ends[i].record()
        torch.npu.synchronize()
        samples_ns = [starts[i].elapsed_time(ends[i]) * 1e6 for i in range(repeats)]  # ms -> ns
    else:  # wallclock fallback (per-iteration)
        import time
        samples_ns = []
        for _ in range(repeats):
            if cache is not None:
                cache.zero_()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            run()
            torch.npu.synchronize()
            samples_ns.append((time.perf_counter() - t0) * 1e9)

    return _stats_ns(samples_ns)

# ============================================================================
# Production sweep support
# ============================================================================
def benchmark_sweep(call_kernel, stage_spec, n_seq, bt_list,
                    warmup=5, repeats=15, timer='event', flush_mib=256):
    """Sweep the contract's production sizes; results keyed by str(BT)."""
    results = {}
    for bt in bt_list:
        print(f"Benchmarking BT={bt}...")
        results[str(bt)] = benchmark_kernel(call_kernel, stage_spec, n_seq, bt,
                                            warmup, repeats, timer, flush_mib)
    return results

def work_units(stage_spec, n_seq, bt):
    """Repeated work count for this stage (chunks/tiles/blocks). Algorithm-agnostic:
    units = problem_size / tile_size. Default = n_seq (chunk count); override per stage."""
    return max(1, int(n_seq))

def fit_slope(results, stage_spec, n_seq):
    """Per-work-unit slope from the two extreme sweep points (the HEADLINE metric).
    slope = (lat@largest - lat@smallest) / (units@largest - units@smallest)."""
    pts = []
    for bt_str, stats in results.items():
        bt = int(bt_str)
        pts.append((bt, work_units(stage_spec, n_seq, bt), stats["median"]))
    pts.sort(key=lambda p: p[1])           # by units
    if len(pts) < 2 or pts[-1][1] == pts[0][1]:
        return {"slope_per_unit_ns": None, "note": "need >=2 distinct-unit sweep points", "points": pts}
    lo, hi = pts[0], pts[-1]
    slope = (hi[2] - lo[2]) / (hi[1] - lo[1])
    return {"slope_per_unit_ns": slope,
            "points": [{"size": p[0], "units": p[1], "median_ns": p[2]} for p in pts]}

def paired_ab(call_a, call_b, stage_spec, n_seq, bt_list, warmup, repeats, timer, flush_mib):
    """Within-process paired A/B: time A and B ALTERNATELY per size so common-mode device
    drift cancels. Returns per-size paired median deltas (B is the --baseline-so)."""
    out = {}
    for bt in bt_list:
        a = benchmark_kernel(call_a, stage_spec, n_seq, bt, warmup, repeats, timer, flush_mib)
        b = benchmark_kernel(call_b, stage_spec, n_seq, bt, warmup, repeats, timer, flush_mib)
        a2 = benchmark_kernel(call_a, stage_spec, n_seq, bt, warmup, repeats, timer, flush_mib)
        b2 = benchmark_kernel(call_b, stage_spec, n_seq, bt, warmup, repeats, timer, flush_mib)
        a_med = statistics.median([a["median"], a2["median"]])
        b_med = statistics.median([b["median"], b2["median"]])
        out[str(bt)] = {"this_median_ns": a_med, "baseline_median_ns": b_med,
                        "speedup_x": (b_med / a_med) if a_med else None}
    return out

# ============================================================================
# CLI and main
# ============================================================================
# Embed the StageSpec so the script is self-contained; --stage-spec overrides it.
STAGE_SPEC = {
    "stage": {
        "name": "{stage_name}",
        "problem": {{"K": 128, "V": 128}},
        "production_dimensions": {{"HV": 256}},
        "inputs": [],
        "outputs": []
    }
}

def main():
    parser = argparse.ArgumentParser(description='Benchmark {stage_name} kernel')
    parser.add_argument('input', help='Path to kernel .so')
    parser.add_argument('--n-seq', type=int, default=16, help='Number of sequences (NT)')
    parser.add_argument('--l-seg', type=int, default=128, help='Segment length (BT)')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations')
    parser.add_argument('--repeats', type=int, default=15, help='Timed iterations')
    parser.add_argument('--timer', choices=['event', 'wallclock'], default='event',
                        help='Timing method (event = torch.npu.Event, default)')
    parser.add_argument('--flush-mib', type=int, default=256,
                        help='L2 flush scratch size in MiB (0 disables)')
    parser.add_argument('--out-json', type=str, help='Output JSON path (default: stdout)')
    parser.add_argument('--l-seg-list', type=str, help='Comma-separated production BT values for the sweep')
    parser.add_argument('--stage-spec', required=False, default=None, help='Path to StageSpec JSON (overrides embedded)')
    parser.add_argument('--baseline-so', type=str, default=None,
                        help='Second .so for a within-process paired A/B comparison')
    args = parser.parse_args()

    if args.stage_spec:
        with open(args.stage_spec, 'r') as f:
            stage_spec = json.load(f)
    else:
        stage_spec = STAGE_SPEC

    call_kernel = setup_kernel(args.input, stage_spec)

    # Always sweep >=2 sizes so the per-work-unit slope (the headline) is well-defined.
    if args.l_seg_list:
        bt_list = [int(x) for x in args.l_seg_list.split(',')]
    else:
        bt_list = sorted({args.l_seg, args.l_seg * 2})   # minimal 2-point sweep
    by_size = benchmark_sweep(call_kernel, stage_spec, args.n_seq, bt_list,
                              args.warmup, args.repeats, args.timer, args.flush_mib)
    results = {"by_size": by_size, "slope": fit_slope(by_size, stage_spec, args.n_seq)}

    if args.baseline_so:
        call_base = setup_kernel(args.baseline_so, stage_spec)
        results["paired"] = paired_ab(call_kernel, call_base, stage_spec, args.n_seq,
                                      bt_list, args.warmup, args.repeats, args.timer, args.flush_mib)

    json_str = json.dumps(results, indent=2)
    if args.out_json:
        with open(args.out_json, 'w') as f:
            f.write(json_str)
        print(f"Results written to {args.out_json}")
    else:
        print(json_str)

if __name__ == "__main__":
    main()
```

## Device-Event Timing Pattern (standard)

```python
cache = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)  # allocate once
for _ in range(warmup):
    run()
torch.npu.synchronize()

starts = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
ends = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
for i in range(repeats):
    cache.zero_()          # flush L2 before each timed call
    starts[i].record()
    run()
    ends[i].record()
torch.npu.synchronize()    # required to read elapsed_time
samples_ns = [starts[i].elapsed_time(ends[i]) * 1e6 for i in range(repeats)]  # ms -> ns
```

**Critical**:
- Allocate the flush scratch ONCE; `.zero_()` it each iteration.
- Record all event pairs, then `synchronize()` ONCE (not per iteration).
- `elapsed_time` returns milliseconds; multiply by 1e6 for nanoseconds.
- Benchmark at production sizes from the contract, not toy dims.

## Statistics Computation

```python
import statistics
samples_ns = [...]  # per-iteration device-timed measurements
s = sorted(samples_ns); n = len(s)
stats = {
    'mean_ns': int(statistics.mean(s)),
    'min_ns': int(s[0]),
    'max_ns': int(s[-1]),
    'median_ns': int(statistics.median(s)),
    'p95_ns': int(s[min(n - 1, int(0.95 * n))]),
    'stddev_ns': int(statistics.stdev(s)) if n > 1 else 0,
}
```

## JSON Output Format

```json
{
  "mean_ns": 123456,
  "min_ns": 120000,
  "max_ns": 130000,
  "median_ns": 123000,
  "p95_ns": 128000,
  "stddev_ns": 3500
}
```

**All fields required**: mean_ns, min_ns, max_ns, median_ns, p95_ns, stddev_ns.
For a sweep, key by `str(BT)`: `results[str(BT)] = stats`.

## Matching an external baseline

When the run's purpose is a head-to-head comparison, set the knobs to the baseline's
exact method (e.g. `--timer event --flush-mib 256 --warmup 5 --repeats 15`) and the
SAME production shapes. Comparisons are only valid when both sides use the same timer,
flush, iteration count, dtype, and shape.

## Common Pitfalls

1. **Host wall-clock as the default**: `perf_counter` around `sync->launch->sync`
   includes Python/ctypes dispatch and inflates sub-ms kernels. Use `torch.npu.Event`.
2. **One event pair around a batch**: record a pair PER iteration, then sync once.
   A single pair around a batch hides per-iteration variance.
3. **Forgetting the final synchronize**: `elapsed_time` is only valid after
   `torch.npu.synchronize()`.
4. **No cache flush**: without the per-iter `.zero_()` flush, reps benefit from L2
   residency and under-report latency. The 256 MiB `int8` flush is safe and standard.
5. **Benchmarking toy dims**: derive sizes from the contract's production sweep, not
   `--sim-mode` minimums.
6. **Integer keys in results dict**: use `results[str(BT)]`, not `results[BT]`.
7. **Non-contiguous launch tensors**: force `.contiguous()` before `.data_ptr()`.
