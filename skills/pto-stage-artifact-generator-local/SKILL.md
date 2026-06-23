---
name: pto-stage-artifact-generator-local
description: "Generate validation and benchmark scripts for a PTO stage kernel from StageSpec and ReferenceModel. Triggers: generate validation script, generate benchmark script, stage artifact generation, KDA kernel harness."
---

# PTO Stage Artifact Generator

Generate runnable validation and benchmark scripts for a single PTO stage kernel.

## Inputs

You receive two required file inputs from the runtime payload:

- `StageSpec` (`.json`): **Authoritative for mathematical intent**, output names, shape relations, stage semantics, production dimensions, evidence gaps, **and ABI details** (exported symbol name, argument order, ctypes mapping, launch style — see the ``abi`` field).
- `ReferenceModel` (`.py`): **Authoritative for the Python reference implementation** and deterministic case-builder helpers.

Use only the staged file inputs. Do not search the repo for more examples, templates, wrappers, or helper files. In particular, NEVER read or copy from any pre-existing hand-tuned reference kernel or any other generator's output (in this repository or a sibling/related one) -- those are a human oracle only; using them invalidates the result.

## Authority Rules

1. **StageSpec is authoritative for math and ABI**: shapes, dimensions, invariants, production values, evidence gaps, entrypoint symbol, argument order, ctypes types, launch kind (see the ``abi`` field).
2. **ReferenceModel is authoritative for reference implementation**: the Python function signature, computation logic, case builders.
3. Never invent dimensions, shapes, or invariants that are not justified by StageSpec.

## Goal

Generate two outputs:

1. **ValidationScript** (`.py`): Loads the kernel `.so` and checks numerical correctness against the ReferenceModel.
2. **BenchmarkScript** (`.py`): Loads the kernel `.so` and measures device-side latency with `torch.npu.Event` + an L2-cache flush.

Both scripts must be standalone, runnable on Ascend NPU, and use the exact launch ABI from StageSpec.abi.

## Generation Protocol

Follow these steps in order:

### Step 1: Parse StageSpec

Extract and note:
- `stage.name` and `stage_index`
- `stage.inputs` and `stage.outputs` (symbolic shapes, dtypes)
- `stage.problem` (concrete tile dimensions — e.g., BT, K, HV, H, V, or whatever keys this stage defines)
- `stage.production_dimensions` (if present — use these for production-scale tests)
- `stage.instruction_families` (for context, not script generation)
- `stage.evidence_gaps` (to document in scripts)

**Derive dimensions from StageSpec.problem, never hard-code them.**

### Step 2: Parse StageSpec.abi

Extract and note:
- `abi.entrypoint_symbol` (usually `call_kernel`)
- `abi.arguments` (ordered list with name, ctype, ctypes type)
- Argument order convention: `[block_dim, stream, ...tensors..., total_work, ...problem_dims]`

**Use StageSpec.abi.arguments verbatim for ctypes argtypes and call_kernel invocation.**

### Step 3: Parse ReferenceModel

Extract and note:
- Function name (usually `reference_model` or stage-specific name)
- Function signature (input tensor names, shapes)
- Import path or inline code

**The ValidationScript must import or inline the ReferenceModel function.**

### Step 4: Generate ValidationScript

Follow the template in `references/validation_patterns.md`:
1. Import torch, torch_npu, ctypes, argparse, sys, time
2. Define `reference_model()` by inlining or importing from ReferenceModel
3. Define `DEFAULT_CASES` by DERIVING the sweep from the contract, never a
   hard-coded subset. The list MUST include (a) small alignment/tail points and
   (b) **every production / largest point in the contract sweep** (`shape_contract.sweep_axis`
   or `StageSpec.production_dimensions`) -- at least 6 distinct sizes x 2 seeds.
   Validating only the small dims and omitting the production maximum yields a
   PASS that does not cover the shape the kernel actually ships at: a silent
   coverage hole. If the contract's top size is too large for a quick default,
   still include it (gate it behind `--num-tests` rather than dropping it).
4. Define `call_kernel()` wrapper using StageSpec.abi.arguments
5. Define `validate_case()` that:
   - Allocates NPU tensors with deterministic seeding
   - Computes the reference where it is safe for the run context (see below)
   - Calls kernel on NPU
   - Compares with the fp64 Frobenius relative-error gate (ftol=2e-3) -- see rule 15

   **Reference placement (NPU vs CPU) is context-dependent:**
   - Under `--sim-mode` (msprof simulator): compute the reference ON NPU and
     compare on-device. A `.cpu()` sync hangs the msprof simulator, so the
     simulator path must stay on-device.
   - On the real-NPU full sweep (NOT under msprof): for contraction-heavy
     references (matmul / einsum / `@`), compute the reference on CPU (move
     inputs with `.cpu()`, run the reference, compare against the kernel output
     copied back). The torch_npu caching allocator can intermittently alias and
     ZERO an on-NPU reference matmul's output, producing spurious FAILs that are
     a harness artifact, not a kernel bug. A CPU reference is deterministic and
     `.cpu()` is fine off-simulator. For pure elementwise/Vec references, an
     on-NPU reference is fine in both contexts.
   - **fp64 precision caveat:** the NPU has no float64 -- `.double()` on an NPU
     tensor SILENTLY downcasts to fp32 (a `Device do not support double dtype`
     warning). So an on-NPU reference is at best an fp32 baseline, even if the
     metric later casts to fp64. To get the TRUE fp64 baseline the strict
     Frobenius gate (rule 15) assumes, compute the reference on CPU in float64
     (`.cpu().double()` the inputs first). For an fp16 kernel an fp32 reference is
     usually close enough (it is still ~1e-7 vs the kernel's ~1e-3), but anything
     numerically sensitive -- matrix inverse / triangular solve, long
     accumulations -- must use the CPU-fp64 reference, not an on-NPU `.double()`.

   **Condition inputs for inverse / triangular-solve stages.** When the stage
   contains a matrix inverse or triangular solve (e.g. `(I - strict_lower(M))^-1`),
   the case builder MUST condition the inputs to the algorithm's real operating
   distribution -- i.e. apply whatever input normalization the model itself
   applies (for example, L2-normalizing the relevant feature vectors). With
   un-normalized unit-variance inputs the inverse can be ill-conditioned and BOTH
   the reference and the kernel blow up (values -> 1e+32). A JOINT
   reference+kernel blow-up is a conditioning signal, not a kernel bug --
   normalize the inputs before concluding the kernel is wrong. A blocked /
   Neumann-doubling Cube inverse has the SAME conditioning requirement -- validate
   it against the same L2-normalized reference with `assert_close`.
6. Define CLI parser with `--stage-spec` (required), `--n-seq`, `--l-seg`, `--num-tests`
7. Define `main()` that iterates DEFAULT_CASES and calls validate_case
8. Add `if __name__ == "__main__": main()`

**Self-check**: Verify the script contains all required patterns from `references/npu_launch_patterns.md`.

**Simulation compatibility**: The script must work with `msprof op simulator --soc-version=Ascend910B1`.
Under `--sim-mode` this means: (a) reference model runs on NPU, (b) no
`torch.npu.synchronize()` calls, (c) comparison is on-device without `.cpu()`
copies. (Off-simulator, the real-NPU sweep may use a CPU reference — see Step 4.)

**msprof results are not authoritative — real NPU is.** The simulator can
reproducibly report FAIL for some valid kernels (observed with multi-tile
dual-store patterns) while the identical dimensions PASS on real hardware. Treat
a `--sim-mode` PASS as a fast smoke test and a `--sim-mode` FAIL as a signal to
confirm on real NPU, not as a verdict. The real-NPU sweep is the source of truth.

A specific, common false-FAIL cause: the `--sim-mode` defaults use tiny feature
dims that can fall BELOW the hardware's 64-fp32 Vec-lane block. A kernel whose
Vec reductions operate on full 64-lane blocks is correct at a production width
(a multiple of 64) but reports FAIL in sim at the tiny width. So when a stage
does full-block Vec reductions, either pick `--sim-mode` dims that are
lane-aligned (a multiple of 64) or treat a sub-64-width sim FAIL as inconclusive
and defer to the real-NPU sweep.

**Two-tier validation (`--sim-mode`)**: The script MUST support a `--sim-mode` flag for
msprof simulator validation. When `--sim-mode` is set:
- Use tiny dimensions: BT=4, K=8, V=8, HV=1, NT=1, H=1 (or stage-appropriate minimums)
- Use `block_dim=1` (simulator is extremely slow with multiple blocks)
- Run only 1 test case
- Print `[SIM-MODE]` banner
- The `generate_default_cases()` function takes a `sim_mode` parameter
- The `call_kernel_wrapper()` function takes a `sim_mode` parameter and sets `block_dim=1`

Without `--sim-mode`, the script runs the full dimension sweep (6+ BT values x 2 seeds)
intended for real NPU hardware. The two-tier approach is:
1. `--sim-mode` with msprof simulator for fast correctness checks (~10 seconds)
2. Full sweep on real NPU hardware for alignment, tail, and scale validation

**Recurrent / loop-carried stages: emit a per-step comparison mode.** When the
reference contains a sequential loop whose state feeds the next iteration (a scan
or recurrence — final output depends on a carried tensor), a single
final-output RMSE cannot localize a bug: a small per-step error compounds, so by
the last step everything looks wrong with no clue where it started. Add an
optional `--per-step` (or `--dump-steps`) flag that runs the reference loop and
the kernel in lockstep and reports the max-abs diff at EACH step index. This
pinpoints the first divergent step (e.g. "exact at step 0, 5e-4 at step 1,
growing after") — the single most useful signal for diagnosing a loop-carried
state-update defect. Refactor the inlined `reference_model` so the per-step
intermediates are accessible (e.g. yield/collect per-iteration state) when the
stage is recurrent.

### Step 5: Generate BenchmarkScript

Follow the template in `references/benchmark_patterns.md`:
1. Import torch, torch_npu, ctypes, argparse, sys, time, json, statistics
2. Define `call_kernel()` wrapper using StageSpec.abi.arguments (same as ValidationScript)
3. Define `benchmark_kernel()` that:
   - Allocates NPU tensors at the contract's PRODUCTION sweep sizes (not toy dims)
   - Runs warmup iterations, then synchronizes once
   - Measures device-side latency with `torch.npu.Event` pairs (one per iter), flushing
     a 256 MiB int8 L2 scratch (`.zero_()`) before each timed call, then `synchronize()`
     ONCE and reads `elapsed_time` (ms -> ns). A `--timer wallclock` per-iteration
     fallback is allowed but not the default.
   - Computes statistics (mean, min, max, median, p95, stddev)
   - Computes the PER-WORK-UNIT SLOPE across the sweep (this is the headline, not the raw
     single-size median — see rule 27). With the >=2 sweep points, fit
     `slope = (lat@largest - lat@smallest) / (units@largest - units@smallest)`, where
     `units` is the stage's repeated work count derived from the contract
     (`units = problem_size / tile_size`, e.g. chunks = seq_len / chunk_len; tiles =
     rows / tile_rows). Report `slope_per_unit` + the `(size, units, median)` points used.
4. Define CLI parser with `--stage-spec`, `--n-seq`, `--l-seg`, `--warmup`, `--repeats`,
   `--timer {event,wallclock}`, `--flush-mib`, `--out-json`, `--l-seg-list`, `--baseline-so`
5. Define `main()` that:
   - Parses args
   - Calls benchmark_kernel
   - If `--baseline-so` is given, runs the PAIRED A/B (rule 28) and adds `paired` to the output
   - Writes JSON summary to --out-json or stdout
6. Add `if __name__ == "__main__": main()`

**Self-check**: Verify the script uses `torch.npu.Event` timing (event default) with a
per-iteration L2 flush, sweeps >=2 production sizes, reports `slope_per_unit` (not just a
single-size median), and supports `--baseline-so` for a within-process paired A/B.

### Step 6: Self-Check Before Return

Before returning the JSON output, verify:

- [ ] Both scripts accept `--stage-spec` (required) to load StageSpec from JSON file
- [ ] ValidationScript accepts `--sim-mode` flag for msprof simulator (tiny dims, block_dim=1)
- [ ] Both scripts contain `import torch_npu  # noqa: F401`
- [ ] Both scripts use `torch.device("npu")` (not `"npu:0"` or `"npu:1"`)
- [ ] Both scripts extract `stream_ptr = getattr(stream, "_as_parameter_", None)`
- [ ] ValidationScript does NOT call `torch.npu.synchronize()` (hangs in msprof op simulator); BenchmarkScript MUST synchronize (event timing requires it, and it never runs under msprof)
- [ ] Both scripts use ctypes with argtypes from StageSpec.abi.arguments
- [ ] Every tensor passed to `call_kernel` is forced `.contiguous()` immediately before `.data_ptr()` (guards against non-contiguous views from inv/solve/transpose/permute/broadcast/slice)
- [ ] ValidationScript has DEFAULT_CASES with ≥6 BT values (from StageSpec, not hard-coded)
- [ ] ValidationScript uses the fp64 Frobenius rel-error gate (ftol=2e-3) with an fp64-built reference (rule 15)
- [ ] BenchmarkScript uses `torch.npu.Event` device timing (default) with a per-iteration 256 MiB L2 flush; `--timer wallclock` is an optional fallback, not the default
- [ ] BenchmarkScript reports all 6 statistics (mean, min, max, median, p95, stddev) in ns
- [ ] BenchmarkScript benchmarks at the contract production sweep and supports `--l-seg-list`
- [ ] BenchmarkScript sweeps >=2 sizes and reports `slope_per_unit` (per work-unit) as the headline, with the `(size, units, median_ns)` fit points (rule 27)
- [ ] BenchmarkScript supports `--baseline-so` for a within-process paired A/B and reports the paired delta (rule 28)
- [ ] No hard-coded dimension values (HV, H, K, V) — all derived from StageSpec.problem
- [ ] No double-escaped `\n` in Python source — plain parseable text after JSON decode

## Hard Rules

### Port Contract Rules

1. Return only one strict JSON object: `{"outputs": {"ValidationScript": "...", "BenchmarkScript": "..."}}`
2. Python outputs must be plain file contents as strings, not double-escaped one-line blobs.
3. Do not emit prose outside the JSON object.
4. Scripts must be standalone — no sibling imports, no external dependencies beyond torch/torch_npu/ctypes/argparse.

### Launch ABI Rules

5. Use `torch.device("npu")` (not `"npu:0"` or `"npu:1"`).
6. Extract stream pointer as `stream_ptr = getattr(torch.npu.current_stream(), "_as_parameter_", None)`.
7. Pass `stream_ptr` directly in call_kernel args — do NOT wrap in `ctypes.c_void_p()`.
8. In the ValidationScript do NOT call `torch.npu.synchronize()` — it hangs in msprof op simulator; compare on-device without `.cpu()` copies. (The BenchmarkScript is exempt: it never runs under msprof and event timing requires `synchronize()`.)
9. Allocate launch tensors on NPU: `torch.randn(..., device='npu')` or `.to('npu')`.
10. When using `torch.Generator()` for seeding, create tensors on CPU first, then `.to('npu')`. Do NOT pass `device='npu'` to randn when also passing a CPU generator.
11. The ctypes argtypes list MUST include ALL arguments from StageSpec.abi.arguments in exact order.
12. Use the StageSpec.abi-provided ctypes types (c_uint32, c_void_p, c_int64, etc.) for each argument.
12a. Load the kernel `.so` by ABSOLUTE path: `ctypes.CDLL(os.path.abspath(kernel_path))`.
    `ctypes.CDLL` resolves a bare or relative filename through the dynamic-loader
    search path (`LD_LIBRARY_PATH`/system dirs), NOT the current working directory,
    so a bare `kernel_x.so` argument silently fails to load even when it sits in
    cwd. Both ValidationScript and BenchmarkScript must `os.path.abspath()` the
    positional `.so` input before `CDLL` so callers can pass a plain filename.
12b. Synchronize device-side input/workspace allocation BEFORE the kernel launch.
    For a kernel that reads a GM workspace or input it expects pre-initialized
    (especially multi-sub-launch kernels), an async allocation -- `torch.zeros(...,
    device='npu')` or a host->device `.to('npu')` copy -- can race the first
    sub-launch (the copy runs on a different stream than the kernel), giving
    intermittent garbage/failures on real NPU only. Allocate all input/workspace
    tensors, then call `torch.npu.synchronize()` ONCE before `call_kernel` (outside
    any timed region for benchmarks). This is a real-NPU race the simulator does
    not show.
12c. Force every launch tensor `.contiguous()` immediately before taking its
    `.data_ptr()`. The kernel reads its GM operands as raw row-major contiguous
    bytes — it has no knowledge of torch strides. Any tensor produced by an
    operation that can return a non-contiguous view — a matrix factorization such
    as `torch.linalg.inv` / `solve` / `cholesky` (these frequently hand back a
    transposed, column-major layout), or `.t()` / `.transpose()` / `.permute()` /
    `.mT`, broadcasting / `.expand()`, or strided slicing — then has a `data_ptr()`
    whose byte order does NOT match the kernel's row-major read, so the kernel
    silently consumes transposed or strided elements. The failure is insidious: it
    surfaces as a *uniform* numerical error that scales with the contracted / inner
    dimension (a row-major reader handed a column-major matrix reads a column where
    it wanted a row), which is easily misattributed to the kernel. ALWAYS chain
    `.contiguous()` after `.to(device)` for every tensor passed to `call_kernel`
    (e.g. `xn = x.contiguous().to(device).contiguous()`), and also `.contiguous()`
    any tensor a reference helper builds (e.g. the output of a factorization) before
    it is handed to the kernel. Diagnostic rule: if the CPU/reference math is exact
    but the kernel shows a uniform error that grows with a contracted dimension,
    check `.is_contiguous()` on the inputs BEFORE editing the kernel — a
    non-contiguous launch tensor is a harness/layout bug, not a kernel bug.

### Validation Rules

13. DEFAULT_CASES must include at least 6 realistic BT values with ≥2 different seeds each.
14. Derive BT values from StageSpec.problem and StageSpec.production_dimensions, not hard-coded.
15. Use the fp64 Frobenius relative-error metric (the strict accuracy gate), NOT a
    loose elementwise `rtol`. A loose elementwise tolerance (e.g. `assert_close`
    rtol~2e-2, or RMSE/mean+R2) PASSES fp16-intermediate-compute kernels while
    MASKING a real precision deficit; the Frobenius gate exposes it.
    - Build the reference in fp64: upcast the fp16 inputs to `.double()` INSIDE the
      reference and compute all elementwise/matmul/inverse/scan math in fp64.
    - PASS gate: cast actual+expected to fp64 and require the Frobenius relative
      error `sqrt(sum((actual-expected)^2) / sum(expected^2)) <= ftol` with
      `ftol = 2e-3`.
    - Keep a SECONDARY loose elementwise sanity bound (does not relax the gate):
      `atol = 0`, `adjusted_rtol = min(0.5, 5e-3 * chunk_size)`, failing only if
      EVERY element exceeds `atol + adjusted_rtol * |expected|` (mirrors megagdn's
      `NumericalAccuracy.stats_ok` `.all()` check).
    - Report the Frobenius rel-error AND PASS/FAIL under `ftol=2e-3` per case, and a
      `max frob_rel_err` summary line. Reference helper (factor into a shared
      `strict_metric.py` and import it from every `validation_<stage>.py`):
      ```python
      def stats_ok(actual, expected, desc, chunk_size=128, ftol=2e-3, rtol=5e-3, atol=0.0):
          a = actual.detach().double(); e = expected.detach().double()
          diff = (a - e).abs(); denom = torch.sum(e ** 2)
          fre = (torch.sqrt(torch.sum(diff ** 2) / denom).item()
                 if denom.item() != 0.0 else torch.sqrt(torch.sum(diff ** 2)).item())
          adj = min(0.5, rtol * chunk_size)
          all_exceed = bool((diff > atol + adj * e.abs()).all().item())
          ok = (not all_exceed) and (fre <= ftol)
          print(f"  {'PASS' if ok else 'FAIL'}  {desc} (frob_rel_err={fre:.3e}, ftol={ftol:.1e})")
          return ok, fre
      ```
    - Note: this gate is dtype-honest for fp16 ONLY when the kernel does its
      numerically-sensitive elementwise math in fp32 (kernel rule C32). A pure
      fp16-compute kernel will not reach ftol=2e-3 on gate/exp/scan-heavy stages.
16. Use StageSpec dimensions for all test cases, not hard-coded values.
17. Map BT→NT using an algorithmic heuristic (e.g., `nt = max(1, total_tokens_target // bt)` where `total_tokens_target` is derived from `stage.production_dimensions`), not a hard-coded dict.

### Benchmark Rules

18. Default to `torch.npu.Event` device timing: record one start/end pair per iteration,
    `.zero_()` a 256 MiB int8 L2 scratch before each timed call, `synchronize()` once at
    the end, read `elapsed_time` (ms) and convert to ns. `--timer wallclock` (per-iteration
    `perf_counter`) is an explicit fallback, not the default. Never report a single batch
    average duplicated across min/max/median/p95.
19. Report mean, min, max, median, p95, and stddev latency in nanoseconds.
20. Benchmark at the contract PRODUCTION sweep; support `--l-seg-list` for multiple sizes.
21. When storing results per-BT, use string keys: `results[str(BT)] = stats`.
22. Write JSON summary to `--out-json` or stdout. Expose `--timer` and `--flush-mib` so the
    harness can be set to match an external baseline's method exactly.
27. **Report the per-work-unit SLOPE as the headline metric, not a single-size median.**
    Total latency ~= `slope * units + intercept`; at production scale the slope dominates
    and a single small-size median is intercept-polluted (a kernel can "win" a tiny benchmark
    on intercept alone while losing by multiples at scale). The sweep MUST span >=2 sizes
    (include a near-production size, not just toy dims) and the JSON MUST carry
    `slope_per_unit` with the `(size, units, median_ns)` points it was fit from. `units` =
    the stage's repeated work count from the contract (problem_size / tile_size). This is
    algorithm-agnostic: every staged kernel has a repeated work unit (chunk / tile / block /
    token-segment) and a contract sweep axis.
28. **Support a within-process PAIRED A/B comparison (`--baseline-so <path>`).** When given,
    time THIS .so and the baseline ALTERNATELY (A,B,A,B... in ONE process, sharing the same
    L2 flush and sizes) and report the paired per-unit delta. Common-mode device drift cancels
    only when paired; a "speedup" that appears only across separate processes/sessions is
    drift, not a real win. Never claim a comparison from two unpaired runs.

### Dimension Rules

23. All dimension constants MUST come from StageSpec.problem dict, never invented or hard-coded.
24. If StageSpec.problem contains HV, H, K, V, use those values. If not, derive from stage.inputs/outputs shapes.
25. Never set dimensions to toy values (HV≤32, H≤8, K≤64, V≤64) unless StageSpec explicitly specifies them.
26. Distinguish helper compile-time constants from logical stage dimensions (see `references/npu_launch_patterns.md`).

## Reference Files

- `references/npu_launch_patterns.md` — NPU launch patterns, ctypes boilerplate, stream handling, dimension derivation
- `references/validation_patterns.md` — ValidationScript template, DEFAULT_CASES generation, two-tier accuracy, CLI template
- `references/benchmark_patterns.md` — BenchmarkScript template, torch.npu.Event timing + L2 flush, statistics, JSON output, production sweep support, per-work-unit slope fit, within-process paired A/B (`--baseline-so`)

## Failure Policy

- If StageSpec is incomplete or missing critical fields, emit a fail-fast script with a precise error message instead of guessing.
- If StageSpec.abi.arguments is empty or malformed, emit a fail-fast script that raises an error on import.
- If ReferenceModel is unreadable, emit a ValidationScript that imports it at runtime and lets the import error surface.
- Do not invent stronger invariants than StageSpec guarantees.
- Do not claim support for shapes, dtypes, or layouts not justified by the inputs.
