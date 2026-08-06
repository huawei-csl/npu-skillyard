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
- [ ] If a VENDOR framework operator is timed, rule 29 is satisfied in full, INCLUDING arity match with byte counts for both sides (h), allocation symmetry on our side (i), both A/B arms measured in the SAME PROCESS (j), and a rep count justified by convergence rather than habit (k): flush enqueued and never drained; the timed region is symmetric on both sides; outputs allocated per call with `torch.empty` (not `torch.zeros`, not preallocated); issue order randomized per repetition; a null control reported whose CI includes 1.0; >=200 reps with a bootstrap CI; arity/semantics match stated with its bias direction. Otherwise the ratio is labelled ADVISORY
- [ ] The validation sweep reports items-per-lane per case and includes at least one case with items_per_lane >= 3 at the production block_dim, so the kernel's pipelined/steady-state path is actually exercised; a sweep where every lane owns <= 1 item is a coverage gap (rule 31)
- [ ] Any launch that can raise an aicore exception is followed by a device health check before a FAIL is recorded, with retry on a second device; a fault that cannot be reproduced on a healthy device is reported as `device-poisoned`, not as a stage failure (rule 30)
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

29. **Comparing against a VENDOR framework operator (e.g. `torch_npu.npu_*`) needs a
    stricter protocol than `.so` vs `.so`, and getting it wrong has repeatedly produced
    ratios that were wrong by 2-3x IN OUR FAVOUR.** Our kernel is launched by a bare
    ctypes call (~6 us of host dispatch); a framework operator goes through
    python -> aten -> allocator -> launch (~31-52 us). Any protocol that lets host
    dispatch fall inside one side's event window and not the other's measures dispatch,
    not the kernel. Required, all of them:

    a. **Flush ENQUEUED, never drained.** `.zero_()` the 256 MiB L2 scratch and do NOT
       call `torch.npu.synchronize()` between the flush and the start event, on either
       side. The flush is a ~200 us device-side spacer that hides host dispatch equally.
       A drain leaves the device idle at the start event, so dispatch lands inside the
       window -- measured effect: vendor 76 us -> 162 us, our side barely moving.
    b. **Symmetry is the invariant.** Whatever is inside the timed region for one side
       must be inside it for the other. The worst observed bug was asymmetric: our side
       timed undrained while the vendor was effectively drained, reporting 0.83x for a
       kernel that is really 1.80x. That is worse than a symmetric drain, which at least
       inflates both sides.
    c. **Allocate outputs the way the vendor does.** A framework op allocates its outputs
       per call and does NOT zero them. If your harness allocates with `torch.zeros`
       inside the timed callable it pays a fill the vendor never pays -- measured 1.330
       with `torch.zeros` vs 1.179 with `torch.empty` on the same kernel. Use
       `torch.empty` per call. Do NOT preallocate once outside the window either: that
       skips an allocation the vendor pays for and biases the other way.
    d. **Randomize the issue order per repetition, and make it BALANCED.** A fixed
       "ours then vendor" order carries a ~0.9% slot bias -- large enough to have
       flipped an optimizer decision. But a free coin flip is not enough either: an
       unbalanced draw leaves a residual positional bias, and one run's null control
       came out 1.0016 CI [1.0007, 1.0027] -- **excluding 1.0 on slot bias alone**,
       which voids every ratio from that harness by (e). Use a balanced permutation:
       exactly half the repetitions "ours first", half "vendor first", shuffled --
       e.g. `order = rng.permutation([True]*(n//2) + [False]*(n - n//2))`. Costs
       nothing and removes the failure mode rather than relying on it averaging out.
    e. **Run a NULL CONTROL.** Time one callable against ITSELF through the identical
       harness. If its confidence interval excludes 1.0, the harness is broken and every
       ratio from it is void. Report the null control alongside the ratio, always.
    f. **>=200 repetitions, median ratios, and a bootstrap CI.** 20 repetitions is not
       enough: at small sizes it produced 68% relative standard deviation and a ratio
       that was wrong by 35%.
    g. **State the arity/semantics match.** If the vendor writes extra outputs, or is
       in-place while you are out-of-place, say so and say which way it biases. An
       in-place vendor op compared against an out-of-place kernel moved one case from
       1.34x to 0.96x once matched.

    h. **Match the ARITY, and say so.** If the vendor writes outputs your kernel does not,
       it is doing strictly more work and the ratio is biased in your favour. Measured
       instance: `npu_fusion_attention` writes 20.000 MB at S=2048 (attention output plus
       `softmax_max` and `softmax_sum`, fp32 `[B,N,S,8]`) where a 1-output kernel writes
       16.000 MB -- 25% more output traffic on a memory-bound stage. Emitting the two
       missing outputs moved the honest ratio from 1.110 to 1.149, and it reversed the
       headline: "beats the vendor below S=2048" became "parity at S=256, ~5% behind at
       S=512-1024". Report the byte counts of both sides, not just the claim.
    i. **Allocate on YOUR side whatever the vendor allocates.** Reusing a preallocated
       output while the vendor allocates per call skips work the vendor pays for. This is
       the same error as (c) but on the harness side rather than the fill side, and it
       compounds with (h): the two together accounted for the whole of a false
       "we beat the vendor" result.
    j. **Compare only WITHIN ONE PROCESS -- the boundary is the process, not the session.**
       Measured directly (`skillyard-runs/drift_study.py`). Inside a single process there
       is NO drift: six blocks of 300 paired reps agreed to **0.15%** against a 0.29% CI,
       first-half vs second-half differed by <=0.11%, and the running estimate settled
       inside +/-1% by **n=10** and +/-0.5% by **n=20**. Between processes, the SAME binary
       at the SAME shape minutes apart gave 1.151 / 1.169 / 1.165 -- a **1.5% offset with
       non-overlapping CIs**, and across sessions 1.094 / 1.110 / 1.120 / 1.131. That is a
       per-process calibration offset, not sampling noise, and **no rep count reduces it**.
       So: both arms of every A/B in the same process, paired and interleaved; never
       compare a number from one process against another.

    k. **Pick the rep count from convergence, not habit.** Since the estimate is converged
       by n~20, **~50 paired reps is enough and 200 is generous**; beyond that you buy
       nothing but wall-clock. This matters at large shapes -- at 361 ms per call, 200 reps
       costs two minutes to shrink an already-converged number. Scale reps down as the
       per-call cost rises (e.g. 200 up to a few ms, 60 for tens of ms, 30 beyond) and say
       what you used. Report the CI either way: a converged estimate with a stated interval
       beats an arbitrary rep count every time.

    If any of a-k cannot be satisfied, label the number ADVISORY in the JSON and in the
    report. Never present an unverified vendor ratio as a result.

30. **An aicore exception POISONS the NPU device. Treat "device faulted" as STOP-AND-MOVE,
    never as "this stage FAILs".**
    Measured on 910B2 / CANN 9.1.0: after one kernel raised an aicore exception
    (`507015`), the *same binary that had just run 30/30 clean on that device* then failed
    on iteration 0, fifty times in a row, while two other devices ran it 50/50 clean at
    the same wall-clock moment. The device recovered on its own roughly half an hour
    later. Detail: `skillyard-runs/isa_probes/README.md`, section 4.

    The failure mode this creates is not a lost run, it is a **fabricated one**: the first
    genuine fault converts every later launch on that device into a fault, and a harness
    that records verdicts blindly reports a whole sweep of stages as broken when only one
    was. This has already invalidated two of our own probe sweeps.

    So a harness that launches kernels which can fault MUST:
    a. **Health-check before trusting a FAIL.** After any launch that returns
       `507015`/`aicore exception`, re-run a known-good launch on that device. If the
       known-good launch also fails, the device is poisoned: the FAIL is unattributable.
       Report it as `device-poisoned`, not as a stage failure.
    b. **Retry on a different device** (`ASCEND_RT_VISIBLE_DEVICES`) before recording any
       verdict, and record which device produced it.
    c. **Never run one process per configuration on one device and read the result column
       as a failure rate.** That column measures the poisoning, not the kernel. If you
       need a rate, rotate devices and health-check between repetitions.
    d. **Say so in the report.** "3 configurations faulted, all after the first fault on
       device 0" is a completely different statement from "3 configurations faulted".

31. **The validation sweep MUST include a size where each lane owns SEVERAL work items.
    A kernel can pass every generated test without its own pipelined path ever running.**
    This is not hypothetical: a double-buffered kernel here validated **100% exact at
    T=8 and T=64 and was corrupt from T=512 up**. At `block_dim=20` (40 lanes) and 2 rows
    per item, T=8 is 4 items and T=64 is 32 items -- fewer items than lanes, so every lane
    owned at most ONE item and the prefetch branch was dead code. Half the sweep tested a
    kernel that, at production size, silently produced wrong data.

    a. **Compute items-per-lane, do not eyeball it.** `items = ceil(rows / rows_per_item)`
       and `items_per_lane = items / lanes`, where **`lanes` depends on which engine the
       stage runs on**:

       | stage archetype | lanes | why |
       |---|---|---|
       | Vec (`vec_only`, and the Vec half of a mixed stage) | `block_dim * 2` | both AIV sub-blocks are workers -- verified |
       | Cube (`cube_only`) | `block_dim` | one Cube worker per block; there are no sub-blocks |

       An earlier version of this rule stated `lanes = block_dim * 2` universally. That
       **double-counts a `cube_only` stage** and makes its `items_per_lane` read half its
       true value, so a sweep can look like a coverage gap when it is fine (or, worse,
       pass a `>= 3` check it did not actually meet). Put the figure and the engine in the
       validation output next to each case so a reader can see which cases exercised the loop.
    b. **Require at least one case with `items_per_lane >= 3`** at the production
       `block_dim`. Three, not two: a 2-slot ring must WRAP and re-enter steady state, not
       just run prologue-then-epilogue. If the contract's largest size cannot reach 3,
       say so explicitly in the report rather than passing quietly.
    c. **Also test at least one case with `items_per_lane < 1`** (more lanes than items),
       since the prologue/drain path with an empty or single-item lane is where
       off-by-one token accounting shows up.
    d. **Report the number of distinct lanes actually used.** A sweep that only ever runs
       one item per lane is a coverage gap of the same kind as testing one dtype.

    The failure mode this prevents is the worst kind: not a crash, not a hang, but a
    kernel that passes its gate and returns wrong numbers only at the size anyone
    would actually run.

### Dimension Rules

23. All dimension constants MUST come from StageSpec.problem dict, never invented or hard-coded.
24. If StageSpec.problem contains HV, H, K, V, use those values. If not, derive from stage.inputs/outputs shapes.
25. Never set dimensions to toy values (HV≤32, H≤8, K≤64, V≤64) unless StageSpec explicitly specifies them.
26. Distinguish helper compile-time constants from logical stage dimensions (see `references/npu_launch_patterns.md`).

## Reference Files

- `references/npu_launch_patterns.md` — NPU launch patterns, ctypes boilerplate, stream handling, dimension derivation
- `references/validation_patterns.md` — ValidationScript template, DEFAULT_CASES generation, two-tier accuracy, CLI template
- `references/benchmark_patterns.md` — BenchmarkScript template, torch.npu.Event timing + L2 flush, statistics, JSON output, production sweep support, per-work-unit slope fit, within-process paired A/B (`--baseline-so`), vendor-baseline comparison protocol with null control (rule 29)

## Failure Policy

- If StageSpec is incomplete or missing critical fields, emit a fail-fast script with a precise error message instead of guessing.
- If StageSpec.abi.arguments is empty or malformed, emit a fail-fast script that raises an error on import.
- If ReferenceModel is unreadable, emit a ValidationScript that imports it at runtime and lets the import error surface.
- Do not invent stronger invariants than StageSpec guarantees.
- Do not claim support for shapes, dtypes, or layouts not justified by the inputs.

## Benchmark reporting: one ratio definition, and cross-check it

Two harness defects from `flash_attention_grad` that produced a **false headline win**
(a claimed 1.331x at S=256 that was really 1.018x):

1. **`ratio-of-medians` != `median-of-paired-ratios`.** Both are defensible; mixing them in
   one table is not. Pick one, name it in the protocol block, and **cross-check it against
   the ratio of the two medians printed in the same row**. If they disagree by more than a
   few percent, that row is a bug — with a skewed kernel (min 38.9 us, max 171.8 us) the two
   diverged by 30% and the wide CI `[1.151, 1.558]` next to neighbours at `[±0.005]` was the
   only visible tell.
2. **A JSON whose `protocol` block differs MUST have different data.** Two files recording
   `reps=200` and `reps=400` carried bit-identical medians to 14 significant digits — one was
   a stale copy written without re-running. Before trusting a re-measurement, confirm its
   numbers actually moved.

**Minimum reps scale with how short the kernel is relative to the flush.** When the timed
region is much shorter than the 256 MiB L2 flush, 60 reps is not enough: the same case read
`1.540x [1.271, 1.757]` at 60 reps and `0.903x [0.897, 0.908]` at 200 — the point estimate
crossed parity. Use >=200 reps for kernels under ~100 us, and treat a CI wider than a few
percent as "not yet measured" rather than as a result.

### The ratio direction MUST be unmissable

Two runs in the same suite reported `1.4975` (meaning **we are slower**) and `18.590x`
(meaning **we are faster**). Both were internally documented; neither was readable at a
glance, and a reader comparing them across cases is silently misled about which way the
suite is going.

**Every reported ratio MUST be accompanied by, in the same row or object:**

1. **both raw medians**, `ours_us` and `vendor_us` -- these are unambiguous under any
   convention and are what a reader should be able to fall back on;
2. an explicit **`ratio_definition`** string naming the numerator and denominator;
3. a **plain-language verdict**: `"we are 2.12x FASTER"` / `"we are 1.50x SLOWER"`.

Prefer the project convention -- **`ours / vendor`, lower is better, `< 1` beats the
vendor** -- but the convention alone is not the safeguard. The safeguard is that the raw
medians and the worded verdict are always present, so the direction survives being copied
into a summary, a table, or a paper.

**Never report a bare ratio in a summary.** A number like `1.5` with no latencies beside it
is not a result; it is a coin flip about which way the comparison ran.

### A within-arm null control CANNOT detect cross-arm interference

Rule 29's null control runs the same callable against itself and checks the ratio is ~1.0.
That catches drift and ordering bias. **It cannot catch one arm perturbing the other**,
because interleaving A-with-A perturbs both sides equally and the effect cancels.

Measured on `attention_sdpa` at S=128, same process, 200 reps:

| | vendor interleaved with ours | vendor run ALONE | within-arm null |
|---|---|---|---|
| median | **68.02 us** | **34.86 us** | 0.998 -- **PASSES** |

Interleaving nearly **doubled** the comparand's time while the null control read clean. Taken
at face value this is "2.04x FASTER"; the honest answer is **parity**. The mechanism is host
dispatch: at small sizes the vendor arm is dispatch-bound, and alternating it with a
different callable inflates that cost. It fades as the kernel grows (1.7% by S=512).

**Therefore, for every reported row:**
1. Measure each arm **ALONE** as well as interleaved.
2. Form the **ratio from PAIRED interleaved runs only** (median of repeated paired runs).
3. Use the ALONE numbers as a **diagnostic**: if an arm's alone and interleaved medians differ
   by more than a few percent, that row is contaminated -- say so, and treat any effect smaller
   than the discrepancy as unmeasured.

> **Do NOT build the ratio from `min(alone, interleaved)` per arm.** An earlier version of this
> rule said "report the comparand's best time", and taking a per-arm minimum mixes two
> different measurement conditions into one quotient. It produced an outright **sign error** on
> a 3% effect -- an attempt read as 1.03x FASTER when it was in fact 1.04x SLOWER. The two arms
> must come from the same interleaved stream for the pairing to cancel drift; the alone runs
> tell you whether to trust that stream, not what to divide.

**Small sizes are where this bites**, and it compounds with the other small-size hazard: a
short kernel is dispatch-bound on BOTH arms, so per-call rows there measure ctypes overhead
rather than the kernels. Prefer batched timing windows at small sizes, and state plainly
which rows are dispatch-bound and therefore not quotable as an algorithmic result.

### Validate each case MORE THAN ONCE -- single-run validation cannot see intermittent corruption

A stage-2 event-ID collision produced **silent** `y_q` corruption in roughly **1-2 runs in 30**.
It passed a 26-case validation sweep, because **each case ran once**. It was found only when a
case was repeated, and the surgical fix took it to **0/200**.

A validation suite whose cases each run once measures the wrong thing: it samples 26 points of
a distribution whose failure rate is ~5%, so it misses the bug with probability ~0.95^26 ~ 26%
per attempt -- and it will keep missing it, run after run, while the kernel ships.

**Requirement:** every validation case runs **>= 3 times** with fresh GM allocations, and the
production point runs **>= 20**. Report the pass count as a fraction (`30/30`), never as a bare
PASS. A determinism check over repeated runs is not a substitute -- determinism compares our
output to itself, so a consistently-wrong kernel passes it.

This is the same failure mode as the benchmark null control: a check that only compares a thing
to itself cannot see a fault that affects both sides equally.

### A null control needs a WIDTH check, not just "the CI contains 1.0"

"Null control valid iff its 95% CI contains 1.0" is necessary but **not sufficient**. A
contended or noisy row produces a CI wide enough to contain 1.0 *while the two identical arms
differ by 30%* — the check passes precisely because the measurement is bad.

**Require both:**
1. the CI contains 1.0, **and**
2. the point estimate is within a few percent of 1.0 **and the CI is narrow** (a practical
   bar: width <= ~3% of the estimate).

A wide CI is not evidence of validity; it is evidence that the row is not yet measured. Widen
reps until the interval tightens, or mark the row not reportable.

**Related harness bug, worth checking for in any bootstrap you write:** pairing the two arms'
**sorted** samples elementwise before resampling destroys the very variation the interval is
supposed to capture. It spuriously rejected 3 of 5 rows in one run. Pair by *rep index* (the
interleaved order they were measured in), never by rank.

### `block_dim` is a VALIDATION axis, not just a tuning knob

A `masked_softmax` kernel passed **49/49 at `block_dim=24`** and was **broken at
`block_dim=1`**. The bug was a slot-recycling race that only manifests when a lane owns more
than one work item -- at the default `block_dim` every lane owned exactly one, so the race
could not occur and the gate reported a clean sweep.

**Validate at a LOW `block_dim` as well as the production one**, chosen so lanes own several
items (`items_per_lane >= 3`, rule 31). `block_dim=1` is the sharpest setting and costs one
extra run per case.

This is the fourth distinct instance of one failure: **a validation sweep that does not vary a
dimension cannot see a bug that lives along it.** The others were single-run validation
(intermittent corruption), determinism checking (inter-item races), and the within-arm null
control (cross-arm interference). When a gate passes everywhere, ask which axis it never moved.

### Randomized order balances SLOTS, not ADJACENCY -- counterbalance it

A randomized interleave gives each arm the same *distribution* of positions, but not the same
distribution of *predecessors*. Back-to-back identical arms were measured to differ by **~5%**
on this hardware, so a schedule that happens to put more of arm A after arm B biases the
comparison even though every arm got its fair share of slots.

**Use a counterbalanced order** (e.g. ABBA / BAAB blocks) so each arm follows each arm equally
often, and report the residual A-after-A vs A-after-B difference as a diagnostic.

Two earlier symptoms of the same effect, now explained: two labels of an *identical* vendor arm
split 3.9% at one size and 29% at another under a global multiset shuffle; and a vendor arm
read 68.02 us interleaved against 34.86 us alone while its within-arm null control passed at
0.998.

**Before trusting a re-measurement that changes a conclusion, cross-check it against the rows
you already have.** When this was fixed mid-run, the earlier 200-rep rows were re-checked and
agreed to -0.11% / +0.84% / +1.11% / +0.10% -- so the defect was real but had not moved those
numbers, which is worth establishing rather than assuming in either direction.


### A per-stage reference derived from the REFORMULATION cannot check the reformulation

`top_k_top_p` reformulated a sort-and-select into a lexicographic threshold on
`(value, index)` -- a large algorithmic win. **All four stages passed their own specs while
the end-to-end gate failed 12/20 runs.**

The cause was an exact fp32 tie straddling the top-p cut (`d[189] == d[190]`): in fp64 both
methods agree the cut is at rank 190, but a pure value threshold also admits the tied twin.
The per-stage references had been transcribed from **the same reformulation**, so they shared
its blind spot precisely where it was wrong.

**Rule: the end-to-end reference must be an INDEPENDENT statement of the original algorithm** --
transcribed from the source definition, not derived from the decomposition you invented. Per-
stage references check that a stage implements its spec; only an independent end-to-end
reference checks that the specs add up to the algorithm.

This is the fifth instance of one failure mode in this project: **a check that compares a
thing to itself cannot see a fault that affects both sides equally.**

| check | blind to |
|---|---|
| within-arm null control | cross-arm interference |
| single-run validation | intermittent corruption |
| determinism check | inter-item races (reproduces identically) |
| a sweep that fixes `block_dim` | bugs that need >1 item per lane |
| per-stage reference from the reformulation | errors in the reformulation |

When a gate passes everywhere, ask what it shares with the thing it is checking.

### A device health check must verify DATA CORRECTNESS, not throughput

Devices on this host silently corrupt data **while still hitting full TFLOP/s** and while
`npu-smi` reports `Health Status: OK`. A throughput probe therefore proves nothing about
whether the numbers you just took are real.

One campaign run reported "device healthy, 226 -> 266 TFLOP/s, band >= 220" before and after
its timed runs. A correctness canary on the same device shortly afterwards found
**2176 wrong elements**. The throughput check passed because throughput was never the
failing property.

**The check must compare a vendor-only computation element-by-element against a CPU reference
and report the count of wrong elements. Zero is the only pass.** A matmul plus an elementwise
chain is sufficient and takes seconds. Run it before AND after every timed run, and record
both counts in the report.

Corollary for the campaign: any case whose device is later found degraded must be
**re-verified on a clean device** before its number is accepted, regardless of what its own
health check said. `masked_softmax` was re-verified this way and reproduced to 0.07%; the
result stood, but that was established rather than assumed.


### Launch on torch's stream, not `NULL`

Passing `stream=NULL` to a ctypes `call_kernel` puts the kernel on a **different runtime
stream** from torch's. `.cpu()` then synchronises only torch's stream and reads the output
buffer **mid-flight**, producing size-dependent, nondeterministic NaNs that look exactly like
a kernel bug and will send a repair loop chasing the kernel.

**Always pass `torch.npu.current_stream().npu_stream`.** Generated harnesses must do this by
construction, and a harness that produces nondeterministic output should have its stream
argument checked before the kernel is suspected.


### A valid null control and a tight CI do NOT make a small effect real -- REPLICATE

This is the single most important measurement finding in the campaign, and it invalidates a
class of results that look rigorous.

Five runs of the **same** A/B comparison, each with a **valid null control** and each with a
tight bootstrap CI, produced:

```
1.0212   1.0136   0.9553   0.9856   0.9588
```

**Mutually exclusive confidence intervals. Opposite signs.** A 6.9% spread on a comparison
that has no true effect that large.

**The bootstrap CI measures within-attempt precision, not across-attempt reproducibility.**
It resamples the reps you happened to take in one session; it cannot see run-to-run state --
allocator layout, cache residue, clock and thermal drift, whatever else moves between
processes. A null control tests for *bias inside* an attempt; it says nothing about whether
the attempt would repeat.

**Rule: any effect under ~3% must be REPLICATED in independent runs before it is kept or
reported.** Report the spread across replicates, not one CI. An effect that changes sign
across replicates is noise regardless of how tight each interval was.

The run that found this retroactively demoted **8 of its own 16 attempts** to "inside the
noise band", leaving only the three that were 1.28x or larger. Do the same: a campaign that
keeps a stack of 1.02x attempts has probably kept a stack of noise, and their product is
reported as a speedup that will not reproduce.

**Corollary for the stop gate.** "We are within 1% of the floor" is not a measurement at this
resolution either -- it is inside the same band. Claim a hardware limit from a margin larger
than your replication spread, or say the margin is not resolvable.


### Contention: a null control cannot see it, and the arms are NOT inflated equally

A concurrent timed benchmark in another case inflated a batch mid-campaign. Measured on the
same kernel, same shape, ~1 hour apart, with the null control valid at 0.17-0.62% throughout:

```
clean    ours 258 us   vendor 426 us
CONTENDED ours 312 us   vendor 711 us    <- ours x1.21, vendor x1.67
clean    ours 258 us   vendor 427 us
```

Two things matter here.

**(1) The null control is blind to it.** It is an ours-vs-ours ratio measured inside the same
contended window, so a slowdown affecting both sides cancels. This is the same structural blind
spot as within-arm controls generally: *a check that compares a thing to itself cannot see a
fault affecting both sides equally.*

**(2) The arms are NOT inflated equally**, so ratios do not survive either. The batch above
claimed `bd=16` was **1.09x faster**; measured clean it is **1.11x slower**. Whichever arm has
more host-side work per device microsecond stretches more.

**Required: an ABSOLUTE ANCHOR, re-checked per batch.** Time one fixed reference configuration
(the shipped kernel at the production shape) at the start and end of every benchmarking session.
If the anchor moves more than a few percent between checks, the whole batch is contaminated --
discard and re-measure. A null control tests for *bias within* a comparison; the anchor tests
whether *the machine itself* was the same. You need both.

**Diagnose contention by cause, not by coincidence.** In the run that found this, the blame was
put on a 21-hour orphaned `--sim-mode` process at 574% CPU. That process was real, and it was
leaked waste -- but on a 192-core host it is **3% of the machine**, and it was running
continuously across the clean measurements *and* the contended one. A constant cannot explain a
transient. The actual cause was a second case benchmarking concurrently. Before blaming a
process you happen to notice, check that it was **absent** when the measurement was clean.

**Never leave a sim process unbounded.** Run every `--sim-mode` / msprof invocation under
`timeout 1800`, and reap it -- the orphan above had been reparented to init after its pipeline
exited, so nothing was left to kill it.

### Scale the null-control gate to the effect size

The gate "null-control CI must contain 1.0" is calibrated for small effects and is wrong for
large ones. On a **quiet** host, a re-check measured a null control of `0.9984`,
CI `[0.9978, 0.9996]`, width `0.18%` -- technically INVALID because the CI excludes 1.0, on a
row claiming **3.63x**. A 0.16% systematic bias cannot threaten a 263% effect.

Judge the null control **relative to the claimed effect**:

* Null-control bias `>= 1/3` of the claimed effect -> **row void**, re-measure.
* Bias `>= 1/10` of the claimed effect -> report the row **with the bias stated**.
* Bias `< 1/10` -> the row stands; still record the null control.

This pairs with the replication rule above: a **sub-3% effect needs more** than a valid null
control (independent replication), and a **multi-x effect needs less** (a sub-1% bias is noise
against it). One fixed threshold serves neither.


### Serialize timed runs across CONCURRENT pipelines with a file lock

Generation can run several pipelines at once; **timed measurement cannot**. If two agents
benchmark simultaneously they corrupt each other, unequally, and no per-run check catches it.

Take a cross-process lock around every command whose **number you will report**:

```bash
flock <campaign_root>/.bench.lock -c '<the full timed command>'
```

Compiles, correctness validation and msprof sim do **not** need the lock -- they are not timed.
Hold it for the timed command only, never across a whole phase, or the other pipelines starve.

This is a hard interlock, not a heuristic: it makes the contention *impossible* rather than
merely *detectable*. Keep the absolute anchor as well -- the lock only covers pipelines that
cooperate, and it cannot see an unrelated process on the host.


### Different errors are separated by different SHAPES -- one shape cannot cover the suite

`gemma_rms_norm` showed the source's own *distribution* can be unable to discriminate a
semantic error. `group_norm_swish` showed the same for *shapes*, in both directions at once:

| defect | separated by | NOT separated by |
|---|---|---|
| contiguous vs strided channel->group mapping | an added `G != C` case (**5.80e-03**) | **every source shape** -- all three use `G == C` |
| biased vs unbiased variance | the *small* Tier-1 case `[24,35,76]` (**6.5e-03**) | the **production** shape (9.86e-06, *under* tolerance) |

Read those two rows together. **The source's own shapes could not catch the first defect, and
the production shape could not catch the second.** A suite built only from the source would
ship a wrong group mapping; a suite built only at the production size -- the thing the
coverage gate insists on -- would ship a wrong variance.

**So the coverage gate is a floor, not a design.** "Validated at the production shape" is
necessary and nowhere near sufficient. Build the case list by asking, per plausible defect,
*which shape makes this defect visible*:

* **Degenerate small shapes** expose accumulation-order and biased/unbiased errors that
  average away at scale.
* **Shapes that break an incidental coincidence in the source** (here `G == C`) expose
  indexing and mapping errors the source structurally cannot reach. Look for any dimension
  the source happens to hold equal, unit, or power-of-two, and add a case that breaks it.
* **The production shape** catches tiling, tail and capacity errors the small cases miss.

State in the report **which case separates which defect**. A pass table that does not say
what each case is *for* cannot be audited, and cannot tell you what it failed to test.


### Queue starvation: the event window can time HOST latency, not the kernel

`torch.npu.Event(enable_timing=True).record()` costs **28-46 us of HOST time per call**
(measured directly; event *creation* is cheap at ~2 us, `record()` is not). Enqueueing a
timed call therefore costs tens of microseconds of host work on top of the launch itself.

**If device work per call is smaller than host enqueue per call, the stream DRAINS**, and the
event window measures how fast the host could feed the queue rather than how fast the kernel
ran. The mandated 256 MiB flush is only ~156-230 us of device work, which does not always
cover the gap.

The failure is not subtle. One run measured the **same kernel** at **12.25 us** paired against
itself and **26.53 us** paired against the vendor, same shape, same process -- and its
headline inverted from "1.003x SLOWER" to "1.011x FASTER" on the *unchanged* binary once
fixed.

**A null control cannot detect this.** Both arms of a null control are the same kernel, so
they starve identically and the ratio stays 1.00. This is the third distinct instance of the
same structural blind spot: *a check that compares a thing to itself cannot see a fault
affecting both sides equally.*

**Block-level ABBA does not fix it either** -- switching from per-call to 25-call-block
pairing left the bias intact, which is what ruled out kernel-type switching and pointed at the
queue.

#### The detector: paired ratio vs ratio-of-medians must AGREE

You already compute both. **Their divergence is the starvation signature.** Measured directly
by varying only the ballast, everything else identical:

| flush ballast | paired ratio | ratio-of-medians | divergence | verdict |
|---|---|---|---|---|
| 1x (documented protocol) | 0.8946 | 0.9309 | **3.9%** | 1.118x slower |
| 4x | 0.9441 | 0.9420 | **0.2%** | 1.059x slower |

The reported ratio moved **5.5%** and the divergence collapsed. Use this:

* divergence **< 1%** -> the queue stayed fed; the row is sound on this axis.
* divergence **1-1.5%** -> borderline; add ballast and confirm the ratio is stable.
* divergence **> 1.5%** -> **treat as starved.** Add flush ballast until the two converge,
  then re-measure. Do not report the row until they agree.

#### Required harness properties

1. **Pool the events.** Allocate them once, not a fresh pair per timed call.
2. **Enqueue ballast** -- extra flushes -- so the device queue can never empty. Size it so
   `device_work_per_call > host_enqueue_per_call`, and **assert it** rather than hoping.
3. **Enqueue the whole loop, sync once at the end.** Per-call `synchronize()` guarantees the
   queue drains on every iteration and maximises the defect.
4. **Report the divergence** on every row alongside the ratio, so a reader can audit it.

**Small kernels are where this bites.** The distortion scales with how small the device work
is relative to host enqueue, so sub-50 us rows are the exposed ones -- the same rows that are
already near the dispatch floor. When a fast kernel is compared against a slower vendor op,
the *faster* arm starves proportionally more, which **understates your own speedup**. Both
runs that hit this found the fix moved the result in their favour.


### The null control must run in the TREATMENT ARM'S SLOT

A null control that runs the unchanged kernel in a **third** buffer is blind to a
**buffer-slot bias**, and slot bias is real: `rope` measured **slot 2 as 0.8-1.5% faster
regardless of which kernel occupied it**. The giveaway was a swapped-arm test in which the
ratio **did not invert** -- a genuine kernel effect reverses when you swap the arms; a slot
effect does not.

That bias retracted a claimed **"+6% from kTT=108/109"** and demoted two further attempts to
noise.

**Required:** run the null control with the unchanged kernel **in the slot the treatment arm
occupies**, not in a spare one. And when an effect is small, **swap the arms and confirm the
ratio inverts.** If it does not invert, you are measuring the harness, not the kernel.

This is the fourth distinct instance of the same structural blind spot -- after the within-arm
control, host contention, and queue starvation. State it as a standing habit rather than a
list of special cases: **any control that holds the suspect factor constant across both arms
cannot see that factor.** Before trusting a control, name what it varies, and check that the
thing you are worried about is actually one of those things.


### CORRECTION: per-call `torch.empty` on INPUTS is what creates buffer-slot bias

An earlier rule here said "per-call `torch.empty` on **both arms**", to stop one arm inheriting
a warm allocation. Allocating fresh **inputs** per call is the part that backfires: the two arms
land in different allocator slots, and slot position is worth 0.8-1.5% on its own.

Measured on the same comparison, changing only the buffer discipline:

| discipline | forward | swapped | null control | replicate spread |
|---|---|---|---|---|
| per-call `empty` on inputs (two slots) | 1.0583 | 1.0277 | **1.54%** | -- |
| **shared input buffers** | **1.0340** | **1.0351** | **0.03%** | **0.20%** (3 reps) |

With two slots the forward and swapped readings disagree by 3%, and the 1.54% null bias is
>= 1/3 of the 4.3% effect, so the scaled gate **voids the row entirely**. With shared inputs
the forward and swapped readings agree to 0.1% and the null collapses to 0.03%.

**Rule: share the INPUT buffers between arms; allocate per-call only the OUTPUTS.** Inputs are
read-only, so sharing them cannot let one arm contaminate the other's data -- while the outputs
still get a fresh allocation so neither arm benefits from a warm destination.

This is the **root cause** of the buffer-slot bias detected earlier via the arm-swap test. Keep
the arm-swap test as the check; this is the fix.

### One process per shape

Sweeping five shapes in a single process produced an **aicore timeout `507014`** -- roughly
1300 unsynchronized tasks with per-call 100 MB allocations outran the runtime. It is not a
kernel bug and it wastes a whole sweep. **Launch one process per shape**, and let the harness
loop over processes rather than over shapes inside one process.


### A SLOW HOST HIDES A DEVICE RACE -- run determinism checks back-to-back

A UB-reuse race stayed hidden through an entire validation suite because the harness computed
a **4096x8192 fp64 CPU reference between cases**. That host work drained the device queue
between every launch, so the racing operations never overlapped. The kernel looked
deterministic. Run back-to-back with no host work in between, the same binary returned
**61 distinct results from bit-identical inputs in 200 runs**.

**The validation harness's own cost can mask the defect it is meant to find.** This is the
same structural blind spot as the rest: the check and the fault share a dependency, so the
check cannot see the fault.

**Required:**

* Run the determinism check as a **tight back-to-back loop** -- launch N times with **no CPU
  reference computation, no `.cpu()`, no printing, nothing** between iterations, then compare
  afterwards.
* Do **not** interleave the fp64 reference with the repeat launches. Compute the reference
  once, run the repeats, then compare.
* Treat a determinism failure as **higher severity than an accuracy failure**. An accuracy
  failure is a wrong kernel; a determinism failure is a *race*, which means every previous
  passing result was luck.

### `x.clone()` is a REFERENCE, not a CEILING

The floor ladder had `torch.clone()` on identical bytes as an acceptable ceiling estimate. A
kernel measured **32% faster than it** (1173 GB/s vs 886), which makes it useless as an upper
bound -- and worse, a kernel that stops at "we hit the clone rate" stops early and reports a
fabricated headroom of zero.

**Put the ENGINE-NULLED ABLATION first in the ladder**: the same kernel, same tiling, same
access pattern, with the arithmetic removed. That is the only floor that measures *your* data
path. `clone()`, `copy_()` and vendor ops are cross-checks to report alongside it, not gates
to stop on.


### The starvation guard must count FLUSH BALLAST as device work

A harness comparing `host_enqueue_per_call` against **the timed arm alone** will cry starvation
on a perfectly healthy measurement. The flush ballast **is** device work sitting in the same
queue, and it is usually the larger term.

Compute the guard as:

```
device_work_per_call = timed_kernel_us + (ballast_count * flush_us)
assert device_work_per_call > host_enqueue_per_call
```

Measured on a healthy run: kernel 110 us + flush 185 us x2 = **481 us of device work** against
**250 us of host enqueue** -- comfortably fed. Comparing 110 against 250 would have declared it
starved and sent the run chasing a non-existent defect.

Record `host_enqueue_us_per_call`, `flush_us`, `device_work_per_call_us` and the
`divergence_verdict` **on every row** so the guard itself can be audited. And confirm the guard
empirically: sweep the ballast (1/2/4/8) and check the result moves less than ~1% -- if it does
not move, you were never starved.

### Check determinism on EVERY output, not just the primary one

A race was found in which the primary output `y` was **bit-identical** to the baseline across
runs while the secondary output `x` returned **8 distinct results from 8 identical back-to-back
launches**. Every accuracy metric was blind to it, and a determinism check on `y` alone would
have been too.

**Run the bitwise determinism comparison over all outputs the kernel writes** -- including
in-out tensors, and including outputs the reference test does not check. A multi-output kernel
gives a race more places to hide, and the one you are watching is not necessarily the one that
moves.
