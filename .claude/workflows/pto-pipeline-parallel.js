export const meta = {
  name: 'pto-pipeline-parallel',
  description: 'Parallel PTO kernel pipeline: decompose once, fan out per-stage kernel gen + compile + validate (one pto-stage-worker per stage), then benchmark serially and fuse. The parallel variant of the stage-pipeline agent; leaves stage-pipeline untouched.',
  whenToUse: 'When a stage_plan has multiple independent stages and you want per-stage kernel generation to run concurrently instead of one-at-a-time. Pass {source, output_dir, platform?, contract?, pto_python?, pto_isa_root?, include_dir?, pto_isa_repo?, devices?, optimize?, fuse?} as args.',
  phases: [
    { title: 'Preflight' },
    { title: 'Decompose' },
    { title: 'Stages' },
    { title: 'Benchmark' },
    { title: 'Optimize' },
    { title: 'Compose' },
    { title: 'Fuse' },
    { title: 'Report' },
  ],
}

// ---- args ----
// args.source     (required) absolute path to the PyTorch algorithm source file
// args.output_dir (required) run directory; stage_plan.json + all artifacts land here
// args.platform   (optional) target platform tag, e.g. "a2a3"
// args.contract   (optional) a pre-agreed shape_contract object; if given, Phase 0 uses it verbatim
// args.pto_python (optional) PATH HINT: venv python with torch_npu. Resolved by Preflight
//                  (hint > $PTO_PYTHON > ./.venv/bin/python). NOT auto-installed unless
//                  bootstrap_venv is set (see below).
// args.bootstrap_venv (optional, default FALSE) when no working torch_npu python is found,
//                  create a venv + pip install torch/torch_npu MATCHED to the detected CANN
//                  version, then RE-VALIDATE (STOP if it still can't see the NPU). Opt-in.
// args.bootstrap_venv_path (optional) where to create that venv (default ./.venv-npu).
// args.torch_version / args.torch_npu_version (optional) explicit pins for the bootstrap;
//                  if omitted, derive the torch_npu release from the detected CANN version.
// args.pto_isa_root (optional) PATH HINT: pto-isa root. Resolved by Preflight
//                  (hint > $PTO_LIB_PATH > ./third_party/pto-isa); cloned if absent (see pto_isa_repo).
// args.include_dir (optional) PATH HINT: dir holding kernel_common.h. Resolved by Preflight
//                  (hint > $PTO_INCLUDE_DIR > ./examples/megakda-pto/include).
// args.pto_isa_repo (optional) git URL to clone pto-isa from when pto_isa_root is absent
//                  (priority: arg > $PTO_ISA_REPO > default gitcode.com/cann/pto-isa).
// args.devices    (optional) list of NPU device indices to spread workers across (default ["0"])
// args.optimize   (optional) run the Optimize phase (pto-kernel-optimizer on the dominant
//                  stages) after Benchmark; default true. Set false to ship the correct
//                  baseline chain without the device-in-the-loop optimization campaign.
// args.optimize_top_n (optional) how many dominant stages to optimize (default 2)
// args.compose_mode (optional, default "ffts") how Part A stitches the integrated chain:
//                  "ffts" -- one launch, stages ordered ON-DEVICE at seams via SYNCALL<Mix>
//                  (removes the per-launch host-dispatch floor; DEFAULT; auto-falls back to
//                  host-stream if it cannot validate deterministically), or "host-stream" --
//                  stream-ordered launch_* calls. Part A always runs; this only picks the sync.
// args.fuse       (optional, default FALSE) OPT-IN Phase 7 Part B compute-fusion ("mix"). Only
//                  runs when set true; even then it ships ONLY if it is compute-fused AND beats
//                  the composed chain (packaging-only fusion is not built/shipped).
// args.report     (optional, default true) final Report phase: organize the run dir into
//                  ref/ src/ reports/, plot benchmark graphs, and write a README + report.md.
// args.make_graphs (optional, default true) plot PNG graphs from the benchmarks (needs
//                  matplotlib; pip-installed into the resolved python, else graphs are skipped).
// args may arrive as a parsed object or as a JSON string depending on how the
// workflow is invoked; accept both.
const ARGS = (typeof args === 'string') ? JSON.parse(args) : (args ?? {})
const SRC = ARGS?.source
const OUT = ARGS?.output_dir
if (!SRC || !OUT) {
  throw new Error('pto-pipeline-parallel requires args.source and args.output_dir')
}
const PLATFORM = ARGS?.platform ?? 'unspecified'
// Path HINTS only -- the Preflight phase resolves these to absolute, validated paths
// (priority: explicit arg > env var > autodetect > documented default).
const PY_HINT = ARGS?.pto_python ?? null
const PTO_ISA_HINT = ARGS?.pto_isa_root ?? null
const INCLUDE_HINT = ARGS?.include_dir ?? null
const PTO_ISA_REPO = ARGS?.pto_isa_repo ?? null
// Canonical public pto-isa source (the bundled npu-coding-mcp uses the same upstream).
const PTO_ISA_REPO_DEFAULT = 'https://gitcode.com/cann/pto-isa.git'
const BOOTSTRAP_VENV = ARGS?.bootstrap_venv === true   // opt-in; default OFF (detect-and-stop)
const BOOTSTRAP_VENV_PATH = ARGS?.bootstrap_venv_path ?? './.venv-npu'
const TORCH_VERSION = ARGS?.torch_version ?? null
const TORCH_NPU_VERSION = ARGS?.torch_npu_version ?? null
const DEVICES = (ARGS?.devices && ARGS.devices.length) ? ARGS.devices : ['0']
// The per-stage worker agent. Defaults to the bare name (works when running inside the
// cloned repo or when the agent is registered unqualified); pass the plugin-namespaced
// name (e.g. "npu-skillyard:pto-stage-worker") if your install exposes it that way.
const WORKER_AGENT = ARGS?.worker_agent ?? 'pto-stage-worker'
const OPTIMIZE = ARGS?.optimize !== false            // default ON; pass optimize:false to skip
const OPTIMIZE_TOP_N = ARGS?.optimize_top_n ?? 2
const COMPOSE_MODE = (ARGS?.compose_mode === 'host-stream') ? 'host-stream' : 'ffts'  // default ffts (auto-falls back to host-stream)
const FUSE = ARGS?.fuse === true                     // OPT-IN; default OFF -- fusion runs only if requested
const REPORT = ARGS?.report !== false                // default ON; organize run dir + write report
const MAKE_GRAPHS = ARGS?.make_graphs !== false      // default ON; plot benchmark graphs (needs matplotlib)
const PLAN_PATH = `${OUT}/stage_plan.json`

const STAGE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['name', 'result', 'repair_attempts'],
  properties: {
    name: { type: 'string' },
    result: { type: 'string', enum: ['PASS', 'FAIL'] },
    repair_attempts: { type: 'integer' },
    kernel_so: { type: ['string', 'null'] },
    accuracy: {
      type: ['object', 'null'],
      properties: {
        metric: { type: 'string' },
        value: { type: 'number' },
        tolerance: { type: 'number' },
        headroom_pct: { type: 'number' },
        validated_dims: { type: 'array', items: { type: 'string' } },
      },
    },
    last_error: { type: ['string', 'null'] },
    locked_dim: { type: ['object', 'null'] },
    notes: { type: ['string', 'null'] },
  },
}

const DECOMPOSE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['confidence', 'needs_confirmation', 'stages'],
  properties: {
    confidence: { type: 'string', enum: ['high', 'needs-confirmation'] },
    needs_confirmation: { type: 'boolean' },
    proposed_contract: { type: ['object', 'null'] },
    stages: { type: 'array', items: { type: 'string' } },
    note: { type: ['string', 'null'] },
  },
}

const PREFLIGHT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ok', 'resolved', 'missing'],
  properties: {
    ok: { type: 'boolean' },
    resolved: {
      type: 'object',
      properties: {
        ascend_home: { type: ['string', 'null'] },
        bisheng: { type: ['string', 'null'] },
        python: { type: ['string', 'null'] },
        pto_isa_root: { type: ['string', 'null'] },
        include_dir: { type: ['string', 'null'] },
      },
    },
    missing: { type: 'array', items: { type: 'string' } },
    cloned_pto_isa: { type: ['boolean', 'null'] },
    bootstrapped_venv: { type: ['boolean', 'null'] },
    note: { type: ['string', 'null'] },
  },
}

const REPORT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ok'],
  properties: {
    ok: { type: 'boolean' },
    structure: { type: ['object', 'null'] },
    graphs: { type: 'array', items: { type: 'string' } },
    readme: { type: ['string', 'null'] },
    report: { type: ['string', 'null'] },
    note: { type: ['string', 'null'] },
  },
}

const CHAIN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['result'],
  properties: {
    result: { type: 'string' },   // "PASS" | "FAIL (kept per-stage kernels)" | "skipped (<reason>)"
    mode: { type: ['string', 'null'] },        // "ffts" | "host-stream" (the sync actually shipped)
    fell_back: { type: ['boolean', 'null'] },  // true if ffts was requested but host-stream shipped
    kernel: { type: ['string', 'null'] },
    validated_end_to_end: { type: ['boolean', 'null'] },
    repair_attempts: { type: ['integer', 'null'] },
    benchmark: { type: ['object', 'null'] },
    note: { type: ['string', 'null'] },
  },
}

// ---------------------------------------------------------------------------
// Phase: Preflight -- resolve + VALIDATE the build environment ONCE, before any
// decomposition. Un-provisionable prerequisites (CANN, bisheng, torch_npu, NPU)
// are detected and STOP the run with guidance; pto-isa is cloned if absent.
// ---------------------------------------------------------------------------
phase('Preflight')

const preflightPrompt = `You are the PREFLIGHT step of the PTO pipeline. Resolve and VALIDATE the
build environment ONCE. Do NOT decompose, generate kernels, or benchmark. Work from the host
project root as cwd. Resolve each path in priority order: explicit hint > environment variable
> autodetect > documented default, then VERIFY it actually exists/works.

1. CANN (cannot be auto-installed): source /usr/local/Ascend/cann/set_env.sh. Verify $ASCEND_HOME_PATH
   is set, \`bisheng\` resolves from it, and the simulator lib dir exists
   ($ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib). If any is missing, add it to \`missing\`.
2. Python with torch_npu. Resolve hint=${PY_HINT ?? 'none'}, else $PTO_PYTHON, else ./.venv/bin/python.
   VALIDATE: <py> -c "import torch, torch_npu; print(torch.npu.device_count())".
   - If it imports AND prints > 0: use it (set resolved.python to its ABSOLUTE path).
   - If it FAILS and bootstrap_venv is ${BOOTSTRAP_VENV ? 'TRUE' : 'FALSE'}:
     * FALSE (default): add 'torch_npu python' (and/or 'npu device' if import works but count==0)
       to \`missing\` -- do NOT create a venv.
     * TRUE: BOOTSTRAP, then re-validate --
       a. Detect the CANN version from $ASCEND_HOME_PATH (the version dir name, or
          $ASCEND_HOME_PATH/version.cfg / .../ascend_toolkit_install.info).
       b. Create a venv at ${BOOTSTRAP_VENV_PATH} using the system python3 (>=3.9), upgrade pip.
       c. pip install torch + torch_npu MATCHED to that CANN version. Use explicit pins if given
          (torch=${TORCH_VERSION ?? 'derive'}, torch_npu=${TORCH_NPU_VERSION ?? 'derive'}); otherwise
          select the torch_npu release that targets the detected CANN version from the official Ascend
          pytorch install table. If you CANNOT determine a safe CANN<->torch_npu match, do NOT guess --
          add 'torch_npu version (unknown CANN match)' to \`missing\` and STOP. ALSO pip install
          matplotlib into the same venv (the Report phase needs it for graphs) -- a matplotlib failure
          is non-fatal, do not STOP on it.
       d. RE-VALIDATE with the SAME import+device_count check on ${BOOTSTRAP_VENV_PATH}/bin/python.
          Accept it (set resolved.python, set bootstrapped_venv=true) ONLY if it now prints > 0;
          otherwise add 'torch_npu python (bootstrap failed)' to \`missing\`. Never proceed on a
          failed/partial install.
3. pto-isa root: hint=${PTO_ISA_HINT ?? 'none'}, else $PTO_LIB_PATH, else ./third_party/pto-isa.
   If the resolved dir is ABSENT, clone it: URL = hint ${PTO_ISA_REPO ?? '(none)'} > $PTO_ISA_REPO >
   default ${PTO_ISA_REPO_DEFAULT}. Run git clone <url> <resolved_path> and set cloned_pto_isa=true;
   verify pto-inst.hpp exists under <resolved_path> afterward. Only add 'pto-isa' to \`missing\` if the
   clone itself fails (e.g. no network).
4. include dir (must contain kernel_common.h -- the plugin SHIPS this header, so this NEVER STOPs):
   if hint=${INCLUDE_HINT ?? 'none'} or $PTO_INCLUDE_DIR points to a dir that already contains
   kernel_common.h, use it verbatim. OTHERWISE create <output_dir>/include/ and put kernel_common.h
   there: copy it from the bundled plugin header ($CLAUDE_PLUGIN_ROOT/include/kernel_common.h if that
   env var is set), else write this exact content to <output_dir>/include/kernel_common.h --
   ----------------------------------------------------------------------
   #pragma once
   #ifndef __CPU_SIM
   #include "acl/acl.h"
   #include <runtime/runtime/rt_ffts.h>
   #endif
   #include <cmath>
   #include <cstdint>
   #if defined(__CCE_AICORE__)
   #include <pto/pto-inst.hpp>
   using namespace pto;
   #elif defined(__CPU_SIM)
   #include <pto/pto-inst.hpp>
   using namespace pto;
   #endif
   #ifndef AICORE
   #define AICORE [aicore]
   #endif
   ----------------------------------------------------------------------
   Set include_dir to that absolute dir. kernel_common.h only needs CANN (acl/rt_ffts, resolved from
   $ASCEND_HOME_PATH) and pto-isa (pto-inst.hpp, resolved from pto_isa_root) on the -I path, both of
   which steps 1 and 3 already validated -- so include_dir is never in \`missing\`.

Return ABSOLUTE resolved paths in \`resolved\` {ascend_home, bisheng, python, pto_isa_root, include_dir},
the \`missing\` list, cloned_pto_isa, and ok = (missing is empty). Your final message IS the structured
result -- no prose.`

const pf = await agent(preflightPrompt, {
  label: 'preflight',
  phase: 'Preflight',
  schema: PREFLIGHT_SCHEMA,
  agentType: 'general-purpose',
})

if (!pf || !pf.ok) {
  log('Preflight failed -- environment not ready.')
  return {
    status: 'needs-setup',
    missing: pf?.missing ?? ['preflight agent failed'],
    note: pf?.note ?? 'Resolve the missing prerequisites and re-run. CANN, bisheng, torch_npu, and the NPU device cannot be auto-installed; pass pto_isa_repo (or set $PTO_ISA_REPO) to auto-clone pto-isa.',
  }
}

// Resolved, validated, absolute paths -- everything downstream uses these.
const PY = pf.resolved.python
const PTO_ISA = pf.resolved.pto_isa_root
const INCLUDE = pf.resolved.include_dir
log(`Preflight OK -- python=${PY}, pto_isa=${PTO_ISA}, include=${INCLUDE}${pf.cloned_pto_isa ? ' (pto-isa cloned)' : ''}${pf.bootstrapped_venv ? ' (venv bootstrapped)' : ''}`)

// ---------------------------------------------------------------------------
// Phase: Decompose (Phase 0 contract + Phase 1 stage plan) -- runs once
// ---------------------------------------------------------------------------
phase('Decompose')

const decomposePrompt = `You are running Phase 0 (Shape & Precision Contract) and Phase 1
(Stage Decomposition) of the PTO stage pipeline. Do NOT generate kernels, validate, or
benchmark -- only decompose and persist the plan.

Inputs:
- AlgorithmSource: ${SRC}
- target platform: ${PLATFORM}
- output_dir: ${OUT}
- python interpreter (torch_npu): ${PY}
${ARGS?.contract ? `- A shape_contract was SUPPLIED; use it verbatim, confidence high:\n${JSON.stringify(ARGS.contract)}` : '- No contract supplied; research and propose one.'}

Steps:
1. Source CANN env: source /usr/local/Ascend/cann/set_env.sh
2. Phase 0 contract: ${ARGS?.contract
    ? 'use the supplied contract verbatim, confidence high.'
    : `apply the torch-algorithm-to-pto-stages skill's "Shape & Precision Contract" process. Assign every dim + the dtype a value + source tier (Tier 1 source-explicit / Tier 2 family-convention / Tier 3 guess). AUTONOMY GATE: proceed only if EVERY dim and the dtype is Tier 1. If ANY dim/dtype is Tier 2 or Tier 3, set needs_confirmation=true, return the proposed_contract, and STOP -- do NOT write a plan or decompose.`}
3. Phase 1: apply the torch-algorithm-to-pto-stages skill to decompose into stages, passing the Phase 0 contract so per-stage interface shapes are DERIVED from the contract's symbolic dims + dtype.
4. Write ${PLAN_PATH} including the top-level shape_contract block and one entry per stage (with each stage's StageSpec/abi).

Provenance boundary: reading SHAPES/DTYPES/problem sizes/benchmark config from references is allowed; reading any existing kernel's IMPLEMENTATION source is barred.

Return: confidence, needs_confirmation, the ordered list of stage names, and (if needs_confirmation) the proposed_contract. Your final message IS the structured result -- no prose.`

const decomp = await agent(decomposePrompt, {
  label: 'decompose',
  phase: 'Decompose',
  schema: DECOMPOSE_SCHEMA,
  agentType: 'general-purpose',
})

if (!decomp) {
  return { error: 'decomposition agent failed' }
}
if (decomp.needs_confirmation) {
  log('Contract is not fully evidence-backed -- stopping for user confirmation (autonomy gate).')
  return {
    status: 'needs-confirmation',
    proposed_contract: decomp.proposed_contract ?? null,
    note: decomp.note ?? 'Re-invoke with args.contract set to the confirmed contract.',
  }
}

const stages = decomp.stages ?? []
log(`Decomposed into ${stages.length} stage(s): ${stages.join(', ')}`)
if (!stages.length) {
  return { error: 'decomposition produced no stages' }
}

// ---------------------------------------------------------------------------
// Phase: Stages -- fan out one pto-stage-worker per stage (PARALLEL)
// Generation + compile are CPU-bound and fully parallel. Real-NPU validation
// shares the device(s); workers are spread across args.devices round-robin.
// ---------------------------------------------------------------------------
phase('Stages')

const workerPrompt = (stageName, device) => `Process ONE stage of an already-decomposed PTO pipeline.

- stage_name: ${stageName}
- output_dir: ${OUT}
- plan_path: ${PLAN_PATH}
- pto_python: ${PY}
- pto_isa_root: ${PTO_ISA}
- include_dir: ${INCLUDE}
- NPU device: before any torch_npu work, export ASCEND_RT_VISIBLE_DEVICES=${device} (then torch.npu.set_device(0) -- index 0 within the visible set).

Run Phase 3 (artifacts) -> Phase 4 (kernel) -> Phase 5 (compile/validate/repair, max 5 attempts) for THIS stage only. Read your stage entry + the shape_contract from plan_path. Honor the coverage gate (validate the full contract sweep incl. the largest size) and the CPU-fp64 reference rule. Return the structured per-stage result.`

const stageResults = await parallel(
  stages.map((s, i) => () =>
    agent(workerPrompt(s, DEVICES[i % DEVICES.length]), {
      label: `stage:${s}`,
      phase: 'Stages',
      schema: STAGE_SCHEMA,
      agentType: WORKER_AGENT,
    })
  )
)

const results = stageResults.filter(Boolean)
const passed = results.filter(r => r.result === 'PASS')
const failed = results.filter(r => r.result !== 'PASS')
log(`Stages complete: ${passed.length} PASS / ${failed.length} FAIL (of ${stages.length})`)

const allPass = results.length === stages.length && failed.length === 0

// ---------------------------------------------------------------------------
// Phase: Benchmark (Phase 6) -- gated on all-pass, runs SERIALLY (one agent,
// stages one-at-a-time) so real-NPU timing is not corrupted by device contention.
// ---------------------------------------------------------------------------
phase('Benchmark')

let benchmark = null
if (!allPass) {
  log('Gate not met -- skipping benchmark and fusion (not all stages passed).')
} else {
  const benchPrompt = `You are running Phase 6 (Benchmarking) of the PTO stage pipeline. EVERY stage
PASSed on real NPU, so the gate is met. Benchmark each stage on real NPU, ONE AT A TIME
(never run two timed harnesses on the same device concurrently -- it corrupts latency).

- stages: ${stages.join(', ')}
- output_dir: ${OUT}
- plan_path: ${PLAN_PATH}
- pto_python: ${PY}
- devices available: ${DEVICES.join(', ')} (use one device; serialize across stages)

For each stage: ensure benchmark_<stage>.py exists (regenerate via pto-stage-artifact-generator-local if missing), then run it on real NPU at the contract's production sweep (shape_contract.sweep_axis), with: npu.Event device timer, a 256 MiB int8 L2 flush before every timed call, warmup 5, repeats 50/iters 15+. Pass the .so as an ABSOLUTE path. Collect mean/min/max/median/p95/stddev (ns) per stage and write a combined benchmarks.json in output_dir.

Return a short summary object: {benchmarked: [stage names], benchmarks_json: path}. Final message IS the result.`

  benchmark = await agent(benchPrompt, {
    label: 'benchmark',
    phase: 'Benchmark',
    agentType: 'general-purpose',
  })
}

// ---------------------------------------------------------------------------
// Phase: Optimize (Phase 6.5) -- drive the dominant stages toward their strong
// form with the pto-kernel-optimizer skill (device-in-the-loop). Gated on
// all-pass + benchmark; correctness/determinism re-gated inside the skill.
// This is the seam the generator leaves open: generation ships a correct
// baseline (with OPTIMIZER-TARGET markers); this phase closes those markers.
// ---------------------------------------------------------------------------
phase('Optimize')

let optimization = null
if (OPTIMIZE && allPass && benchmark) {
  const optPrompt = `You are running Phase 6.5 (Optimize) of the PTO stage pipeline. Every stage PASSed
and Phase 6 benchmarks exist. Apply the pto-kernel-optimizer skill to the DOMINANT stages.

- output_dir: ${OUT}
- plan_path: ${PLAN_PATH}
- benchmarks_json: ${OUT}/benchmarks.json
- pto_python: ${PY}
- devices available: ${DEVICES.join(', ')} (use ONE device; never run two timed harnesses on it at once)
- optimize the top ${OPTIMIZE_TOP_N} stages by per-work-unit SLOPE (fall back to median share if no slope is recorded)

Method (pto-kernel-optimizer): for each dominant stage, decompose the slope, classify the
bottleneck, apply the matching lever, and RE-MEASURE with a within-process paired A/B before
trusting any win. PRIORITIZE the levers the generator already flagged: scan each stage .cpp for
'OPTIMIZER-TARGET(' banner markers and attack those first (they name the exact pattern + reason).
Bound the work: at most 3 levers per stage; stop at the irreducible floor (serial recurrence /
wholesale-clone boundary) and record it honestly.

HARD GATES (every step): re-validate vs the fp64 reference at small AND production sizes and
re-run the determinism check after EVERY change. Never re-add a flush/barrier to mask a race.
Keep the last-good kernel as the deployable fallback; if a lever regresses or breaks determinism,
revert it. PROVENANCE: study a reference's STRUCTURE only; the kernel stays generated/derived.
Overwrite a stage's kernel_<stage>.cpp/.so and its benchmark entry IN PLACE only after a paired win.

Return: {optimized: [{stage, lever, before_slope, after_slope, speedup_x, kept: bool, floor_reason?}],
markers_closed: [..], benchmarks_json: path}. Final message IS the result.`

  optimization = await agent(optPrompt, {
    label: 'optimize',
    phase: 'Optimize',
    agentType: 'general-purpose',
  })
} else {
  log(OPTIMIZE ? 'Skipping optimize (gate not met or no benchmark baseline).'
               : 'Skipping optimize (args.optimize = false).')
}

// ---------------------------------------------------------------------------
// Phase: Compose (Phase 7 Part A) -- DEFAULT. Stitch the validated per-stage
// kernels into ONE integrated deliverable: a single call_kernel that chains each
// stage's launch on one stream (intermediates via GM, lean-then-compose, ~0
// penalty). This is the canonical end deliverable; the fused "mix" is opt-in below.
// ---------------------------------------------------------------------------
phase('Compose')

let chain = null
if (!(allPass && benchmark)) {
  log('Skipping compose (gate not met or no benchmark baseline); per-stage kernels remain the output.')
} else {
  const composePrompt = `You are running Phase 7 Part A (Compose the chain) of the PTO stage pipeline. Every
stage PASSed on real NPU and Phase 6 benchmarks exist. Produce the ONE integrated deliverable
kernel_chain_<algo>.cpp/.so the run ships by default. Do NOT attempt in-kernel compute-fusion / residency
/ overlap merges here -- that is the separate opt-in Part B.

- stages (dataflow order): ${stages.join(', ')}
- output_dir: ${OUT}
- plan_path: ${PLAN_PATH}
- pto_python: ${PY}
- devices: ${DEVICES.join(', ')} (benchmark on ONE device, serially)
- compose_mode (requested): ${COMPOSE_MODE}

Two sync modes for stitching the validated per-stage kernels; both allocate one GM buffer per inter-stage
intermediate and share ONE layout (intermediates transit GM; on-chip CV FIFO is A5-only):
- "ffts" (DEFAULT): a SINGLE-launch kernel where stages hand off through GM but ORDERING is enforced
  ON-DEVICE at stage SEAMS with SYNCALL<Mix> (COOK-§8.6 / C6 -- SYNCALL is for stage seams / a single-launch
  multi-stage kernel, NEVER a per-tile Cube<->Vec edge). Removes the per-launch host-dispatch floor (the win
  when launch-overhead-bound). Its failure mode is a run-to-run COHERENCY RACE, not a logic bug.
- "host-stream": one call_kernel issuing each stage's launch_* in dataflow order on ONE stream
  (COOK-§8.6P #21 lean-then-compose). Correct by construction, ~0 penalty when launches pre-enqueue.

If compose_mode is "host-stream", build that directly. Otherwise build "ffts" and validate it END-TO-END
vs the composed full-algorithm CPU-fp64 reference AND run a DETERMINISM check (repeat runs must match) --
up to 5 repairs. AUTO-FALLBACK: if the ffts chain cannot be made to validate DETERMINISTICALLY within
budget (a race surgical edits do not fix in ~2 attempts), STOP repairing it and build the "host-stream"
chain instead (correct by construction) -- NEVER ship a flaky ffts kernel. Benchmark whichever chain you
ship on the Phase-6 sweep. PROVENANCE: GENERATED from the stage plan + your per-stage kernels + ISA docs +
cookbook only -- do NOT read any pre-existing kernel's source.

Return: {result:"PASS"|"FAIL (kept per-stage kernels)"|"skipped (<reason>)", mode:"ffts"|"host-stream",
fell_back:<bool>, kernel, validated_end_to_end, repair_attempts, benchmark, note}. Final message IS the result.`

  chain = await agent(composePrompt, {
    label: 'compose',
    phase: 'Compose',
    schema: CHAIN_SCHEMA,
    agentType: 'general-purpose',
  })
  if (chain) log(`Compose: ${chain.result}${chain.mode ? ` (${chain.mode}${chain.fell_back ? ', fell back from ffts' : ''})` : ''}${chain.kernel ? ` -- ${chain.kernel}` : ''}`)
}

// ---------------------------------------------------------------------------
// Phase: Fuse (Phase 7 Part B) -- OPT-IN (args.fuse). Even when requested, ships a
// fused kernel ONLY if it is compute-fused AND beats the composed chain; else keeps it.
// ---------------------------------------------------------------------------
phase('Fuse')

let fusion = null
if (!FUSE) {
  log('Skipping fusion (not requested; pass fuse:true to enable). Chain is canonical.')
} else if (!(allPass && benchmark)) {
  log('Skipping fusion (gate not met or no benchmark baseline).')
} else {
  const fusePrompt = `You are running Phase 7 Part B (compute-fusion / the "mix") of the PTO stage pipeline.
Fusion was REQUESTED, every stage PASSed on real NPU, and Phase 6 benchmarks exist. Part A already
produced the composed chain (kernel_chain_<algo>) -- your job is the tightly-coupled in-kernel merge,
and it ships ONLY if it measurably BEATS that composed chain.

- stages: ${stages.join(', ')}
- output_dir: ${OUT}
- plan_path: ${PLAN_PATH}
- pto_python: ${PY}
- devices: ${DEVICES.join(', ')} (benchmark the fused kernel on ONE device, serially)

Fusion is CONDITIONAL on a MEASURED win, not automatic. First confirm a concrete win exists (else
record "fusion":"skipped (no overlap/streaming win available; chain is canonical)" and stop): a
same-core sub-chain that goes L1/UB-resident (COOK-§8.6P #2), a GM-heavy Cube op with a GM-light Vec
partner to overlap (#9, proven by a #10 noop-floor probe), or a large intermediate that must be
STREAMED to scale. a2a3 reality: a Cube<->Vec intermediate MUST transit GM (on-chip CV FIFO is
A5-only), so a GM round-trip on that edge is EXPECTED -- residency applies only WITHIN a same-core
sub-chain. Pure launch-collapse is NOT a win. NEVER put SYNCALL<Mix> on a per-tile Cube<->Vec hand-off
(#5). Pursue an in-kernel merge ONLY when it is the sole remaining lever AND its cost is hideable
(the plain composed chain already exists from Part A -- do not just rebuild it). Write the win +
residency/launch budget up front. Generate via pto-stage-kernel-generator-v2 (all rules); compile ->
validate end-to-end vs the composed full-algorithm CPU-fp64 reference (up to 8 repairs); benchmark
fused vs the Part A composed chain (kernel_chain) AND vs the Phase-6 reference/roofline. It is a PASS
ONLY if compute-fused AND it BEATS the composed chain. If the build collapses to packaging-only (same
launches/GM traffic, no overlap/residency/streaming win), do NOT ship it -- record "not attempted
(packaging-only: <reason>)" and keep the composed chain. PROVENANCE: the fused kernel must be
GENERATED, never copied from any pre-existing kernel.

Return the fusion result object: {result, classification:"compute-fused", win_captured, launch_count:{target,actual}, residency, speedup_vs_chain, kernel}. Final message IS the result.`

  fusion = await agent(fusePrompt, {
    label: 'fuse',
    phase: 'Fuse',
    agentType: 'general-purpose',
  })
}

// ---------------------------------------------------------------------------
// Assemble the machine summary (also handed to the Report phase, and persisted)
// ---------------------------------------------------------------------------
const pipelineSummary = {
  algorithm: SRC,
  platform: PLATFORM,
  output_dir: OUT,
  plan_path: PLAN_PATH,
  stages: results,
  summary: { pass: passed.length, fail: failed.length, total: stages.length },
  benchmarking: allPass ? (benchmark ?? 'attempted') : 'skipped (not all stages passed)',
  optimization: optimization ?? (OPTIMIZE ? (allPass && benchmark ? 'attempted' : 'skipped (gate not met)') : 'skipped (disabled)'),
  chain: chain ?? (allPass && benchmark ? 'attempted' : 'skipped (gate not met)'),
  fusion: fusion ?? (!FUSE ? 'skipped (not requested; chain is canonical)' : (allPass && benchmark ? 'attempted' : 'skipped (gate not met)')),
}

// ---------------------------------------------------------------------------
// Phase: Report -- always runs last. Organize the flat run dir into ref/ src/
// reports/, plot benchmark graphs, and write a human README + report.md. A
// failed/partial run still gets a report (blockers + what was tried).
// ---------------------------------------------------------------------------
phase('Report')

let report = null
if (REPORT) {
  const reportPrompt = `You are the REPORT & PACKAGING step (the FINAL phase) of the PTO pipeline. Do NOT
modify kernels, re-run validation, or re-benchmark. Organize the run directory, generate benchmark
graphs, and write a human-readable report from the data that already exists.

- output_dir: ${OUT}
- source algorithm: ${SRC}
- python (for plotting): ${PY}
- make graphs: ${MAKE_GRAPHS}
- Pipeline summary (AUTHORITATIVE -- use these numbers, do not re-derive):
${JSON.stringify(pipelineSummary)}

1. Create subdirs under output_dir: ref/, src/, reports/.
2. Organize files (MOVE, do not duplicate; leave .tmp/ where it is):
   - ref/      : a COPY of the source algorithm (${SRC}), plus stage_plan.json and spec_*.json
   - src/      : kernel_*.cpp, kernel_*.so, kernel_fused_*.{cpp,so}, validation_*.py, benchmark_*.py
   - reports/  : benchmarks.json and bench_*.json
   Update any path references you write so they point at the new locations.
3. Graphs (only if make graphs is true AND benchmarks.json exists). benchmarks.json is schema
   "benchmarks_v1": { sweep_axis:{dim:D, values:[...]}, stages:{ <name>:{ "per_"+D : { "<val>":
   {mean,min,max,median,p95,stddev} in ns }, slope_per_unit_ns, optimized?:{before_slope_ns,
   after_slope_ns,speedup_x,kept} } } }. Read the per-stage series from the "per_<D>" map (the key
   is literally "per_" + sweep_axis.dim). Ensure matplotlib in ${PY} (\`${PY} -c "import matplotlib"\`
   else \`${PY} -m pip install --quiet matplotlib\`); if it cannot be installed (e.g. no network) SKIP
   graphs and note it -- do NOT fail. Convert ns -> us for axes. Write PNGs into reports/ (plot only
   what the data supports; skip any with no data):
   - latency_vs_sweep.png  : x = sweep values D, y = mean latency (us), one line per stage.
   - stage_breakdown.png   : bar of mean latency at the LARGEST sweep value, one bar per stage,
                             sorted descending (highlights the dominant stage).
   - slope_by_stage.png    : bar of slope_per_unit_ns per stage (the per-work-unit production cost).
   - optimized_before_after.png : for stages with an "optimized" block, grouped before/after
                             slope bars annotated with speedup_x (skip if no stage was optimized).
   - fused_vs_chain.png    : ONLY if the summary's fusion has speedup_vs_chain -- fused vs chain per
                             sweep point (data from the summary above, NOT benchmarks.json).
   - accuracy_vs_tol.png   : per-stage relative error vs tolerance from the summary's
                             stages[].accuracy {value,tolerance,headroom_pct} (NOT benchmarks.json);
                             log-scale y if the errors span orders of magnitude.
   Label axes + units and title each.
4. reports/report.md: the shape_contract, a per-stage table (result | rel-err vs tol | headroom% |
   repair_attempts | last_error), a benchmark table, the graphs embedded via ![](relative.png),
   the fusion classification + speedups, and the optimization outcomes.
5. output_dir/README.md: a top-level NARRATIVE -- what the run ACHIEVED (stages passed, the headline
   benchmark, fusion result), the BLOCKERS and what was TRIED (per failed stage: repair_attempts +
   last_error; any locked-dim contract amendments; sim advisory-mismatches; optimizer markers/floors;
   fallbacks taken in fusion), how to REPRODUCE (the validate/benchmark commands), and the final
   directory layout (ref/ src/ reports/). Be honest about partial or failed runs -- this is the
   primary artifact a human reads first.

Return ok, the created structure, the list of graph PNGs, and the README + report.md paths.
Your final message IS the structured result -- no prose.`

  report = await agent(reportPrompt, {
    label: 'report',
    phase: 'Report',
    schema: REPORT_SCHEMA,
    agentType: 'general-purpose',
  })
  if (report) log(`Report written -- ${OUT}/README.md (${(report.graphs ?? []).length} graph(s))`)
} else {
  log('Skipping report (args.report = false).')
}

return { ...pipelineSummary, report: report ?? (REPORT ? 'attempted' : 'skipped (disabled)') }
