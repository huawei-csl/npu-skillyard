export const meta = {
  name: 'pto-pipeline-parallel',
  description: 'Parallel PTO kernel pipeline: decompose once, fan out per-stage kernel gen + compile + validate (one pto-stage-worker per stage), then benchmark serially and fuse. The parallel variant of the stage-pipeline agent; leaves stage-pipeline untouched.',
  whenToUse: 'When a stage_plan has multiple independent stages and you want per-stage kernel generation to run concurrently instead of one-at-a-time. Pass {source, output_dir, platform?, contract?, pto_python?, pto_isa_root?, include_dir?, pto_isa_repo?, devices?} as args.',
  phases: [
    { title: 'Preflight' },
    { title: 'Decompose' },
    { title: 'Stages' },
    { title: 'Benchmark' },
    { title: 'Optimize' },
    { title: 'Fuse' },
  ],
}

// ---- args ----
// args.source     (required) absolute path to the PyTorch algorithm source file
// args.output_dir (required) run directory; stage_plan.json + all artifacts land here
// args.platform   (optional) target platform tag, e.g. "a2a3"
// args.contract   (optional) a pre-agreed shape_contract object; if given, Phase 0 uses it verbatim
// args.pto_python (optional) PATH HINT: venv python with torch_npu. Resolved by Preflight
//                  (hint > $PTO_PYTHON > ./.venv/bin/python). Never auto-installed.
// args.pto_isa_root (optional) PATH HINT: pto-isa root. Resolved by Preflight
//                  (hint > $PTO_LIB_PATH > ./third_party/pto-isa); cloned if absent (see pto_isa_repo).
// args.include_dir (optional) PATH HINT: dir holding kernel_common.h. Resolved by Preflight
//                  (hint > $PTO_INCLUDE_DIR > ./examples/megakda-pto/include).
// args.pto_isa_repo (optional) git URL to clone pto-isa from when pto_isa_root is absent
//                  (else $PTO_ISA_REPO). If unset and the path is missing, Preflight STOPs.
// args.devices    (optional) list of NPU device indices to spread workers across (default ["0"])
// args.optimize   (optional) run the Optimize phase (pto-kernel-optimizer on the dominant
//                  stages) after Benchmark; default true. Set false to ship the correct
//                  baseline chain without the device-in-the-loop optimization campaign.
// args.optimize_top_n (optional) how many dominant stages to optimize (default 2)
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
const DEVICES = (ARGS?.devices && ARGS.devices.length) ? ARGS.devices : ['0']
const OPTIMIZE = ARGS?.optimize !== false            // default ON; pass optimize:false to skip
const OPTIMIZE_TOP_N = ARGS?.optimize_top_n ?? 2
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
2. Python with torch_npu (do NOT create a venv this run): hint=${PY_HINT ?? 'none'}, else $PTO_PYTHON,
   else ./.venv/bin/python. Verify it works AND sees an NPU:
   <py> -c "import torch, torch_npu; print(torch.npu.device_count())". If it errors or prints 0, add
   'torch_npu python' (and/or 'npu device') to \`missing\`.
3. pto-isa root: hint=${PTO_ISA_HINT ?? 'none'}, else $PTO_LIB_PATH, else ./third_party/pto-isa.
   If the resolved dir is ABSENT and a clone URL is available (hint=${PTO_ISA_REPO ?? 'none'}, else
   $PTO_ISA_REPO): git clone <url> <resolved_path> and set cloned_pto_isa=true. If absent and NO URL,
   add 'pto-isa' to \`missing\`. Verify the dir exists afterward.
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
log(`Preflight OK -- python=${PY}, pto_isa=${PTO_ISA}, include=${INCLUDE}${pf.cloned_pto_isa ? ' (pto-isa cloned)' : ''}`)

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
      agentType: 'pto-stage-worker',
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
// Phase: Fuse (Phase 7) -- gated on all-pass + benchmark, runs once
// ---------------------------------------------------------------------------
phase('Fuse')

let fusion = null
if (allPass && benchmark) {
  const fusePrompt = `You are running Phase 7 (Kernel Stitching / Fusion) of the PTO stage pipeline. The
gate is met: every stage PASSed on real NPU and Phase 6 benchmarks exist.

- stages: ${stages.join(', ')}
- output_dir: ${OUT}
- plan_path: ${PLAN_PATH}
- pto_python: ${PY}
- devices: ${DEVICES.join(', ')} (benchmark the fused kernel on ONE device, serially)

Compose the (now optimized) per-stage kernels into ONE GENERATED deliverable. DEFAULT to
LEAN-THEN-COMPOSE (cookbook §8.6P #21), NOT a from-scratch in-kernel merge-then-tune monolith:
the lean per-stage kernels are already the production slope (#12/#18), so share ONE layout across
them and chain each stage's launch_* on a single stream in one host call_kernel (stream ordering is
the free seam; fused slope = sum of lean slopes, ~0 fusion penalty). Reserve a tightly-coupled
in-kernel merge (intermediates resident, recurrence loop in-kernel, launch->1) ONLY for the one
stage where it is the sole remaining lever AND its cost is hideable -- a from-scratch 6-stage
in-kernel-FFTS monolith reliably blows the repair budget and runs SLOWER (the documented failure
mode). Write a residency + launch budget up front (Step 1). Generate via
pto-stage-kernel-generator-v2 (all rules). Compile -> validate end-to-end vs the composed
full-algorithm CPU-fp64 reference (up to 8 repairs). Benchmark fused-vs-chain on the production sweep.
Classify the deliverable: compute-fused (PASS) vs packaging-fused (INCOMPLETE Phase 7) -- a packaging-only
result that keeps ~all the chain's launches/GM round-trips is NOT a clean pass; record per-corner reasons.
PROVENANCE: the fused kernel must be GENERATED, never copied from any pre-existing kernel.

Return the fusion result object: {result, classification, launch_count:{target,actual}, residency, fallbacks_taken, speedup_vs_chain, kernel}. Final message IS the result.`

  fusion = await agent(fusePrompt, {
    label: 'fuse',
    phase: 'Fuse',
    agentType: 'general-purpose',
  })
} else {
  log('Skipping fusion (gate not met or no benchmark baseline).')
}

// ---------------------------------------------------------------------------
// Assemble pipeline_results-style summary (the orchestrator can persist it)
// ---------------------------------------------------------------------------
return {
  algorithm: SRC,
  platform: PLATFORM,
  output_dir: OUT,
  plan_path: PLAN_PATH,
  stages: results,
  summary: { pass: passed.length, fail: failed.length, total: stages.length },
  benchmarking: allPass ? (benchmark ?? 'attempted') : 'skipped (not all stages passed)',
  optimization: optimization ?? (OPTIMIZE ? (allPass && benchmark ? 'attempted' : 'skipped (gate not met)') : 'skipped (disabled)'),
  fusion: fusion ?? (allPass ? 'skipped (no benchmark baseline)' : 'skipped (not all stages passed)'),
}
