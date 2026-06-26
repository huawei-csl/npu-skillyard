# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is **not an application** — it is a **distributable Claude Code plugin** (skills,
subagents, a bundled MCP server, and one workflow) that drives an automated pipeline
turning a **PyTorch algorithm into validated, benchmarked PTO kernels for Ascend NPUs**.
The repo is laid out as a plugin *and* a single-entry marketplace (see Packaging below);
the pipeline itself operates on a *host/sibling* project that supplies the actual build
inputs (`.venv`, `third_party/pto-isa`, `examples/megakda-pto/include`). When you run
anything here, you are running against that host project, not this repo.

## Packaging (plugin + marketplace)

Components are at the **repo root** (plugin layout), not under `.claude/`:

- `.claude-plugin/plugin.json` — the plugin manifest.
- `.claude-plugin/marketplace.json` — single-entry marketplace, `"source": "."`.
- `skills/` — the four PTO skills.
- `agents/` — `stage-pipeline` (installable orchestrator) and `pto-stage-worker`.
- `include/kernel_common.h` — the single boilerplate header every kernel includes, **bundled**
  so no external example include dir is needed (Preflight drops it into the run dir).
- `.mcp.json` — bundles `npu-coding-mcp`, fetched+run from GitHub via `uvx --from git+…`
  (requires `uv` on the host; builds its doc indexes on first `serve`, no API key).
- `.claude/workflows/pto-pipeline-parallel.js` — **not** an installable plugin component;
  shipped for cloners/power-users (plugin-only users copy it in, or use the
  `stage-pipeline` agent instead). `.claude/settings.json` stays for in-repo dogfooding.

Run `claude plugin validate .` before publishing. See `README.md` for install/usage.

PTO = a tile-based ISA / C++ template library for the Ascend AI Core (Cube matrix engine +
Vec vector engine, separate cores). Kernels are `.cpp` translation units compiled with
`bisheng` in CCE mode and run on real NPU hardware.

## The pipeline (the central architecture)

A single algorithm flows through these phases. The same phase numbering appears across the
agents and the workflow — learn it once:

- **Phase 0 — Shape & Precision Contract.** One model-level contract (dtype, symbolic dims,
  tolerance, benchmark sweep) is the single source of truth for every per-stage shape. Each
  dim + the dtype gets a **source tier** (1 = evidenced in source, 2 = family convention,
  3 = guess). **Autonomy gate:** proceed only if everything is Tier 1; otherwise STOP and
  surface the proposed contract for human confirmation.
- **Phase 1 — Stage decomposition.** Split the algorithm into named tile-computation stages
  at dataflow boundaries. Per-stage shapes are *derived from* the Phase 0 contract.
  Output: `stage_plan.json` (top-level `shape_contract` + `stages[]`).
- **Phase 3 — Artifact generation.** Per stage: `validation_<stage>.py` and
  `benchmark_<stage>.py`. ABI comes from `StageSpec.abi` verbatim.
- **Phase 4 — Kernel generation.** Per stage: `kernel_<stage>.cpp`.
- **Phase 5 — Validation loop.** compile → msprof sim (advisory) → real-NPU (authoritative)
  → surgical repair → retry. Bounded repair budget.
- **Phase 6 — Benchmarking.** Gated on all stages passing; device-side `npu.Event` timing
  at the contract's production sweep.
- **Phase 6.5 — Optimize** (workflow only). Drive dominant stages toward their strong form.
- **Phase 7 — Fusion.** Stitch validated per-stage kernels into one deliverable.
- **Phase 8 — Report & Packaging.** Always runs last: organize the run dir into
  `ref/` (inputs) + `src/` (generated kernels/harnesses) + `reports/` (graphs + `report.md`),
  plot benchmark graphs (matplotlib), and write a top-level `README.md` narrating what was
  achieved, the blockers, and what was tried. A failed/partial run still gets a report.

### Two ways to drive it

1. **`stage-pipeline` agent** (`agents/stage-pipeline.md`) — full pipeline in one
   agent; fans Phase 3–5 out to one `stage-pipeline` sub-agent per stage (scoped to
   single-stage mode), then runs Phase 6/7 once.
2. **`pto-pipeline-parallel` workflow** (`.claude/workflows/pto-pipeline-parallel.js`) —
   decompose once, fan out one **`pto-stage-worker`** agent per stage in parallel, then
   benchmark *serially* (device timing must not contend), optimize, fuse. Invoke with
   `args: {source, output_dir, platform?, contract?, pto_python?, pto_isa_root?,
   include_dir?, pto_isa_repo?, bootstrap_venv?, devices?, optimize?, report?}`.

Both drivers begin with a **Preflight** step that resolves every path in priority order
(**explicit arg → env var → autodetect → documented default**) and validates it before any
work. CANN, `bisheng`, and the NPU device are detected and **STOP the run** with guidance —
never auto-installed. `pto-isa` is auto-cloned if absent (from `pto_isa_repo` >
`$PTO_ISA_REPO` > default `gitcode.com/cann/pto-isa`). `torch_npu` is **detect-and-stop by
default** (version-coupled to CANN; a wrong pin silently corrupts numerics) — opt in with
`bootstrap_venv: true` to create a venv and install torch/torch_npu matched to the detected
CANN version, then re-validate (still STOPs if it can't see the NPU). This keeps Phase 0 pure
(contract only) and turns "environment not ready" into an early, actionable failure. Env
vars: `$PTO_PYTHON`, `$PTO_LIB_PATH` (pto-isa), `$PTO_INCLUDE_DIR`, `$PTO_ISA_REPO`.

The skills (`torch-algorithm-to-pto-stages`, `pto-stage-artifact-generator-local`,
`pto-stage-kernel-generator-v2`, `pto-kernel-optimizer`) are the *per-phase* building
blocks both drivers invoke. `pto-stage-kernel-generator-v2/SKILL.md` is the largest and
most important file — it defines the C1–C32 critical rules, the archetype decision tree,
and the C24 compile recipe; its `references/` (`platform_model*.md` = `PLAT-§`,
`cookbook.md` = `COOK-§`, `cpu_sim_patterns.md` = `BUILD-§`) and `examples.md` (`EX-§`) are
binding and cited by stable ID throughout.

## Non-negotiable rules (these are what break runs when ignored)

- **Provenance boundary (hard rule).** NEVER read, grep, import, or copy from any
  pre-existing kernel on disk — including hand-tuned references and any other generator's
  output, in this repo or a sibling. Kernels are generated *only* from the StageSpec, the
  npu-coding-mcp ISA docs, and the skill cookbook. Borrowing invalidates validation. The
  boundary is on **implementation only** — reading shapes/dtypes/problem sizes/benchmark
  config from an external reference is allowed.
- **Real NPU is the authoritative gate; msprof sim is advisory.** The sim runs tiny sub-cap
  dims; a sim FAIL that can't represent the contract shape is `advisory-mismatch` — proceed
  to real NPU anyway. Never record an advisory sim mismatch as a stage FAIL.
- **CPU-fp64 reference.** The NPU has no float64 (`.double()` on an NPU tensor silently
  downcasts to fp32). The real-NPU gate computes the reference on **CPU in float64** and
  copies the kernel output back. On-device, no-`.cpu()` comparison is for the msprof sim
  path ONLY (a `.cpu()` sync hangs the simulator).
- **Coverage gate.** A PASS is valid only if the validated dims cover the *full* contract
  sweep including the largest/production size. Generated validation scripts sometimes
  hard-code a small subset — widen and re-run before recording PASS.
- **Contract-derived tolerance.** fp16/bf16 accumulate far from fp32 (expect rtol ~2e-2,
  not ~1e-5). A pass tighter than the dtype warrants is suspicious; an fp16 FAIL at 1e-5 is
  not a real bug.
- **Discovered constraint → amend the contract, never silently substitute** a different
  value for a user-supplied dim. If a free user-confirmed dim is infeasible, STOP.

## Environment & key commands

These run against the **host project**, not this repo. Resolve paths from the CANN env —
never hardcode a CANN version.

```bash
# Always first: sets $ASCEND_HOME_PATH; resolve bisheng + all includes from it
source /usr/local/Ascend/cann/set_env.sh
```

- **Python:** the host project's venv with `torch_npu` — resolved by Preflight
  (`pto_python` arg > `$PTO_PYTHON` > `.venv/bin/python`), not hardcoded. Always
  `import torch_npu`, `torch.npu.set_device(0)`, allocate with `device='npu'`, and **never**
  call `torch.npu.synchronize()` (hangs in the simulator).
- **Compile (Phase 5):** `bisheng` in CCE mode (`-xcce`, `-std=gnu++17`,
  `--cce-aicore-arch=dav-c220`) per rule **C24** in `pto-stage-kernel-generator-v2`.
  A CCE compile can exceed Bash's 2-min default — use a ~600s timeout; only a non-zero
  `bisheng` exit is a failure.
- **msprof simulator (advisory):** `msprof op simulator ... --soc-version=Ascend910B1
  <python> validation_<stage>.py --sim-mode kernel_<stage>.so`. Run in background under
  `timeout 1800` and poll (foreground Bash caps at 10 min). Output dir must be
  non-world-writable (`chmod 700`). Export the sim lib:
  `export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH"`.
- **Real-NPU validation (gate):** `<python> validation_<stage>.py "$(realpath
  kernel_<stage>.so)" --num-tests 12`. Pass the `.so` as an **absolute path** — `ctypes.CDLL`
  resolves a bare name via the loader path, not cwd.
- **Benchmark:** `<python> benchmark_<stage>.py "$(realpath kernel_<stage>.so)" --stage-spec
  spec_<stage>.json --out-json bench_<stage>.json`. Real hardware only; pass only flags the
  generated script actually defines.

There is no build/lint/test for this repo's own files — the "tests" are the generated
per-stage validation runs above.

## npu-coding-mcp (ISA source of truth)

The `npu-coding-mcp` server (bundled in `.mcp.json`; also enabled in `.claude/settings.json`
for in-repo dogfooding) serves PTO-ISA, AscendC,
CCE, and Runtime docs. Read `npu-coding://guide` for orientation. Use
`get_cpp_intrinsic` / `get_constraints` / `get_instruction` / `search_instructions` to
**verify every instruction family** (existence, dtype support, shape constraints) before
claiming it usable — never fabricate instruction names; record unverifiable ones as evidence
gaps.

## Conventions for generated kernel source

(Full detail: the C-rules in `pto-stage-kernel-generator-v2/SKILL.md`.)

- `#include "kernel_common.h"` only — no other includes (the header is bundled at
  `include/kernel_common.h`; it pulls CANN + pto-isa headers and defines `AICORE`).
- **ASCII-only** source and comments (bisheng rejects em-dashes, arrows, unicode).
- GM access only through MTE (`TLOAD`/`TSTORE`) — never scalar-index a `__gm__` pointer.
- UB budget: 192 KB (A2/A3) / 256 KB (A5); `static_assert` static UB layouts.
- Vec intrinsics under `#if defined(__DAV_C220_VEC__)`, Cube under `__DAV_C220_CUBE__`.
- Cross-core Cube↔Vec handshakes need *both* AIV sub-blocks (don't `if (vid != 0) return;`
  before a handshake); signal READY from the committing store pipe. Prefer stream-serialized
  split launches for stateless single contractions.
- Output of a generation skill is **raw file body** — no JSON envelope, no markdown fences,
  no commentary (the workflow framework handles file routing).

## Local settings

`.claude/settings.local.json.example` shows the recommended permission allowlist (bisheng,
msprof, the CANN env source, the two npu-coding-mcp read tools, arxiv/huggingface WebFetch).
Copy it to `.claude/settings.local.json` to pre-approve the common pipeline commands.
