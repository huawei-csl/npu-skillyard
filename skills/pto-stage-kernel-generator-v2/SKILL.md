---
name: pto-stage-kernel-generator-v2
description: "Generate one high-quality PTO kernel for one stage spec. Produces a single compile-oriented C++ translation unit grounded in PTO ISA and Ascend platform constraints."
---

# PTO Stage Kernel Generator v2

Generate one stage kernel that is mathematically faithful to the stage spec
and structurally valid for Ascend PTO compilation and runtime.

Be strict: avoid inventing unsupported math or ABI details. Prefer conservative
launch-domain guards and explicit evidence gaps over speculative lowerings.
For semantically specified stages with concrete outputs, never return
`skeleton_only` final kernels.

---

## Input

- `StageSpec` (`.json`): one stage specification file. This may come from either:
  - A `stage_spec_v1` file: `{"schema_version":"stage_spec_v1", "algorithm":..., "stage":{...}}`
  - A stage plan entry: a single stage object extracted from a `stage_plan.json` `stages[]` array

**Regardless of source format**, the following fields are REQUIRED and always present:
  - `name` — stage name (string)
  - `inputs` — list of `{name, shape, dtype, role}` objects
  - `outputs` — list of `{name, shape, dtype, role}` objects
  - `problem` — dict of dimension constants (e.g., `{"tile_size":64, "feature_dim":128}`)

**Shapes** may be symbolic (`["B","HV","NT","BT","K"]`) or concrete (`[1,256,8,128]`).
Each symbolic dimension name maps to a value in `problem` or a workflow-level
override (`--n-seq`, `--l-seg`). Concrete shapes should be decomposed into
symbolic names where possible — record any unmapped dimensions in evidence gaps.

**Fields that may be present depending on source format:**
  - `instruction_families` — list of PTO instruction names (from MCP verification)
  - `lowering_hint` — free-text tile constraints and lowering guidance
  - `reference_source` — pure-torch reference function (may be inline string or separate file)
  - `stage_family`, `stage_subfamily`, `stage_traits` — semantic classification (stage_spec_v1 only)
  - `production_dimensions` — production-scale dim values (stage_spec_v1 only; if absent, use `problem` values)
  - `code_region` — source location (stage plan only; informational)
  - `evidence_gaps` — known uncertainties (record, do not guess)

**When `production_dimensions` is absent**, use `problem` values for all dimension sizing.
**When `stage_traits` is absent**, classify the stage from `instruction_families` and `reference_source`.

## Required Local References

Read and follow these before writing code. They are binding constraints.

| File | Stable ID prefix | Contents |
|------|------------------|----------|
| `references/platform_model.md` | `PLAT-§` | Hardware model, memory hierarchy, legal/illegal paths, UB budget, A2/A3 |
| `references/platform_model_a5.md` | `PLAT-§` | Hardware model, memory hierarchy, legal/illegal paths, UB budget, A5 |
| `references/cookbook.md` | `COOK-§` | Compile-proven PTO code patterns (type surfaces, sync protocols, GEMM, layout) |
| `references/cpu_sim_patterns.md` | `BUILD-§` | Compile flags, call_kernel template, platform guards, msprof validation recipe |
| `examples.md` | `EX-§` | Annotated failure patterns and full archetype examples (FA, MatMul, LayerNorm) |
| `REVIEWER.md` | `REV-§` | Reviewer/fixer mode rules (read only when invoked as reviewer) |

The cookbook is self-contained. Reuse the embedded code portions and adapt them.
Do not depend on opening foreign example files during generation.

## Output Contract

Return only the complete C++ translation unit as raw text:

- no JSON envelope
- no markdown fences
- no commentary before or after the code
- the response is the file body — the workflow framework handles file routing

Never write to a hard-coded file path. Never wrap the output in
`{"outputs": {"Kernel": "..."}}` — the framework adds that wrapper automatically.

---

## Provenance Boundary (hard rule)

Generated kernels MUST be produced ONLY from:
- the `StageSpec` (and any explicitly staged inputs),
- the PTO ISA documentation (npu-coding-mcp -- `get_cpp_intrinsic`, `get_constraints`, etc.),
- this skill's own cookbook and `examples.md`.

NEVER read, open, grep, import, or copy from any pre-existing kernel anywhere on
disk -- including hand-tuned reference kernels and any other generator's output,
whether in this repository or a sibling/related one. Such kernels exist as an
independent correctness/performance oracle for humans ONLY. If a generated kernel borrows
from one of them, the validation and benchmark are meaningless -- you would be
grading a hand-optimized kernel, not a generated one. Treat any access to those
files during generation as a hard failure. This supersedes and strengthens A4.

---

## Rule Tiers

Every rule in this skill belongs to one of three tiers. Tier determines severity.

| Tier | Meaning | Consequence of violation |
|------|---------|--------------------------|
| 🔴 **CRITICAL** | Hardware safety, compile correctness | NPU crash, compile failure, silent data corruption |
| 🟡 **STANDARD** | Algorithmic correctness, code quality | Wrong results, validator rejection, poor performance |
| 🟢 **ADVISORY** | Style, readability, maintainability | Harder review, future brittleness |

Rules are marked with their tier on first appearance. In case of conflict,
higher tiers override lower tiers.

**Performance forms are not optional.** "Poor performance" sits in STANDARD, but the
*algorithmic form* (the Strong-Form Defaults below) is chosen with the archetype, not
deferred: emit the strong form by default, and if it cannot validate in budget, fall back
to the correct baseline ONLY with an explicit `OPTIMIZER-TARGET` marker (see Strong-Form
Defaults). A silent baseline is a rule violation, not a free pass.

---

## Pre-Generation Checklist

Complete these steps in order before writing any code. Each step references
a cookbook section or platform model section for details.

```
1. □ Parse StageSpec — verify required keys: name, inputs, outputs, problem
     Map symbolic shape dims to problem values or workflow overrides.
     If shapes are concrete, infer symbolic names and record unmapped dims as evidence gaps.
2. □ Classify archetype — use the decision tree below
2b. □ Select the strong algorithmic FORM — Strong-Form Defaults table (just below the
     decision tree). Match each trigger against this stage's math/shape/loop structure.
     Default to the strong form; if it cannot validate in budget, emit the correct
     baseline + an OPTIMIZER-TARGET marker. A matched trigger with no strong form and
     no marker is a rule violation.
3. □ Identify instruction families — from instruction_families, or derive from reference_source if absent
4. □ Select type surface family — COOK-§0.5 (Family A, B, or C)
5. □ Compute UB budget — PLAT-§UB, verify fits in 192KB (A2/A3) or 256KB (A5)
     If Cube path (TMATMUL): add L1 Mat staging tiles and L0 Left/Right tiles to budget (C26).
6. □ Select sync protocol — from archetype (Vec-only flags vs cross-core FFTS)
7. □ Plan work distribution — COOK-§1.68/§1.69 (grid-stride or varlen)
     Derive elements_per_iteration from problem and input shapes.
8. □ Choose tile shapes — fixed compile-time tiles, runtime outer loops
9. □ Draft UB address map — COOK-§1.6/§3 with static_assert guard
     If Cube path: allocate addresses for L1 Mat staging tiles and L0 tiles (C26).
10. □ Plan data path to TMATMUL — GM -> L1 Mat (TLOAD) -> TEXTRACT -> L0 Left/Right -> TMATMUL (C26).
      Never use TMOV from Vec to Left/Right. Pad M to 16 if needed for TEXTRACT alignment.
11. □ Plan Vec pipeline data flow — after every TLOAD, push data to pipeline with TMULS(x, x, 1.0f)
      before any Vec op (TMUL, TADD, TEXP, etc.) reads it (C27).
12. □ Generate kernel — follow the structure below
```

### Archetype Decision Tree

Use `StageSpec.stage.instruction_families` as the primary signal.
`stage_family` is semantic guidance only — it tells you WHAT the stage computes,
not HOW to lower it.

```
IF reference_source / instruction_families contain a matrix contraction
(TMATMUL, TMATMUL_ACC, TTRI, einsum, @, torch.matmul, torch.triu, torch.tril):
  │
  ├─ FIRST classify the contraction SHAPE — this, not the presence of einsum/@,
  │   decides Cube vs Vec (see S3):
  │   • dense matrix-MATRIX (M, N, and contraction dim all >= 16, realistically
  │     >= 64), batchable into one issue        → Cube (TMATMUL)
  │   • matrix-VECTOR (M = 1, a GEMV), OR rank-1 OUTER product (contraction
  │     dim = 1), OR a tiny contraction (K, V <= ~16) carried INSIDE a
  │     sequential / loop-carried scan          → Vec, NOT Cube
  │       (TROWEXPAND/TCOLEXPAND + TMUL + TCOLSUM/TROWSUM)
  │     Reason: the Cube fixed cost — L1->L0 staging, TEXTRACT with M padded
  │     to 16, and a per-issue FFTS Vec<->Cube handshake — is not amortized by
  │     a small vector op, and that cross-core handshake inside a scan loop is a
  │     deadlock / correctness hazard (see C6). This is a TILE-based Vec
  │     contraction, not a forbidden scalar fallback.
  │
  ├─ Cube path — stage has Vec pre/post-processing?
  │   YES → cube_vec_pipeline  → COOK-§8, §8.5-§8.12, EX-§3
  │        HOW to wire Cube<->Vec: if the stage is just Vec-prep -> ONE Cube
  │        contraction with NO loop-carried state crossing the boundary, DEFAULT
  │        to a stream-serialized SPLIT LAUNCH (Vec-prep kernel then Cube kernel,
  │        no in-kernel cross-core flags). An in-kernel handshake buys nothing here
  │        (no overlap, no resident state) and risks the cross-core coherency race
  │        (C6 / COOK-§8.6). Use an in-kernel handshake ONLY when state stays
  │        resident across an iteration loop.
  │   NO  → cube_only          → COOK-§7, §8.7-§8.9, EX-§3
  │   Vec path (GEMV / outer-product / small loop-carried) → treat as vec_only
  │     below, with the recurrent-state layout rules of S9 + C28.
  │
ELSE (pure Vec ops: TLOAD, TADD, TMULS, TMOV, TSTORE, TEXP, no Cube signals):
  │
  ├─ variable-length sequences required?
  │   YES → varlen_tail        → COOK-§1.69, §11
  │   NO  → vec_only           → COOK-§1, §1.5, §1.6, §1.65-§1.67, §2, §6, EX-§2
  │
IF stage is underspecified (no reference_source, no concrete outputs):
  └─ → skeleton_only (last resort only) → COOK-§17
```

### Strong-Form Defaults (the performance-FORM decision — apply WITH the archetype)

The archetype tree picks Cube vs Vec. This table picks the **algorithmic form**. Every
trigger is a STRUCTURAL property of the operation/dataflow you can read straight off the
StageSpec (the math, the shapes, the loop structure) — **none names a specific algorithm or
kernel**. If a stage's dataflow matches a trigger, the strong form is the DEFAULT emission,
not an optimization to defer to a later phase. These change the op COUNT or the GM traffic,
so they move the per-work-unit slope (the production cost) — not just the fixed intercept.

| Structural trigger (read from the StageSpec math/shape/loop — NOT a kernel name) | Default strong form | Cookbook |
|---|---|---|
| Inverting a unit-(lower/upper)-triangular `M = I + L` with N larger than the cube fractal size — i.e. a `torch.inverse`/solve of a `tril`/`triu`, or a Neumann/iterative series run over the FULL N | **block-recursive fractal inverse** (invert the F×F diagonal blocks, then resolve off-diagonals) — NOT full-N Neumann doubling | §8.6P #13 |
| A value is **loop-carried across the work-unit (tile/chunk/block) loop** — a recurrence `S_{n+1} = f(S_n, x_n)` | keep `S` **resident in a named UB tile**, update in place; park to GM only the one irreducible cross-core transit per iteration — never reload it | §8.6P #20 |
| **Two consecutive ops on the SAME engine** with a producer→consumer dependency (Vec→Vec, Cube→Cube) | a local `pipe_barrier(PIPE_V / PIPE_FIX)` between them — **never a GM store+load round-trip** to "commit" the intermediate | §8.6P #16 |
| A **per-row / per-element scan or reduction** over a tile | a **block-resident** scan kept in UB, with loop-invariant masks/constants hoisted out of the work loop — NOT a per-row GM round-trip | §8.6P #17 |
| A **contraction-axis scalar** multiplies a matmul operand (a gate / scale / beta applied on the dimension being contracted) | **fold the scalar into the matmul operand** so the raw tensor loads Cube-direct — no separate Vec pre-scale + GM round-trip | §8.6P #19 |
| Composing ≥2 already-correct stages into ONE deliverable | **lean-then-compose**: lean each stage standalone, share ONE layout, chain `launch_*` on one stream — NOT a from-scratch in-kernel merge-then-tune | §8.6P #21 |

**Default-or-mark contract** (this is what makes a default mandatory WITHOUT breaking
correctness-first). Emit the strong form by default. If the strong form cannot be made to
VALIDATE within the repair budget, fall back to the correct baseline **and emit a banner
annotation**:

```
// OPTIMIZER-TARGET(<pattern#>): <stage> uses <baseline form>; strong form is
//   <one line: what + why it pays>; blocked by <the concrete reason it didn't validate>.
```

The fallback ships — correct beats fast — but the marker tells the optimizer phase exactly
which lever to attack and why generation could not land it. A baseline emitted with **no
marker** asserts "the strong form does not apply to this stage"; the absence of a marker is
itself a claim you must be able to defend at review. This is the seam between generation
(correct baseline) and the `pto-kernel-optimizer` skill (drives the marked stages to the
strong form, device-in-the-loop).

---

## Generation Rules

### 🔴 CRITICAL Rules (C-series)

Violations cause NPU crashes, compile failures, or silent data corruption.

**C1. GM access is ONLY through MTE.**
The Ascend AI Core cannot address global memory directly. Any scalar indexing
of a `__gm__` pointer (`ptr[idx]`, `ptr[idx] = val`) crashes the NPU into
Alarm state requiring a hardware reset. ALL data transfer between GM and UB
uses `TLOAD` (GM→UB) and `TSTORE` (UB→GM) with `GlobalTensor` descriptors.
This applies to mask generation, workspace init, output writes, assertion
checks — everything. Never `reinterpret_cast` then scalar-index; always wrap
in `GlobalTensor` and use TLOAD/TSTORE. → PLAT-§Illegal

**C2. Include and namespace gating.**
Use ONLY `#include "kernel_common.h"` as the single include at the top of the file.
This header provides ALL necessary includes: `acl/acl.h`, `runtime/rt_ffts.h`, 
`pto/pto-inst.hpp`, and all required system headers. Do NOT add any other 
`#include` directives (no `#include <runtime/rt_ffts.h>`, no `#include <cmath>`, 
no `#include "acl/acl.h>`, etc.). The `kernel_common.h` header already defines 
`AICORE` as `[aicore]` when `__CCE_AICORE__` is defined, so do not redefine it.
Keep `using namespace pto;` and all PTO tile/template instantiations under 
`#if defined(__CCE_AICORE__)` guard so the host Bisheng pass never sees PTO types. 
Do not use indirection macros (e.g. `WF_HAS_PTO_STAGE_IMPL`) for gating — use 
direct `defined(__CCE_AICORE__)`. → PLAT-§Manual

**C3. Valid host/device split.**
Required structure with EXACTLY ONE `launch_*` definition:

```cpp
// Device compute function (inside #if defined(__CCE_AICORE__) || defined(__CPU_SIM) guard)
#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
AICORE void stage_kernel(...) {
    // device code
}
#endif

// Launch entrypoint (OUTSIDE the #if guard, defined exactly once)
extern "C" __global__ AICORE void launch_*(...) {
#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
    stage_kernel(...);
#endif
}

// Host wrapper (always present, outside all guards)
// Use <<<...>>> syntax — msprof op simulator intercepts it.
extern "C" void call_kernel(...) {
    uint32_t ffts_len = 0; uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    uint32_t blocks = (block_dim > 0) ? block_dim : 1;
    launch_*<<<blocks, nullptr, stream>>>(...);
}
#endif
}
```

CRITICAL: The `launch_*` function must be defined EXACTLY ONCE in the entire file.
Do NOT define it inside `#if defined(__CCE_AICORE__)` and then again in `#else`.
The launch function body should use `#if defined(__CCE_AICORE__) || defined(__CPU_SIM)` 
to conditionally call `stage_kernel`, but the launch function itself is defined only once.
Do NOT provide `#if !defined(AICORE) #define AICORE __aicore__ #endif` — 
`kernel_common.h` already defines `AICORE`. → COOK-§1

**C3.1. CPU-SIM UB pointer arithmetic.**
When using `ub<T>()` helper to cast UB offsets to pointers, add CPU-SIM guard:

```cpp
template<typename T> AICORE inline __ubuf__ T* ub(int32_t offset) {
#ifdef __CPU_SIM
    // CPU-SIM: add offset to UB base from memory model
    char* ub_base = pto::NPUMemoryModel::Instance().GetUBBase();
    return reinterpret_cast<__ubuf__ T*>(ub_base + offset);
#else
    // NPU: cast offset directly (hardware UB is at fixed address)
    return reinterpret_cast<__ubuf__ T*>(static_cast<uintptr_t>(offset));
#endif
}
```

This is REQUIRED for CPU-SIM because UB memory is dynamically allocated in CPU-SIM
but at a fixed hardware address on real NPU. → CPU-SIM

**C4. Approved type surface only.**
Use exactly one of the three families from COOK-§0.5. Do not invent aliases
(`VecShape`, `VecStride`, `VecGlobal`, `MakeGlobal`). Do not mix partially
qualified and partially invented APIs. For Family B dynamic shapes, keep the
exact unqualified `Shape<1,1,1,DYNAMIC,DYNAMIC>` and `Stride<1,1,1,DYNAMIC,1>`
spellings. → COOK-§0.5

**C5. Cube layout rules.**
Mat tiles: `BLayout::ColMajor, SLayout::RowMajor` (L1Mat) or
`BLayout::RowMajor, SLayout::ColMajor` (L1MatZN). NEVER `SLayout::NoneBox`
on Mat tiles. NEVER swap L1Mat↔L1MatZN destinations for TEXTRACT.
Transposed operands must route through `TRESHAPE(L1MatZN, L1Mat)` first.
Left operand: `BLayout::RowMajor, SLayout::RowMajor`.
Right operand: `BLayout::RowMajor, SLayout::ColMajor`.
Accumulator: `BLayout::ColMajor, SLayout::RowMajor`. → COOK-§8.5, §8.7, §13

**C6. Cross-core FFTS bootstrap.**
Never `wait_flag_dev(N)` without a prior producer `set_cross_flag` on
iteration 0. Bootstrap free-slot signals before the first consumer wait.
On A2/A3, Cube-side `wait_flag_dev` for V→C reduces over both Vec subblocks;
if `vid != 0` returns early, Cube cannot safely wait on that V→C flag.
Use `pipe_barrier(PIPE_ALL)` only for intra-core sync, never cross-core. → COOK-§8, §8.6

**For an ALL-CORE barrier, use the library `SYNCALL<Mix>` -- do NOT hand-roll.**
`aicore exception 507015` (invisible to the simulator -- C25) is most often a
hand-rolled cross-core barrier gone wrong: a non-deterministic race that passes a
few runs, then lets a core read un-committed GM (all-zeros), then hard-faults or
deadlocks at scale. When you just need a full Cube+Vec barrier (between fused
stages, or a one-shot global sync), call `pto::SYNCALL<pto::SyncCoreType::Mix>()`
(`pto/common/pto_instr.hpp`; a2a3 impl `SyncAll.hpp`) -- it is correct by
construction, uses reserved system flags 11-14, and does its own `dcci`. It caps
`block_dim` at the AIC count (`kCvMaxCores=25`; see A6). Hand-roll a barrier only
when `SYNCALL<Mix>` is measurably the bottleneck.

**SCOPE: `SYNCALL<Mix>` is for STAGE BOUNDARIES / one-shot global syncs ONLY -- NEVER
as a per-chunk / per-item Cube<->Vec hand-off.** Its all-core scope plus the built-in
bulk `dcci` make it ruinously expensive in a hot loop. VALIDATED: a fused KDA kernel
that used `SYNCALL<Mix>` (~44/chunk) as the per-chunk Cube<->Vec hand-off was both
RACY (the bulk dcci masked an in-place region-reuse bug) and 3.5-6.6x SLOWER than the
per-stage split-launch chain. For per-item/per-chunk hand-offs use the point-to-point
3-rule recipe (COOK-§8.6): same-pipe FFTS signal + NO bulk dcci + a distinct GM region
per cross-core-live intermediate. That recipe benchmarked 6.6-7.3x faster than the
SYNCALL/dcci version and FASTER than split-launch, deterministic 30/30 at HV=32.
For making a SINGLE-LAUNCH FUSED multi-stage kernel actually beat the split-launch
chain (rendezvous-count diagnostic, L1-resident Cube-only sub-chains via TMOV Acc->Mat,
both-AIV vid-split of row-parallel Vec prep, scan-as-Cube-matmul), see **COOK-§8.6P**.
Launch-count collapse alone buys nothing -- the chain already overlaps its sub-launches.

**Manual fine-grained handshake (the FALLBACK): signal READY from the storing pipe.**
When you do need a per-slot / pipelined cross-core Cube<->Vec handshake (finer than
an all-core barrier), it IS achievable on dav-c220 (see COOK-§8.6) -- still NOT a
platform limitation. The 507015 here is a cross-core ORDERING bug: the producer's
READY `ffts_cross_core_sync` must be issued FROM THE PIPE THAT COMMITTED THE GM
STORE -- `PIPE_FIX` after a Cube L0C->GM `TSTORE`, `PIPE_MTE3` after a Vec/UB->GM
`TSTORE` -- with a `pipe_barrier(PIPE_ALL)` drain immediately before it. Signalling
from the wrong pipe (e.g. MTE3 after a Cube FIX store) lets the consumer run before
the write commits. Do NOT add a bulk `dcci` on the hand-off data: the DMA
`TSTORE`->GM->`TLOAD` path never passes through the scalar Data Cache that `dcci`
manages, so a same-pipe-ordered signal already makes the read coherent (COOK-§8.6).
`dcci` is for a scalar software signal word only. A non-deterministic, run-to-run race
is almost always IN-PLACE GM REGION REUSE, not a cache miss -- fix it with a distinct
GM region per cross-core-live intermediate (COOK-§8.6 rule 3), never by flushing.

**Validated scope (real-NPU).** A single-kernel Cube<->Vec handshake is reliable on
dav-c220 INCLUDING in a LOOP (per-chunk / per-iteration), once the iterated
flag-counter + both-vids protocol is followed (COOK-§8.6): both AIV sub-blocks must
run every mode-2 cross-core signal/wait (do NOT `if (vid != 0) return;` before a
handshake -- see C12), the back-edge FREE flag is bootstrapped on its producer
side, each flagID is balanced per iteration, the AIC drains a Vec->Cube reduce once
(not once-per-AIV), and READY/FREE are signalled from the committing store pipe
after a `pipe_barrier(PIPE_ALL)`. The earlier looped-handshake deadlocks were a
mode-2 reduce starved by a silenced second AIV -- a fixable protocol bug, not a
platform limit. A stream-serialized SPLIT launch (Cube kernel then Vec kernel, no
cross-core flags) remains a valid simpler alternative; the layout-robust Vec
micro-GEMM (S3) is the no-Cube fallback for small contractions.

**Performance caveat -- correct is not faster.** A single-launch IN-KERNEL chunk
loop using this handshake validated 8/8 but ran ~4% SLOWER than a stream-serialized
host sub-launch loop for a GEMM-work-bound per-chunk recurrence. Measured reasons
on dav-c220: (a) stream sub-launches already OVERLAP with device execution
(wall-clock approx device-only -- no host-dispatch cost to recover); (b) keeping
state S resident in UB still needs a per-chunk S->GM snapshot whenever a Cube L1
GEMM consumes S, so "no GM round-trip" is only half true; (c) a single-buffered
in-kernel handshake re-serializes Cube/Vec the same way the stream did, while
adding ~4 FFTS round-trips/chunk. Do NOT fuse a per-chunk recurrence into one
launch just to cut launch count -- only pursue it when you can ALSO double-buffer
(overlap Cube_{t+1} with Vec_post_t) or keep S in L0C/L1 to remove the GM snapshot.
For a GEMM-bound stage, launch-count reduction alone is not a win.

**SCOPE of that caveat -- it is the GEMM-WORK-BOUND regime, not a blanket rule.**
The ~4% data point above is a single stage whose per-chunk GEMM work already
dwarfs launch overhead, so collapsing launches recovers little. The OPPOSITE regime
is common and inverts the conclusion: when per-stage work is small relative to the
~5-10us per-launch dispatch floor (small dims, and ESPECIALLY a multi-STAGE fused
kernel that would otherwise issue ~one launch per sub-step of every stage), the
wall-clock is LAUNCH-OVERHEAD-BOUND and reducing launches IS the dominant win.
Diagnose the regime before deciding: if the chain's measured per-launch time is
flat across a sweep (does not grow with the work dim), you are launch-bound and
fusion helps; if it scales with the work, you are compute-bound and launch-count
reduction alone will not.

**Residency and the in-kernel loop are ORTHOGONAL to the FFTS handshake -- do not
conflate them (G).** "Avoid the in-kernel Cube<->Vec handshake" (a real coherency
risk) does NOT mean "round-trip intermediates through GM" or "issue the outer loop
from the host." Those are three independent decisions:
- You can keep a `[C,C]`/`[K,V]` intermediate RESIDENT in UB/L1/L0 across adjacent
  sub-steps with ordinary intra-core `pipe_barrier(PIPE_ALL)` sync -- no cross-core
  flags needed when the producer and consumer run on the same core/pipe sequence.
- You can run the outer/recurrence loop INSIDE one kernel carrying state on-chip and
  STILL use stream-serialized split launches' moral equivalent inside it (sequential
  Cube then Vec, intra-core barriers) -- the in-kernel handshake is only required
  when you additionally want cross-core OVERLAP, which a serial recurrence cannot use.
So a serial loop-carried dependency is a reason to skip OVERLAP, never a reason to
skip residency or to push the loop back to the host.

**Recipe -- keep intermediates resident / carry recurrent state on-chip (F).** When
fusing stages (or a recurrence) into one kernel, the default should be on-chip
residency, not GM hand-off:
- Allocate the inter-stage intermediate (e.g. the `[C,C]` L and its inverse, `u`/`w`,
  the `[K,V]` state S) ONCE in the UB/L1 address map (COOK-§1.6/§3, static_assert the
  budget). Producer sub-step writes it to that UB/L1 tile; consumer sub-step reads the
  SAME tile. Separate the two with `pipe_barrier(PIPE_ALL)` (intra-core) -- no GM
  TSTORE/TLOAD between them.
- For a Cube consumer (TMATMUL) of a resident operand, stage UB->L1 Mat (TLOAD from
  UB is legal) -> TEXTRACT -> L0, keeping the operand on-chip; only snapshot to GM if
  the L1/L0 budget genuinely cannot hold it (record that as a per-intermediate
  fallback reason -- see stage-pipeline Phase 7 Step 1 budget).
- For a recurrence, hold S in UB (or L0C/L1 if a Cube GEMM consumes it) across the
  in-kernel loop; update S in place each iteration. The ONLY forced GM snapshot is
  when a Cube L1 GEMM must consume S and the budget cannot keep S in L1 -- minimize,
  do not default to, that snapshot.
- Budget reality at C=128, K=V=128, fp16 on A2/A3 (192KB UB): a `[128,128]` fp16 tile
  is 32KB; fp32 is 64KB. Several resident at once is feasible; size the map before
  concluding "must go to GM."

**Cube is the DEFAULT for a dense contraction -- do not "play it safe" with Vec.**
For a dense matrix-MATRIX contraction (all of M, N, K >= 16) the Cube path is the
expected lowering, and the in-kernel handshake above is a PROVEN, validated recipe
(see the data point below), not an experimental risk. Picking the Vec micro-GEMM for
a dense GEMM to dodge the handshake is an order-of-magnitude perf regression -- it is
NOT a valid "correctness-first" choice. If you want Cube throughput with zero
cross-core flags, a stream-serialized SPLIT launch (Cube kernel then Vec kernel) is
always available and carries none of the handshake risk -- reach for that before you
reach for a Vec micro-GEMM. Do NOT preemptively avoid Cube because some earlier kernel
faulted: a wrong-pipe single-shot signal is fixable with the rule above. The "correct
is not faster" caveat above is SCOPED to FUSING a per-chunk recurrence into one launch
-- it is NOT an argument against running the matmuls on Cube. The only real subtlety:
do not assume a pipelined in-kernel handshake is fixed by the signal pipe alone (see
above) -- but a split launch sidesteps that entirely.

**Validated (dav-c220, real NPU).** A unit-lower-triangular inverse (Neumann
doubling, two dense `TMATMUL`s per step) with an in-kernel mode-2 Cube<->Vec
counting-semaphore handshake -- the exact pattern an earlier run abandoned for a
Vec micro-GEMM after a 507015 fault -- ran CLEAN (no fault, no deadlock), exact to
~5e-8, and **9.8x-37.6x faster than the Vec fallback** (the Cube version is nearly
flat ~80-86us across BT 32/48/64 while the Vec micro-GEMM scaled ~quadratically
0.78/1.83/3.23 ms). The 507015 was a fixable cross-core ORDERING bug, not a Cube
limit. Lesson: for a contraction-heavy stage, treat the Vec micro-GEMM as a
LAST-RESORT fallback and exhaust the Cube path (split-launch or a correctly-signalled
in-kernel handshake) first -- the perf gap is order-of-magnitude, not marginal.

**C7. UB capacity and alignment.**
UB: 192KB (196608 bytes) on A2/A3, 256KB on A5. 32-byte alignment.
Compute summed live-buffer bytes for all concurrently live UB tiles.
Emit `static_assert(kMaxUbAddr <= 196608)` when using static TASSIGN addresses.
Derive UB usage from PLAT-§UB before choosing addresses. Never guess
hard-coded UB layouts without a budget derivation and guard. → PLAT-§UB, COOK-§1.6, §4

**C8. Vec subblock UB sharing.**
UB is shared by both Vec sub-blocks (vid=0 and vid=1). Static TASSIGN
addresses are not private per sub-block. If both vids stay active,
partition UB address ranges explicitly. Otherwise return on nonzero vid
before any shared TASSIGN addresses are reused.
**Exception -- cross-core stages:** you may NOT early-return vid 1 in a stage that
does a mode-2 FFTS Cube<->Vec handshake -- the Vec->Cube reduce needs both AIVs to
signal or it deadlocks (C12, COOK-§8.6). There, keep both vids running the
handshake and gate the shared-UB DATA work to `vid == 0` by branching, not by
returning. → PLAT-§Topology, COOK-§1.5

**C9. Pipe barrier correctness.**
`pipe_barrier(PIPE_ALL)` is required after every TLOAD and TSTORE to maintain
MTE↔Vec ordering. Do not relax this. Do not blanket-barrier after every
operation either — use narrow `set_flag`/`wait_flag` pairs for pipeline
overlap. 8 event IDs per core (EVENT_ID0–7); reuse only after full retirement. → PLAT-§Sync, §Events

**C12. Target feature guards for Vec/Cube-specific code.**
All Vec-specific intrinsics (`set_vector_mask`, `set_mask_norm`, `get_subblockid`,
`vector_dup`, etc.) MUST be inside `#if defined(__DAV_C220_VEC__)` guards.
All Cube-specific intrinsics MUST be inside `#if defined(__DAV_C220_CUBE__)` guards.
The standard Vec-only preamble is:
```cpp
#if defined(__DAV_C220_VEC__)
  auto vid = get_subblockid();
  if (vid != 0) return;
  set_mask_norm();
  set_vector_mask(-1, -1);
  // ... rest of Vec code ...
#endif
```
**Exception -- cross-core handshakes need BOTH vids.** The `if (vid != 0) return;`
above is correct for a Vec-ONLY kernel with shared UB (C8). It is WRONG before a
mode-2 cross-core Cube<->Vec handshake: a Vec->Cube reduce requires BOTH AIV
sub-blocks to signal, so an early-returned vid 1 starves it and deadlocks --
immediately, even at niter=1 (COOK-§8.6). In a cross-core stage, run every
`ffts_cross_core_sync` / `wait_flag_dev` on BOTH vids and gate only the DATA work
to `vid == 0` (branch, do not return).

Do NOT call Vec intrinsics outside this guard. The compiler will reject them
with "does not support the given target feature" errors.

**Pure-Vec stage: wrap the ENTIRE device body in `#if defined(__DAV_C220_VEC__)`,
not just the mask preamble.** Many library Vec ops expand to Vec intrinsics under
the hood -- `TTRI`, `TADD`, `TSUB`, `TMUL`, `TMULS`, `TCOLSUM`, `TROWSUM`, `TEXP`,
`TSEL` -- so even a stage that calls only these still emits `set_vector_mask` /
`vector_dup` / `vsub` etc. The kernel is compiled once per subtarget; in the Cube
(`__DAV_C220_CUBE__`) pass those intrinsics are illegal and fail with "does not
support the given target feature." For a stage with NO Cube work, put the whole
compute body under `#if defined(__DAV_C220_VEC__) ... #endif` (the Cube pass then
compiles an empty function). Only a mixed Cube+Vec stage splits the body across
`#if __DAV_C220_CUBE__` / `#elif __DAV_C220_VEC__` branches. → PLAT-§Topology

**C10. NaN and uninitialized data.**
Never introduce NaN-producing placeholder arithmetic or uninitialized
accumulation as a repair shortcut.

When building decay / transition terms of the form `exp(g_i - g_j)`, arrange the
exponent argument to be `<= 0` (build only the causal half, e.g. `j <= i`, and
prefer the non-positive form `exp(g_last - g)`) so factored exponentials do not
overflow fp32 to Inf. Zero-initialize product/accumulator tiles before use so a
stale lane cannot seed a NaN.

**C11. Runtime strides match packed layout.**
Compile-time caps (`BT_CAP`, `K_CAP`, `V_CAP`) guard budgets only; they are
not substitute leading dimensions. Use runtime packed strides (`bt`, `k_pad`,
`v_pad`) when addressing GM/L1/L0 operands. Do not call `copy_gm_to_l1`,
`copy_l0c_to_gm`, or `gemm_v0` with cap-based leading dimensions when the
live matrix was packed with narrower runtime strides. → COOK-§13

**C13. ASCII-only source files.**
Use ONLY ASCII characters (0-127) in ALL source code and comments. The bisheng
compiler REJECTS non-ASCII characters with "unexpected character" errors.
FORBIDDEN characters include:
- Em-dashes (—), en-dashes (–), curly quotes ("", ''), arrows (→, ←, ↔)
- Mathematical symbols (×, ÷, ≤, ≥, ∑, ∏, √, ∞)
- Any Unicode character outside the ASCII range

REQUIRED ASCII alternatives:
- Use `--` instead of em-dash (—) or en-dash (–)
- Use `->` instead of arrow (→)
- Use `<=` instead of ≤, `>=` instead of ≥
- Use `*` instead of ×, `/` instead of ÷
- Use straight quotes (`"` and `'`) instead of curly quotes

This rule applies to ALL text in the file: banner comments, inline comments,
string literals, and code. → COMPILER

**C14. Tile alignment and minimum sizes.**
PTO tiles have strict alignment requirements. FORBIDDEN tile configurations:
- `UbND<T, 1, 1>` or any 1x1 tile — violates 32-byte alignment
- `UbND<float, R, C>` where `R * C * sizeof(float) < 32` bytes
- Tiles with `Cols < 8` for float32 (minimum 8 floats = 32 bytes)

REQUIRED minimum tile sizes:
- For `float32`: minimum `UbND<float, 1, 8>` (1 row × 8 cols = 32 bytes)
- For `float16`: minimum `UbND<half, 1, 16>` (1 row × 16 cols = 32 bytes)
- For scalar values: use `UbND<float, 1, 8>` and access element 0 via `GetValue(0)`

When you need to store a single scalar result (e.g., from a reduction), use:
```cpp
UbND<float, 1, 8, 1, DYNAMIC> scalar_tile(1);  // Valid: 1×8 = 32 bytes
TASSIGN(scalar_tile, SCALAR_UB_ADDR);
TROWSUM(scalar_tile, source_tile, temp_tile);
float result = scalar_tile.GetValue(0);
```

Do NOT use `UbND<T, 1, 1>` — it will fail compilation with alignment errors. → PLAT-§Alignment

**C15. Reduction instruction correctness.**
PTO reduction instructions have specific input/output shape requirements:

- `TROWSUM(dst, src, temp)`: Reduces each row of `src` to a single value in `dst`
  - `src`: `UbND<T, R, C>` (R rows, C cols)
  - `dst`: `UbND<T, R, 8>` (R rows, minimum 8 cols for alignment)
  - `temp`: `UbND<T, R, 8>` (workspace, same shape as dst)
  - Result: `dst.GetValue(i)` contains sum of row `i` from `src`

- `TCOLSUM(dst, src)`: Reduces each column of `src` to a single value in `dst`
  - `src`: `UbND<T, R, C>` (R rows, C cols)
  - `dst`: `UbND<T, 1, C>` (1 row, C cols) — NOT `UbND<T, 1, 8>`
  - Result: `dst.GetValue(j)` contains sum of column `j` from `src`

Common mistake: Using `TCOLSUM` to reduce a 1×K tile to 1×1. This is WRONG.
Correct approach for reducing 1×K to scalar:
```cpp
// WRONG: TCOLSUM(sum_1x1, g_row_1xK);  // sum_1x1 is UbND<T,1,1> — INVALID

// CORRECT: Use TROWSUM with proper shapes
UbND<float, 1, 128, 1, DYNAMIC> g_row(k);      // Source: 1×K
UbND<float, 1, 8, 1, DYNAMIC> sum_tile(1);     // Dest: 1×8 (aligned)
UbND<float, 1, 8, 1, DYNAMIC> temp_tile(1);    // Workspace: 1×8
TASSIGN(g_row, G_ROW_ADDR);
TASSIGN(sum_tile, SUM_ADDR);
TASSIGN(temp_tile, TEMP_ADDR);
TROWSUM(sum_tile, g_row, temp_tile);           // Reduces 1×K → 1×1 (stored in element 0)
float scalar_result = sum_tile.GetValue(0);
```

**Vec reduction lane limit (wide tiles).** A single multi-row `TROWSUM`/`TCOLSUM`
over a tile whose row is wider than the 64-fp32 Vec lane block reduces ONLY the
first 64 lanes -- the tail is silently dropped, giving a partial sum with no
error. For any feature width above 64 this is a silent-wrong reduction. Reduce
either row-by-row (`1 x width -> 1 x 8` per row) or tile the reduction into
64-lane blocks and sum the partials. (Observed: a multi-row reduction over a
128-wide tile truncated to its first 64 lanes; the per-row form was correct.)
For the correct wide matvec/GEMV pattern -- reduce the wide axis with `TROWSUM`
(per-row output, NOT `TCOLSUM` whose per-column output truncates at 64), and since
`TROWSUM` ALSO reduces only the first 64 lanes of a >64-wide row on this build
(dav-c220/CANN 9.1.0; verified on-NPU), split the reduced axis into <=64-lane blocks
and `TADD` the partials, plus the paired rank-1 update in one orientation -- see
COOK-§10.5.

Always verify reduction instruction signatures in the PTO ISA reference. → PLAT-§Reductions

**C16. TCMP requires matching dtypes between dst and src.**
`TCMP(dst, src0, src1, mode)` compares `src0` and `src1` elementwise and writes
the boolean result to `dst`. All three tiles MUST use the same element type.
Using `Tile<int32_t>` for `dst` while `src0`/`src1` are `Tile<float>` will fail
compilation with strict type checking (e.g., `-g` on bisheng).

```cpp
// WRONG: dst is int32_t but src tiles are float
using MaskRow = Tile<TileType::Vec, int32_t, 1, MAX_BT, ...>;
MaskRow mask_tile(1, bt32);
TCMP(mask_tile, float_tile_a, float_tile_b, CmpMode::LT);  // TYPE ERROR

// CORRECT: all tiles use the same element type
using MaskRow = Tile<TileType::Vec, float, 1, MAX_BT, ...>;
MaskRow mask_tile(1, bt32);
TCMP(mask_tile, float_tile_a, float_tile_b, CmpMode::LT);
```

The comparison result stored in `dst` will be 0.0f (false) or ~0.0f/all-ones (true)
when using float type — `TSEL` interprets these correctly as false/true masks. → PLAT-§Types

**C17. Guard `get_block_num()` against zero return.**
`get_block_num()` returns the total number of compute blocks. In simulation or
edge-case hardware configurations it may return 0. Using 0 as a loop stride
(`wi += block_num`) produces an infinite loop that hangs the kernel.

```cpp
int64_t block_num = static_cast<int64_t>(get_block_num());
if (block_num <= 0) block_num = 1;  // REQUIRED guard
```

Add this guard immediately after calling `get_block_num()`. → PLAT-§GridStride

**C18. Work-item loop bound must match memory addressing stride.**
The ABI passes `total_work` — the total number of **output elements** across all
dimensions. If your kernel processes multiple rows per work item (e.g., each
iteration handles `BT` rows), the loop bound must be `total_work / rows_per_item`,
NOT raw `total_work`. Using raw `total_work` with a per-iteration stride of
`rows_per_item * K` advances the memory pointer far past tensor bounds (DDR fault).

General rule: `loop_bound = total_work / elements_per_iteration`, and the
memory offset per iteration MUST be proportional to `elements_per_iteration`.

Example for a kernel where each iteration processes `rows_per_group` rows of
`cols_per_row` elements:
```cpp
// total_work = total number of output elements (from ABI)
// rows_per_group, cols_per_row = derived from kernel data layout
int64_t num_groups = (rows_per_group > 0) ? total_work / rows_per_group : 0;
for (int64_t gi = get_block_idx(); gi < num_groups; gi += block_num) {
    __gm__ float* g_base = g_prefix + gi * rows_per_group * cols_per_row;
    ...
}
```

The `elements_per_iteration` value MUST be derived from the kernel's data layout
and work distribution — it is NOT always `BT`. If in doubt, trace the memory
addressing: multiply the maximum loop index by the per-iteration byte stride
and verify it stays within the tensor allocation. → PLAT-§GridStride

**Confirm what `total_work` actually counts against the validation harness — it is
not always "elements."** A real harness was observed to pass `total_work = num_mat
* chunk_size` (ROWS), not `num_mat * chunk_size^2` (elements). A kernel that divided
by `chunk_size^2` then computed `num_mat = 0` for every test and silently produced
UNINITIALIZED output (R^2 ~ 0, uniform garbage that looks like a compute bug). The
loop count and the `call_kernel_wrapper` that feeds `total_work` MUST agree: derive
`num_mat` from the SAME quantity the harness passes (here `total_work / chunk_size`),
and sanity-check that `num_mat > 0` for the smallest test case before blaming the
math. When the stage processes one fixed-size matrix per work item, prefer deriving
the count from the explicit dim arg (`chunk_size`) over re-deriving it from a
`total_work` whose unit is ambiguous.

**C19. Never call GetValue after Vec PTO ops — data is stale.**
Vec pipeline operations (`TMUL`, `TADD`, `TMULS`, `TSUB`, `TEXP`, `TSEL`, `TROWSUM`,
`TCOLSUM`) write results to pipeline registers, NOT to the tile buffer. `GetValue()`
reads from the tile buffer, returning the PRE-OP value. `pipe_barrier(PIPE_V)`
synchronizes execution order but does NOT flush registers to the buffer.

```cpp
// WRONG: GetValue after TMUL returns pre-TMUL data
TMUL(result_row, result_row, beta_exp);
pipe_barrier(PIPE_V);
float val = result_row.GetValue(0);  // STALE — reads pre-TMUL value!

// CORRECT: Read before the Vec op
float val = result_row.GetValue(0);  // read now
TMUL(result_row, result_row, beta_exp);
pipe_barrier(PIPE_V);
// Use `val` (pre-TMUL) with formula to derive post-TMUL result
```

This affects ALL Vec ops including TEXP. The only reliable way to read post-op
values is the TSTORE→GM→TLOAD round-trip:

```cpp
// C19 round-trip: store exp(gate) to GM, reload — GetValue is then safe
TEXP(gate_tile, gate_tile); pipe_barrier(PIPE_V);
GF wsg(reinterpret_cast<__gm__ float*>(workspace), shape);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(wsg, gate_tile); pipe_barrier(PIPE_ALL);
TLOAD(gate_tile, wsg);
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
// gate_tile now holds post-TEXP data loaded from GM — GetValue returns correct values
float exp_val = gate_tile.GetValue(j);
```

Do NOT use scalar `expf()` / `logf()` / `__builtin_expf()` — CCE mode prohibits
[host] functions in [aicore] device code (see C23). → PLAT-§Pipeline

**C20. Vec pipeline state leaks between loop iterations.**
When a tile is reused across loop iterations (same UB address, same TASSIGN),
the Vec pipeline registers from iteration N may not be fully retired when
iteration N+1 starts. Always add `pipe_barrier(PIPE_V)` at the START of the
loop body to flush stale pipeline state:

```cpp
for (int i = 0; i < n; ++i) {
    pipe_barrier(PIPE_V);  // flush previous iteration's pipeline
    // ... tile operations on reused tiles ...
}
```

Without this barrier, operations on the first iteration may produce zero output
and subsequent iterations may accumulate phantom scaling factors. → PLAT-§Pipeline

**C21. Scalar writes to GM must use separate TSTORE.**
When a per-element scalar value (e.g., diagonal correction) needs to be written
to GM, do NOT use `GetValue`/`SetValue` on the result tile — use a SEPARATE
`TSTORE` with a dedicated temporary tile:

```cpp
// WRONG: SetValue on result tile after TMUL (value won't persist)
TMUL(result_row, result_row, beta_exp);
pipe_barrier(PIPE_V);
result_row.SetValue(i, corrected_val);
TSTORE(a_gm, result_row);  // writes stale pipeline data, not SetValue

// CORRECT: Separate TSTORE for scalar write
float correction = /* compute algebraically from pre-TMUL values */;
TEXPANDS(temp_tile, correction);  // fixed-size 1x8 tile
pipe_barrier(PIPE_V);
Shape<1,1,1,1,1> ds; GlobalTensor<float, ...> dg(dst_ptr, ds);
TSTORE(dg, temp_tile);  // writes directly to GM, independent of pipeline
pipe_barrier(PIPE_ALL);
```

This bypasses the pipeline register / tile buffer disconnect entirely. → PLAT-§Pipeline

**C22. msprof op simulator validation.**
Kernels can be validated without NPU hardware using the Ascend simulator:
```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH"
msprof op simulator --output=<dir> --aic-metrics=PipeUtilization \
    --launch-count=1 --soc-version=Ascend910B1 \
    python validation_script.py kernel.so
```
Requirements: output dir must be non-world-writable (`chmod 700`), validation
script must NOT call `torch.npu.synchronize()` (hangs), and must compare on-device
without `.cpu()` copies. → PLAT-§Simulator

**C23. CCE mode prohibits [host] functions in [aicore] device code.**
Under `-xcce` compilation, functions declared in system headers (`<cmath>`,
`<math.h>`) are marked `[host]` and CANNOT be called from `[aicore]` functions.
This means `expf()`, `logf()`, `sqrtf()`, `powf()`, `sinf()`, `cosf()`, and all
other scalar math functions from `<cmath>`/`<math.h>` are unavailable.

```cpp
// WRONG: expf is a [host] function, not callable from [aicore]
// error: call to [host] function from [aicore] function
float val = expf(gate_pre[j]);

// CORRECT: use PTO tile ops (TEXP, TRSQRT, etc.) on tiles, then
// C19 round-trip (TSTORE→TLOAD) to read individual values
TEXP(gate_tile, gate_tile); pipe_barrier(PIPE_V);
// ... TSTORE→GM→TLOAD round-trip (see C19) ...
float exp_val = gate_tile.GetValue(j);  // safe after round-trip
```

This is a HARD compile error under `-xcce`. Do not work around it with
`__builtin_expf()` or inline assembly — use PTO tile ops. → PLAT-§CCE

**C24. Correct compile recipe: CCE mode, gnu++17, bisheng from the active CANN toolkit.**
The kernel MUST compile with this exact recipe. Resolve all toolkit paths from
`$ASCEND_HOME_PATH` (set by `set_env.sh`) — do NOT hardcode a CANN version path:
```bash
source /usr/local/Ascend/cann/set_env.sh   # -> ASCEND_HOME_PATH (default: cann-9.0.0)
"$ASCEND_HOME_PATH/bin/bisheng" -fPIC -shared -xcce -DMEMORY_BASE -O2 \
  -std=gnu++17 --cce-aicore-arch=dav-c220 \
  -Wno-macro-redefined -Wno-ignored-attributes \
  -I<kernel_dir> -I<example>/include \
  -I<pto_isa_root> -I<pto_isa_root>/include \
  -I"$ASCEND_HOME_PATH/include" \
  -I"$ASCEND_HOME_PATH/pkg_inc" \
  -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
  -I"$ASCEND_HOME_PATH/pkg_inc/profiling" \
  kernel.cpp -o kernel.so
```

Key constraints:
- Use the active CANN toolkit (default 9.0.0 via the `/usr/local/Ascend/cann` symlink); resolve `bisheng` and includes from `$ASCEND_HOME_PATH`, never a hardcoded version path
- `-xcce` (CCE language mode, NOT `-x cce` with space)
- `--cce-aicore-arch=dav-c220` — auto-defines `__CCE_AICORE__`, `__DAV_C220_VEC__`
- `-std=gnu++17` — C++17 with GNU extensions (NOT c++20, NOT c++17)
- Do NOT add `-D__CPU_SIM` — CCE provides its own device runtime
- Do NOT add `-nostdinc++` — CCE headers are self-contained
- `<<<...>>>` kernel launch syntax is a CCE compiler extension in `-xcce` mode
- No GCC STL is used; CCE provides its own device-side runtime → BUILD-§

**C25. CPU_SIM masks CCE pipeline bugs — always validate under -xcce.**
Under `-D__CPU_SIM`, `set_flag`, `wait_flag`, and `pipe_barrier` are EMPTY
macros. Kernels tested only in CPU_SIM mode may appear correct but produce
garbage when compiled with `-xcce` (real CCE mode). Missing sync flags
before TSTORE, incorrect flag ordering, and pipeline leaks are invisible
in CPU_SIM.

```cpp
// In CPU_SIM: set_flag/wait_flag are no-ops — this compiles and "works"
TSTORE(out_gm, result_tile); pipe_barrier(PIPE_ALL);

// In CCE mode: MTE3 engine not synced — TSTORE may write garbage or 0!
// CORRECT:
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(out_gm, result_tile); pipe_barrier(PIPE_ALL);
```

**Every kernel MUST be validated with `-xcce --cce-aicore-arch=dav-c220`
before being considered correct.** Validation under CPU_SIM alone is
insufficient. → BUILD-§, PLAT-§Pipeline

**C26. TMATMUL operands must go through L1 Mat + TEXTRACT — not TMOV from Vec.**
On A2/A3 (`dav-c220`), `TMATMUL` supports `(float, float, float)` directly —
TCVT to fp16 is NOT required. However, TMATMUL operands (Left, Right) CANNOT
be populated via `TMOV` from Vec tiles. The correct data path is:

1. Load fp32 data from GM into an L1 Mat tile via `TLOAD` (using a Mat-typed `GlobalTensor`)
2. `TEXTRACT` from L1 Mat into L0 Left/Right tiles
3. Run `TMATMUL` on the L0 tiles

```cpp
// CORRECT: GM -> L1 Mat -> TEXTRACT -> L0 -> TMATMUL
using L1Mat = pto::Tile<pto::TileType::Mat, float, M, N, pto::BLayout::RowMajor, ...>;
using L0Left = pto::Tile<pto::TileType::Mat, float, M, K, pto::BLayout::RowMajor, ...>;
using L0Right = pto::Tile<pto::TileType::Mat, float, K, N, pto::BLayout::ColMajor, ...>;

// Load A from GM into L1 Mat
pto::Shape<1,1,1,M,K> shape_a; pto::GlobalTensor<float, ...> gm_a(a_ptr, shape_a);
TLOAD(l1_a, gm_a);
pipe_barrier(PIPE_ALL);

// Extract to L0 operands
TEXTRACT(l0_left, l1_a);   // Left operand from L1 Mat
pipe_barrier(PIPE_CUBE);

// TMATMUL on fp32 directly
TMATMUL(l0_acc, l0_left, l0_right);
pipe_barrier(PIPE_CUBE);
```

**TEXTRACT constraints**:
- Source L1 Mat row dimension (M) MUST be aligned to 16. If M=1 (vector-matrix
  product), pad to M_PAD=16 and use only row 0 of the result.
- `TMOV` from Vec tiles to Left/Right tiles does NOT work on A2/A3.

fp32 inputs are NEVER a valid reason to skip TMATMUL or fall back to scalar loops.
→ PLAT-§Cube, COOK-§8.5, §8.7

**C27. TMUL reads BOTH sources from the pipeline — push TLOAD'd data first.**
`TMUL` (elementwise multiply) reads both source operands from the Vec pipeline,
NOT the tile buffer. `TLOAD` writes to the buffer only. `TMULS` (scalar multiply)
reads from the buffer and writes to the pipeline. Therefore:

```cpp
// WRONG: TLOAD→TMUL — k_tile in buffer, TMUL reads pipeline garbage
TLOAD(k_tile, ...); TEXP(gate_tile, gate_tile);
TMUL(k_tile, k_tile, gate_tile);  // FAILS!

// CORRECT: push k_tile to pipeline with TMULS(*1.0) before TMUL
TLOAD(k_tile, ...);
TMULS(k_tile, k_tile, 1.0f); pipe_barrier(PIPE_V);  // push buffer→pipeline
TEXP(gate_tile, gate_tile); pipe_barrier(PIPE_V);    // TEXP: pipeline→pipeline
TMUL(k_tile, k_tile, gate_tile);  // OK: both in pipeline
```

This applies to ALL Vec ops that read from the pipeline: `TEXP`, `TADD`, `TSUB`,
`TMUL`, `TSEL`, `TROWSUM`, `TCOLSUM`. If an operand came from `TLOAD`, push it
with `TMULS(x, x, 1.0f)` first. → PLAT-§Pipeline

**C28. Vec-contraction tile-shape traps (matvec / outer-product / recurrent state).**
Three layout traps surface whenever a stage does matvecs and rank-1 outer
products on the Vec core (the GEMV / outer-product path of the decision tree and
S3, and recurrent state updates per S9):

(a) **GM-load extent must equal the RUNTIME dimension, not the compile-time CAP.**
The `Shape` / `GlobalTensor` extent passed to `TLOAD` must be the real runtime
size (e.g. the actual K), never a padded compile-time cap (e.g. `CAP=32`). A
`TLOAD` over a CAP-sized extent reads `CAP - K` floats of adjacent tail garbage
past each element's data; that garbage then enters every reduction
(`TCOLSUM`/`TROWSUM`) and, in a recurrent stage, is baked into the carried state
and recirculated -- producing an error that is exact for the first position(s)
and COMPOUNDS over the scan. Caps size UB budgets only; the load/store extent is
the runtime dim. (Load-side complement of C11.)

(b) **Single-column tiles are rejected -- keep vectors >= 8 wide.**
A RowMajor `[N,1]` tile is rejected by the Tile library (no `data()` /
`GetValidRow`); a ColMajor `[N,1]` tile is rejected by `TADD`/`TSTORE`
(RowMajor-only). You therefore cannot carry a vector as a literal column. To
feed `TROWEXPAND` (which reads column 0) a per-row scalar, TTRANS a
`[PAD>=8, N]` tile (row 0 holds the vector) into a `[N, 8]` tile (column 0 holds
the vector); the 8-wide shape is accepted as both TTRANS dst and TROWEXPAND src.
Keep reduction outputs as `[1, N]` (N >= 8) rows, not `[N, 1]`. (Generalizes
C14's minimum-size rule to the single-column case.)

(c) **TROWEXPANDMUL needs a ColMajor `[N,1]` per-row scalar.**
`TROWEXPANDMUL(dst, src0, src1)` computes `dst[i,j] = src0[i,j] * src1[i,0]` and
requires `src1` as a ColMajor `[N,1]` tile. Passing a RowMajor `[N,8]` `src1`
selects device Mode 2 (32B/row) and multiplies element-wise within the block,
ZEROING the result outside column 0 -- a silent-wrong rank-1 update. When the
per-row-scalar layout is not provably ColMajor `[N,1]`, build the outer product
as `TROWEXPAND` (broadcast the scalar) + `TMUL` instead; it is layout-robust.
→ PLAT-§Pipeline, PLAT-§Alignment

**C29. Never name a constant with a bare short token -- it can collide with a PTO
library symbol and SILENTLY mis-size tiles.** `using namespace pto;` (via
`kernel_common.h`) pulls in many short identifiers as enum constants / typedefs.
A kernel that wrote `constexpr int BT = 64;` collided with a library `BT` whose
value was 5: the compiler EITHER errors `redefinition of 'BT' as different kind of
symbol`, OR -- worse -- the library symbol wins inside template arguments and every
`Tile<float, BT, BT>` instantiates as 5x5 instead of 64x64, cascading into bogus
`no member 'data'/'GetValidCol'` and alignment static_asserts that hide the real
cause. Give EVERY compile-time constant a distinctive, kernel-private name: a prefix
like `kBT`, `INV_BT`, `CHUNK_BT`, `kTile`. Never use a bare 1-3 letter all-caps token
(`BT`, `K`, `V`, `N`, `M`, `C`, `T`) as a constant/typedef name. → COMPILER, PLAT-§Alignment

**C30. `TTRI` template signature: `TTRI<TileType, isUpperOrLower>(dst, diagonal)`.**
`isUpperOrLower` is a NON-TYPE template arg (`0` = lower/`TTril`, `1` = upper/`TTriu`);
`diagonal` is a runtime int (`0` = include main diagonal, `-1` = strictly below).
`TileType` is the first template param and cannot be skipped -- pass `decltype(dst)`.
Do NOT write `TTRI<float, 0>(dst, 0)` -- that binds `TileData = float`, so `dst` (a
tile) fails to convert. Build an identity matrix as `lower(diag 0) - strictly_lower(diag -1)`:
```cpp
TTRI<decltype(lo), 0>(lo, 0);    // 1s where col <= row
TTRI<decltype(sl), 0>(sl, -1);   // 1s where col <  row
TSUB(eye, lo, sl);               // 1s only on the diagonal
```
→ PLAT-§Types

**C31. `TTRANS` reads its SOURCE from the tile BUFFER, and is reliable only at the
full STATIC tile height.** Two silent-wrong traps when transposing:
(a) `TTRANS` -- like `TLOAD`/`GetValue` -- reads the source tile's BUFFER, not the
Vec pipeline. If the source was produced by a Vec pipeline op (`TMUL`/`TADD`/`TSUB`/
`TMULS`/`TEXP`/...), the fresh value is in the PIPELINE and the buffer is stale, so
the transpose silently yields a WRONG, often NONDETERMINISTIC, result. Commit the
source to the buffer first with a `TSTORE`->`TLOAD` GM round-trip (the C19 pattern)
before `TTRANS`. (This is the C19/C27 buffer-vs-pipeline family applied to TTRANS.)
(b) A transpose is correct at its full STATIC declared tile height; feeding a
runtime-SHORT row count into a transpose can produce a wrong transpose. For a
runtime-variable matrix size whose path includes a transpose that feeds a Cube
operand, prefer the S3 "template the device function on the size" approach (each
instantiation transposes a full static tile) over DYNAMIC / runtime-short transpose
tiles. If you must pad, ZERO-pad to the full static tile explicitly before the
transpose. → PLAT-§Pipeline, PLAT-§Cube

**C32. Numerically-sensitive elementwise math MUST run in fp32 (load fp16 -> TCVT
fp32 -> compute -> TCVT fp16 -> store).** fp16 has ~11 mantissa bits (~3e-4 relative
quantization). For a stage dominated by gates / exponentials / decays / reductions /
cumulative sums / normalizations -- especially a LOOP-CARRIED scan whose per-chunk
fp16 requantization ACCUMULATES over the sequence -- doing the elementwise math in
fp16 fails a strict Frobenius gate (ftol ~2e-3 against an fp64 reference; the loose
elementwise rtol ~2e-2 that fp16 passes MASKS this deficit). The reference technique
(and the correct generated pattern) is: `TLOAD` the fp16 operand, `TCVT(fp32_tile,
fp16_tile, pto::RoundMode::CAST_NONE)` to widen, do ALL the sensitive elementwise ops
in fp32 -- the gate decays `exp(g_cs[r]-g_cs[c])`, `exp(+/-g_cs)`, `exp(g_total)`,
cumulative gate sums, beta scaling, the L-matrix build, the `u - w@S` correction, the
masked `Aqk`, and any carried-state recurrence `S = decay*S + kv` -- then
`TCVT(fp16_tile, fp32_tile, pto::RoundMode::CAST_NONE)` back to fp16 ONLY at GM stores.
The matmul ACCUMULATOR is already fp32 via `TileAccF<float>` -- that part needs no change.
For a loop-carried scan, ALSO keep the carried state RESIDENT in fp32 in its GM workspace
(do not store it fp16 between chunks) -- the per-chunk fp16 store is a compounding source.

**Carried recurrence state must reach its CONSUMING matmul in fp32 -- no fp16 shadow of
scan state.** When a loop-carried state `S` (fp32-resident) is also an OPERAND to a Cube
matmul inside the scan (e.g. `wS = w @ S`), do NOT keep a separate fp16 `S` shadow to feed
the matmul: feed the carried fp32 `S` DIRECTLY via the fp32 `(float,float,float)` A2/A3
TMATMUL path (C26 -- widen the other fp16 operand to fp32 in GM via a small Vec pre-step,
then `TLOAD` both as fp32 L1 Mat tiles). A full `[128,128]` fp32 L0 operand is 64 KB and
fills L0A/L0B exactly; if both operands plus margin do not fit, K-split the contraction
(two `AccPhase::Partial`/`Final` phases, each L0 operand `[bt,64]`/`[64,V]` = 32 KB). An
fp16 shadow of `S` diverges from the carried fp32 `S` and breaks recurrence consistency.

**Build per-row decay / per-element broadcast factors with a RELIABLE transpose, never a
height-1 TTRANS (C31).** A loop-carried scan's decay step `S = exp(g_total[k]) * S` needs a
per-K-row scalar broadcast across V: `decay[k,v] = exp(g_total[k])`. The robust build is the
C28(b) form -- a STATIC `[16,K]` source tile (the `[1,K]` vector loaded into row 0, the tile
zero-filled) `TTRANS`'d into `[K,16]` (col 0 = the vector), then `TROWEXPAND` (reads col 0)
across V. Transposing a height-1 `[1,K]` source tile directly is the UNRELIABLE TTRANS case
(C31) and SILENTLY corrupts ~half the rows of the broadcast. This is INVISIBLE while the
scaled operand is zero (the first chunk, where the carried `S` is still 0) and only surfaces
from the SECOND non-zero-state chunk onward -- where a wrong decay injects a large bogus
`decay*S` term (its norm can be ~9x the correct value even though sampled elements look
right) and breaks the recurrence. A short scan (e.g. T=256 = 2 chunks) can PASS because its
only non-zero-state update is never re-consumed/snapshotted; a longer scan (T>=512) FAILS.
When a per-chunk-precision change leaves a longer-sequence error byte-IDENTICAL, the bug is
structural (a broadcast/transpose layout fault), not precision -- localize per-chunk and per
S-update term (decay vs kv) rather than chasing operand dtypes.

**UB-budget implication (the practical cost):** an fp32 tile is 2x the bytes of the
fp16 tile (a 128x128 fp32 tile = 64 KB; the fp16 = 32 KB). The 192 KB UB (A2/A3) holds
at most THREE live 128x128 fp32 tiles. So budget fp32 working slots explicitly
(re-derive the UB map per C7), TILE/STAGE the computation if the fp32 footprint does
not fit (process sub-tiles, alias dead fp32 slots, stage fp16<->fp32 through one shared
slot), and keep a small fp16 staging slot for the load/store boundary. Update the
`static_assert(kMaxUbAddr <= 196608)` for the fp32 layout.

**Buffer/pipeline discipline with TCVT (C19/C27/C31 apply):** `TCVT` is a Vec op -- it
reads its source from the tile BUFFER and writes its result to the PIPELINE. So after a
`TCVT`, the widened/narrowed value is in the pipeline; a following op that reads it from
the BUFFER (`TTRANS`, `TLOAD`-then-`GetValue`, or a TROWEXPAND/TMUL src that came only
from `TCVT`) sees STALE data. Commit with the C19 GM round-trip (or a `TMULS(x,x,1.0f)`
push for an immediate pipeline consumer) before re-reading. Do NOT cross an intervening
`TLOAD` between producing a pipeline value and consuming it in a `TADD`/`TMUL` -- the
intervening op can clobber the pipeline slot; load the other operand FIRST, then produce
and consume back-to-back. → PLAT-§Pipeline, COOK-§8.11

### 🟡 STANDARD Rules (S-series)

Violations produce wrong results, validator rejections, or degraded performance.

**S1. PTO-op-centric compute.**
Keep compute loops tile-based: TLOAD/TSTORE/TMATMUL/TADD/TMUL/TEXP.
No scalar fallback bodies (`out[i] = ...`) as the main path.
Scalar `GetValue`/`SetValue` loops are allowed only for narrow approved tasks
(head-lane extraction, small UB-resident prefix accumulation). They must not
implement dominant BTxK/BTxV/BTxB math or walk GM-backed outputs directly. → COOK-§6, §7

**S2. No scalar math — PTO tile ops only.**
Do NOT use `expf`, `logf`, `sqrtf`, `powf`, `sinf`, `cosf`, `std::exp`,
`__builtin_expf`, or any other scalar math function from `<cmath>`/`<math.h>`.
These are `[host]` functions unavailable in CCE `[aicore]` device code (C23).

Use PTO tile ops (`TEXP`, `TRSQRT`, etc.) on tiles. To extract individual
post-op values, use the C19 round-trip pattern (TSTORE→GM→TLOAD). → COOK-§6, §13

**S3. Contractions require Cube.**
When StageSpec math contains `@`, matmul, einsum contractions, or
`torch.matmul`, lower them on a Cube-core contraction path (`TMATMUL` or
another proven dense Cube surface). Scalar `for`-loop matrix multiplications
are forbidden for dominant math. Do not replace required contractions with
scalar dot/row helper loops, Vec copy-through fallback paths, or
`Lowering gap` comments. If runtime-symbolic dimensions are the only
uncertainty, keep the contraction and make the runtime bound explicit.

**Data path to TMATMUL**: operands must go through GM -> L1 Mat (TLOAD) ->
TEXTRACT -> L0 Left/Right -> TMATMUL. Never use TMOV from Vec tiles to
Left/Right operands (C26). fp32 inputs are supported directly by TMATMUL
on A2/A3 -- no TCVT to fp16 needed. The only valid reasons to defer TMATMUL are:
- The contraction dimensions are genuinely unknown (no shape info at all)
- The operands cannot be tiled into L1 due to UB budget (document in evidence gaps)
- The contraction is a matrix-VECTOR product (M=1 GEMV), a rank-1 OUTER product
  (contraction dim=1), or a small (M, N, or contraction dim <= ~16) contraction
  carried inside a sequential / loop-carried scan. These do NOT fill a Cube
  fractal (>=16x16) and pay the full L1->L0 + TEXTRACT(M padded to 16) + per-issue
  FFTS cross-core cost, which a small vector op cannot amortize; the
  per-iteration Vec<->Cube handshake inside a scan is also a deadlock/correctness
  hazard (C6). Lower them on the Vec core with TROWEXPAND/TCOLEXPAND + TMUL +
  TCOLSUM/TROWSUM. This is tile-based Vec contraction, NOT a forbidden scalar
  `for`-loop. Rule of thumb: dense matrix-MATRIX with all of M, N, contraction
  >= 16 (realistically >= 64) and batchable into one issue -> Cube; matrix-VECTOR,
  rank-1 outer product, or a tiny contraction stepped one vector at a time in a
  scan -> Vec.
- A **triangular solve / matrix inverse** (e.g. `(I +/- strict_lower(M))^-1`) is
  sequential along the solved axis, but HOW to lower it depends on SIZE:
  - **Small** (solved axis <= ~16, or a tiny block stepped inside a larger scan):
    keep it a **row-sequential Vec** forward substitution
    (TROWEXPAND/TMUL/TCOLSUM). A per-iteration Cube handshake would not amortize.
  - **Large** (solved axis >= ~32, a dense unit-triangular block that fits a Cube
    tile): **BLOCK it** so the inverse is dominated by dense Cube matmuls -- do
    NOT leave a large solve as a row-sequential Vec loop (that was a real
    dominant-stage bottleneck). For the strictly-lower / nilpotent case (small
    ||N||, e.g. the L2-normalized regime) Neumann doubling is exact and simplest:
    with `P = -strict_lower`, `inv = product_s (I + P^(2^s))` via
    `X <- X + X@P; P <- P@P`, converging in `ceil(log2(BT))` steps -- ~log2(BT)
    dense `TMATMUL`s instead of BT sequential rows (BT=64 -> 6 steps; ~10x faster
    on dav-c220, validated). **Perf caveat (measured):** "fastest" here is relative
    to the row-sequential Vec loop, NOT to a blocked recursion. Neumann doubling
    issues ~2*log2(BT) FULL-WIDTH BTxBTxBT matmuls, which is materially more work
    than a block-recursive forward substitution; a generated Neumann inverse on a
    full 128-block measured ~9-13x SLOWER than a hand-tuned blocked tri-inverse and
    became the pipeline's single dominant stage. So Neumann is the correctness-first
    / simplest choice; when inverse latency matters at the largest block size,
    prefer the blocked TRTRI recursion below. The general blocked form (LAPACK TRTRI) also works:
    off-diagonal updates `inv[i,j] = -inv(L_ii) @ (sum_{j<=k<i} L_ik @ inv[k,j])`
    are Cube matmuls (dependency O(BT/blk) blocks). See COOK-§8.13.

    **Worked instance -- the KDA chunk inverse `(I + strict_lower(L))^-1`.** In KDA
    `L` is the strictly-lower gated `K.K^T` matrix of one `[BT, BT]` chunk, so
    `M = I + strict_lower(L)` is UNIT lower-triangular: 1s on the diagonal and a
    nilpotent strictly-lower part. Two equivalent dense-Cube routes, both
    `~log2(BT)` deep (not `BT` sequential rows):
      - **Neumann doubling** (above), exact precisely because `strict_lower(L)` is
        nilpotent: with `P = -strict_lower(L)`, `inv = prod_s (I + P^(2^s))` via
        `X <- X + X@P; P <- P@P`.
      - **Recursive 2x2 block inverse:** split `M = [[M11, 0], [M21, M22]]`,
        recurse on the two diagonal blocks (each itself unit lower-triangular),
        then fill the off-diagonal block as `-M22^-1 @ M21 @ M11^-1` (two dense
        `TMATMUL`s). Every recursion level is dense Cube matmuls on `>=16`-wide
        blocks -- no row-sequential Vec loop.
    Pick ONE unit-triangular-inverse primitive and SHARE it across every stage (and
    every algorithm) that needs a triangular solve, rather than reimplementing it
    per stage. When MANY independent chunk matrices are inverted in one launch,
    choose the per-call unroll/batch factor from `num_matrices / block_num`
    (1 matrix/core, 2/core, 4/core, ...) so the whole core grid stays busy.

    **Runtime-variable matrix size on the Cube path: template + dispatch, don't go
    DYNAMIC.** Cube `TMATMUL`/`TEXTRACT` fractal tiles must be statically sized, and
    there is no proven dynamic-extent Cube GEMM surface on a2a3 (the DYNAMIC valid-extent
    idiom is Vec-only). So when the same stage must handle several runtime tile sizes
    (e.g. BT in {16,32,48,64}, all multiples of 16), DO NOT try to make one kernel with
    runtime tile dims. Instead template the device function on a compile-time size,
    `template <int BT_C> stage_kernel(...)`, derive size-dependent constants from `BT_C`
    (tile dims, `NSTEPS = ceil(log2(BT_C))`, GM strides `mi * BT_C * BT_C`), and have the
    `extern "C" call_kernel` host wrapper `switch (chunk_size)` to the matching
    `launch_*<BT_C><<<...>>>` instantiation. Each instantiation is the proven fixed-size
    path; the cost is N copies of the kernel (validated: 4 instantiations -> one ~52 KB
    `.so`, all sizes exact). Note: a template cannot have C linkage, so only the
    dispatching `call_kernel` is `extern "C"`; the `launch_*<BT_C>` template is plain C++
    linkage. Compute size-dependent loop counts with a constexpr expression, NOT a
    `[host]` constexpr helper function called from `[aicore]` code (C23 forbids it --
    it surfaces as a confusing "no matching function" error). Worked replacement --
    you are dispatching on a KNOWN set of sizes, so key a constexpr ternary on the
    template size:
    ```cpp
    template <int N>
    AICORE void stage_kernel(...) {
      constexpr int NSTEPS = (N <= 16) ? 4 : (N <= 32) ? 5 : 6;  // ceil(log2 N)
      ...
    }
    ```
    For an unbounded N use a recursive template-constant struct instead of a free
    function -- `template<int N> struct CeilLog2 { static constexpr int value =
    1 + CeilLog2<(N + 1) / 2>::value; };` with a `CeilLog2<1>` (value 0) base case.
    Never a plain `constexpr int f(int)` free function: it is `[host]` and the
    `[aicore]` body cannot call it.
  - **ISA caveat (A2A3/dav-c220):** `TEXTRACT` into a `TileAcc` is A5-only, so a
    `TMATMUL_ACC` C0 cannot be streamed from GM here -- add the unit-diagonal
    "I +" term with a Vec `TADD`, keep the products on Cube, round-trip operands
    through GM (C19) between matmuls, and stream-serialize the steps (no per-step
    Vec<->Cube handshake -> no C6 deadlock).

**Layout-robust Vec micro-GEMM** -- a Vec recipe for the matrix-VECTOR / rank-1 /
tiny-contraction cases of the decision tree, or a genuine LAST resort. Do NOT use it
for a dense matrix-MATRIX contraction (all of M, N, K >= 16) just to avoid the Cube
handshake -- that is a validated order-of-magnitude regression (C6); use Cube there
(split-launch needs no handshake at all). "Correctness-first" does not justify Vec
for a dense GEMM. When it IS the right call (Cube genuinely unavailable, or a
small/GEMV contraction), the recipe:
for each output row, `TLOAD` the row -> `TTRANS` to a ColMajor `[N,1]` column ->
`TROWEXPAND` broadcast -> `TMUL` with the source tile -> `TCOLSUM` reduce over
the contraction axis. Numerically exact (~1e-8) for a dense row-major GEMM at
moderate tile sizes. It avoids `GetValue` on `TLOAD`'d tiles (which returned
garbage on this HW) by keeping everything in tile ops.
→ COOK-§8.7, §8.8

**S4. No hard-coded logical dimensions.**
Do not freeze `BT`, `K`, `V`, `H`, `HV`, `NT`, batch size, or head counts as
guessed `#define` or `constexpr` values unless the StageSpec or chosen
cookbook pattern explicitly fixes them. Fixed compile-time constants may
describe helper tile widths, workspace block sizes, or validated inner tiles.
If you need a compile-time helper, make it obviously a helper block
(`COL_BLOCK`, tile width, ping-pong slot), not a silent replacement.

**S5. Semantically specified stages are not skeletons.**
If `reference_source` and concrete outputs are present, the stage is
semantically specified. Do not emit `skeleton_only`, no-op, zero-fill,
copy-through, or output-preserving placeholder kernels. Use the closest
real cookbook pattern first and leave only genuinely missing sub-lowerings
as evidence gaps. → COOK-§17

**S6. Stage-trait semantic preservation.**
When `stage_family`, `stage_subfamily`, or `stage_traits` are present, treat them
as the primary semantic contract. When absent, infer the contract from
`reference_source` and `instruction_families`. Preserve invariants by trait class:

| Trait class | Key invariants |
|-------------|----------------|
| `prep` / `preprocess` / `cumsum` | Query scaling applied exactly once; inclusive cumsum over BT; lane-correct gather `[rows, heads] → [rows]` |
| `correction` / `transfer_and_projection` | Anchored-difference seed; distinct beta row vs column applications; strict-lower closure before identity handling; true A-projection contractions for `w` and `u` |
| `recurrent` / `chunk_scan` | State update semantics preserved; no `o = u` copy-through; contractions remain tile/Cube based; both output terms and full state update across NT |

For correction stages: keep `BT`/`K`/`V` runtime-symbolic; do not downgrade
to structural-only placeholders. For recurrent stages: preserve both output
terms `(q_i * exp(g_i)) @ S`, `Aqk @ v_i`, and the decay + accumulate
update of `S`. → REV-§Semantic Invariants

**S7. Gather layout preservation.**
For beta/gate gathers (`[BT, HV] → [BT]`), keep the cookbook ND block-load
plus head-lane extraction shape. Do not replace with a guessed contiguous
1D segment load. `TLOAD(VecTile, GlobalTensor)` must preserve layout class
(ND2ND, DN2DN, NZ2NZ). Do not pair ColMajor/DN Vec tiles with ND GlobalTensor. → COOK-§0.5

**S8. Correction-stage specific rules.**
For correction-factor stages:
- Do not collapse two distinct beta applications into a single factor
- Reject factorized seed paths that materialize `exp(g_prefix) * k_chunks`
  workspaces before seeding; use direct anchored-difference lowering instead
- Do not use the output buffer as the recurrence workspace during closure
- Do not implement dominant closure or contractions as GM pointer-walk
  scalar loops; keep main math on PTO tiles or guarded UB-local state → COOK-§10

**S9. Recurrent-stage specific rules.**
For recurrent/scan stages:
- Do not instantiate dynamic `TileType::Mat` objects then `TEXTRACT` unless
  that exact constructor/extract surface is proven in the cookbook
- Do not emit `TEXTRACT` paths with `SLayout::ColMajor` on extracted tiles
- Do not issue `TLOAD` directly into `TileLeft`/`TileRight` destinations
- Do not instantiate 1x1 Vec tiles (`Vec1D<1>`) in recurrent dominant paths
- Do not gate dominant math under `#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)`
  for mixed Vec/Cube bodies; require joint capability or split → COOK-§8.5, §8.7
- State-tensor layout MUST be consistent end-to-end. When the state is updated
  by BOTH a matvec (reduction over one axis) and a rank-1 outer product, pick a
  state orientation so the reduction axis and the outer-product orientation
  agree and the SAME vector orientation flows through produce -> add -> store ->
  consume. For a state `S[reduce_axis, free_axis]` reduced over `reduce_axis`,
  store it as `[reduce_axis, free_axis]` (reduce axis = rows) and reduce with
  TCOLSUM (dst is `[1, free_axis]` RowMajor). Do NOT store it transposed and
  reduce with TROWSUM: that forces every free-axis quantity into column 0,
  disagrees with the outer-product orientation, and yields a per-step error that
  COMPOUNDS over the scan (exact at step 0, growing every step). See C28(a)/(b).
  EXCEPTION (wide free axis): when the free/output axis exceeds 64 lanes (e.g.
  dv=128), `TCOLSUM` silently truncates its output to the first 64 columns. In that
  case use the reduce-over-columns `TROWSUM` orientation (state `[dv, dk]`, reduce
  over dk cols) from COOK-§10.5, which handles widths >64 and keeps the matvec and
  rank-1 update in one consistent orientation.
- Build the rank-1 update from broadcasts, not TROWEXPANDMUL: a row-indexed
  column scalar via TROWEXPAND (reads col 0) times a `[1, free_axis]` row via
  TCOLEXPAND, then TMUL. To place a vector in column 0 without a rejected
  single-column `[N,1]` tile, TTRANS a `[PAD>=8, N]` tile (row 0 = vector) into
  `[N, 8]` and use col 0. See C28(b)/(c).
- **Cross the Cube<->Vec boundary as FEW times as possible per iteration.** A
  loop-carried stage that mixes Cube and Vec work should use EITHER (a) ONE coarse
  Cube-then-Vec boundary per iteration (a few large sub-launches), OR (b) a single
  correctly-signalled in-kernel handshake (C6) -- NOT a fine-grained per-op relay
  (many small Cube/Vec sub-launches per iteration, each round-tripping the carried
  state through GM). The fine-grained relay is both SLOW (a launch + GM round-trip
  per op) and coherency-FRAGILE: every hand-off needs an explicit `dcci` flush
  (COOK-§8.6) and can still show residual nondeterminism. Fold consecutive
  same-engine ops into one launch; minimize boundary crossings.
- **For a runtime-variable tile size, template the WHOLE device function on the size
  (S3) -- not DYNAMIC tiles or runtime-short transposes.** Then every per-iteration
  Cube operand and transpose is a full static tile (avoids the C31(b) short-transpose
  trap). Dispatch on the runtime size from the `extern "C" call_kernel` host wrapper.

**S10. Preferred performance patterns.**
Apply when stage math supports them:
1. Tile-first decomposition, not scalar element decomposition
2. Explicit load-compute-store segmentation
3. Tail-aware handling (static fast path, dynamic tail only where needed)
4. Mixed precision (fp16/bf16 inputs with fp32 accumulation) -- but ONLY a win
   for stages that are genuinely Cube-FLOP-bound. For stages dominated by Vec
   prep/epilogue, GM round-trips, or many stream-serialized launches (i.e.
   memory-/launch-bound), feeding fp16 matmul operands barely helps and can even
   slow a tiny stage (extra TCVT + the cast/launch overhead outweighs the byte
   saving). Validated: fp16-converting a memory-bound matmul chain closed only
   ~3% of its gap to an fp16 reference. PROFILE whether a stage is FLOP- vs
   memory/launch-bound before reaching for fp16; the fp16 bandwidth win is real
   only when intermediates stay resident as half across a matmul chain (halving
   GM traffic), not from the matmul FLOPs alone.
5. Layout-aware transforms (reshape/reinterpret over explicit transpose)
6. Overlap (MTE2/Vec/MTE3, or staged Cube/Vec) when dataflow allows
7. Workspace ping-pong for multi-phase producer/consumer chains

**S11. Bulk tiled data movement -- never row-by-row + PIPE_ALL.**
For any vec_only copy / reorder / gather / scatter / scale stage, move WHOLE TILES,
not rows. Issue ONE strided `TLOAD` per tensor that gathers the entire `[BT, D]`
chunk -- a `GlobalTensor` with `Shape<1,1,1,BT,D>` whose BT-axis (DIM_3) stride
carries the source per-token stride (`Stride<...,DYNAMIC,1>` + runtime ctor when
the stride depends on H/HV) -- into a `[BT,D]` UB tile, do the elementwise op in
ONE instruction, then ONE `TSTORE` to the contiguous destination. NEVER loop over
BT issuing one-row `Shape<1,1,1,1,D>` `TLOAD`/`TSTORE`, and NEVER put
`pipe_barrier(PIPE_ALL)` inside such a loop -- that row-by-row + full-drain idiom
cost ~10x bandwidth on a real stage (45 ms -> 4.6 ms when tiled, validated).

Two correctness gotchas in the strided path:
- **The gm<->ub burst engine only honors a stride on the BT/DIM_3 axis; the inner
  DIM_4 is always contiguous.** A per-token SCALAR gather (e.g. a gate/beta whose
  source stride = head count) MUST put the gather count on DIM_3 with DIM_4=1
  (32-byte-aligned tile width, validCol=1) -- a stride on a `[1,N]` COLUMN axis is
  SILENTLY IGNORED and reads contiguous/wrong data.
- **Give each concurrently-live tensor its own disjoint UB buffer.** A 2-slot
  buffer reused across tensors under narrow flags has a WAR hazard (needs an
  explicit MTE3->MTE2 dependency) -- otherwise it corrupts the other buffers.

Pipeline with narrow `set_flag`/`wait_flag` (MTE2->Vec->MTE3); use exactly ONE
`pipe_barrier(PIPE_ALL)` at the end of the work item (for the loop-carried
dependency), none between tensors/rows. Use BOTH AIV sub-blocks (partition
work-items or BT-rows across vid 0/1 with disjoint UB) -- do not `if (vid != 0)
return;` for a pure data-movement stage. EXEMPTION: sequential in-UB algorithms
(triangular solves, cross-row prefix scans) keep their row loop, but still load/
store the chunk in bulk ONCE at the boundaries and use `PIPE_V` (not `PIPE_ALL`)
for intra-Vec deps inside the loop.

### 🟢 ADVISORY Rules (A-series)

Violations make review harder or the code more brittle.

**A1. Required file banner.**
At the VERY TOP of every generated kernel, BEFORE any `#include` directive, emit:
```
// ============================================================================
// <stage_name>.cpp — <algorithm> stage kernel
//
// Stage role:
//   <1-3 lines: what this stage computes and which outputs it writes>
//
// Architecture / dataflow:
//   <vec_only | cube_only | cube_vec_pipeline | varlen_tail | skeleton_only>
//   <core work partition, workspace usage, chunk loop, cross-core protocol>
//
// Key PTO ops used:
//   <comma-separated ops actually used in this file>
//
// Evidence gaps / conservative choices:
//   <only when needed; otherwise omit>
// ============================================================================

#include "kernel_common.h"
```

The banner MUST appear before the first `#include`. The file structure must be:
1. Banner comment block (lines starting with `//`)
2. Blank line
3. `#include "kernel_common.h"` (the ONLY include)
4. Rest of the kernel code

Do not claim Cube/Vec cooperation unless implemented. Do not mention foreign
kernels, repo paths, or links. If skeleton, say so explicitly. → COOK-§0

**A2. Narrow synchronization.**
Keep `pipe_barrier(PIPE_ALL)` only where truly required (after TLOAD/TSTORE).
Use named `SetFlag`/`WaitFlag` helpers for repeated MTE handoffs. → COOK-§5

**A3. Tail handling.**
Keep static fast path first, dynamic tail path only where needed. Do not
build whole-kernel dynamic machinery when only the tail is dynamic. → COOK-§11

**A4. No repo paths or foreign references.**
Do not mention repo file paths, third-party file paths, or links in
generated comments. Keep the kernel self-describing.

**A5. Prefer fused instructions when available.**
When the stage math supports it, prefer fused PTO instructions over
decomposed sequences to reduce intermediate tiles and data movement:
- `TAXPY` (fused multiply-add) over separate `TMUL` + `TADD`
- `TMATMUL_ACC` (fused accumulate matmul) over `TMATMUL` + `TADD` for K-sliced loops
- `TMATMUL_BIAS` (fused bias matmul) over `TMATMUL` + `TADDS`
- `TROWEXPANDADD/SUB/MUL/DIV` (fused broadcast+op) over `TROWEXPAND` + separate op
- `TADDC/TSUBC` (fused ternary ops) over separate binary operations
- `TROWEXPANDEXPDIF` / `TCOLEXPANDEXPDIF` (fused exp-difference) over `TSUB` + `TEXP`

Verify fused instruction existence and signature with MCP (`get_cpp_intrinsic`)
before using. If MCP is unavailable, fall back to cookbook-proven decomposed
sequences. → COOK-§18

**A6. Multi-stage single-launch fusion (mega-kernel composition).**
When several already-validated stage kernels form a feed-forward (and/or scan)
chain and you want ONE deployable launch, you do NOT need a separate host launch
per stage, and the stages do NOT need to share an on-chip layout. The launch
boundary is not the only global barrier available: an on-device ALL-CORE
cross-core FFTS barrier (a full Cube+Vec sync) placed BETWEEN stage calls gives
the same global ordering inside a single kernel.

**Use the LIBRARY barrier -- do NOT hand-roll it.** The all-core Cube+Vec barrier
already ships as `pto::SYNCALL<pto::SyncCoreType::Mix>()` (public entry
`pto/common/pto_instr.hpp`; a2a3 impl `pto/npu/a2a3/SyncAll.hpp`). `SyncCoreType`
(`pto/common/type.hpp`) is `{AIVOnly=0, AICOnly=1, Mix=2}`; `Mix` is the full
Cube+Vec reduce you want between fused stages. It uses the RESERVED system FFTS
flags 11-14 (`SYNC_AIC_FLAG=11`, `SYNC_AIV_FLAG=12`, `SYNC_AIC_AIV_FLAG=13`,
`SYNC_AIV_ONLY_ALL=14`; `SYNC_FLAG_ID_MAX=16`) and an internal `dcci`+`dsb`, so it
is correct by construction and carries no flag-ID-collision risk with stage-internal
user flags (keep those in 0-10). Hand-rolling an all-core barrier from
`set_cross_flag`/`wait_flag_dev` is the classic source of `aicore exception 507015`
(a non-deterministic cross-core race: passes a few runs, then reads un-committed GM
-> all-zeros, then hard-faults/deadlocks at scale) -- reach for `SYNCALL<Mix>` first
and only hand-roll a finer-grained per-slot handshake when you have measured that
the coarse all-core barrier is the bottleneck.

**Hard constraint: `SYNCALL<Mix>` caps `block_dim` at the physical AIC count.**
It deadlocks when `block_dim` exceeds the available AIC cores (`kCvMaxCores=25`,
`pto/npu/a2a3/custom/TSyncCVID.hpp`; measured: fine at bd<=16, deadlocks at bd=32).
A one-work-item-per-block fused grid must therefore keep `block_dim <= ~24`. Also,
as with any cross-core handoff, a producer's GM stores must be committed on the
STORING pipe (Vec MTE3 / Cube FIX) before the barrier.

Composition pattern (generic, algorithm-agnostic):
- Reuse each stage's device function VERBATIM -- do not regenerate it. `#include`
  each stage translation unit into its OWN namespace, neutralizing each file's
  host `call_kernel` wrapper with a `#define call_kernel <unused>` / `#undef`
  guard so host symbols do not collide. Call each stage's templated device entry
  in sequence. This keeps the standalone and fused builds on one source.
- Between consecutive stages insert exactly ONE all-core barrier that reduces
  over BOTH Vec sub-blocks AND the Cube core -- `pto::SYNCALL<SyncCoreType::Mix>()`
  is that barrier. It uses the reserved system flag band (11-14), DISTINCT from any
  stage-internal user flags (0-10), so a stage's exit-sync and the inter-stage
  barrier cannot reuse the same flag back-to-back and race.
- Put the barrier in the fused ORCHESTRATOR, not inside the per-stage Vec device
  functions. A stage's Vec body may keep its `if (vid != 0) return;` -- vid 1
  returns up to the orchestrator and reaches the next `SYNCALL`, so BOTH vids still
  hit every barrier. (The "branch, don't return" rule of C12 applies to a handshake
  INSIDE a stage; for fusion the barrier lives one level up.)
- A layout/regroup between stages (e.g. permuting an intermediate to a different
  head/token grouping) is just another barrier-separated stage: pre-permute the
  STATIC inputs host-side and do only the dynamic reorder on-device.
- A serial scan stage is NOT a fusion blocker. It is one stage call that loops
  internally; the mega kernel only sequences it after its producer with a barrier.

When it actually pays off: launch-count reduction and a single resident grid help
when stages are individually CHEAP (per-launch setup is a real fraction of their
time), or when fusion lets intermediates stay resident as half across a matmul
chain (S10.4 -- the only regime where fp16 truly wins). When the chain is dominated
by HOST-DISPATCH overhead -- many small launches, each `<<<>>>` a real fraction of
the total -- collapsing N launches into one is a genuine SPEEDUP, not just
packaging (measured: a 6-stage / ~20-sub-launch KDA chain fused to one launch ran
1.6-1.7x faster end-to-end, with identical FLOPs and the same GM round-trips). What
fusion does NOT speed up is a SINGLE stage that is already Cube-FLOP-bound and
round-trips its intermediates through GM regardless: there, fusion is a PACKAGING
win (one `.so`, one launch), not a speedup (cf. the C6 performance caveat). So:
measure where the time goes -- if it's dispatch, fuse for speed; if it's one heavy
matmul, fix engine choice (Cube vs Vec, S3) and precision residency (S10.4) FIRST
and fuse only for deployment. → COOK-§8

---

## Generator Workflow

After completing the pre-generation checklist:

1. **Emit the file banner** (A1)
2. **Emit includes and guards** — follow the skeleton in COOK-§1
3. **Emit device-only type aliases** — under `#ifdef __CCE_AICORE__`
   - If Cube path: include L1 Mat and L0 Left/Right tile types for TEXTRACT path (C26)
4. **Emit compile-time constants and UB address map** — with `static_assert` (C7)
   - If Cube path: allocate UB addresses for L1 Mat staging and L0 tiles (C26)
5. **Emit the device compute function** — `AICORE void stage_kernel(...)`:
   - FFTS base addr, core/block IDs
   - Vec preamble if Vec-only (COOK-§1.5): `vid != 0` return, mask setup
   - Tile declarations with `TASSIGN` (including L1 Mat and L0 tiles if Cube)
   - Work loop (`get_block_idx()` / `get_block_num()`)
   - PTO tile-based compute body:
     - If Cube path: TLOAD GM->L1 Mat, TEXTRACT L1->L0, TMATMUL on L0 tiles (C26)
6. **Emit the launch function** — single `extern "C" __global__ AICORE void launch_*(...)`
7. **Emit the host wrapper** — `extern "C" void call_kernel(...)` with FFTS setup
8. **Validate against MCP** (mandatory):
   - For each PTO instruction you plan to use, call `get_cpp_intrinsic(instruction_name)` to verify its C++ signature and parameter types
   - For dtype/shape/backend constraints, call `get_constraints(instruction_name)` to verify legal combinations
   - If MCP returns no result or an error, fall back to cookbook patterns (COOK-§) and record the uncertainty as an evidence gap in the banner
   - Cross-check the chosen structure against cookbook pattern families
   - Never invent PTO instruction names — if MCP says an instruction doesn't exist, it doesn't exist
9. **Self-check** against the rule tiers — no CRITICAL violations allowed
   - Verify: no scalar `for`-loop matmul where TMATMUL is required (S3)
   - Verify: no TMOV from Vec to Left/Right — must use L1 Mat + TEXTRACT (C26)
   - Verify: TEXTRACT source M dimension is aligned to 16 (C26)
   - Verify: no Vec op (TMUL, TADD, TEXP, etc.) reads an operand that was only TLOAD'd — must push to pipeline with TMULS(x, x, 1.0f) first (C27)
   - Verify: every Strong-Form Defaults trigger this stage matches is EITHER satisfied by the emitted form OR carries an `OPTIMIZER-TARGET(<#>)` banner marker — no silent baseline (Strong-Form Defaults / checklist 2b)

---

## StageSpec Field Reference

Key fields the generator must inspect when reading a StageSpec:

| Field | Type | Required | Role |
|-------|------|----------|------|
| `name` | string | yes | Stage name (e.g. `"gate_prefix"`, `"recurrent_chunk_scan"`) |
| `inputs` | array | yes | Input tensors: each has `name`, `shape` (symbolic names or concrete ints), `dtype` |
| `outputs` | array | yes | Output tensors: same structure as inputs |
| `problem` | object | yes | Named dimension values (e.g. `{"BT":64, "K":128}` or `{"tile_size":64, "feature_dim":128}`) |
| `instruction_families` | array of strings | no | PTO instruction families — **primary archetype signal** (see decision tree) |
| `reference_source` | string | no | Python/PyTorch source code — reveals math semantics (`@`, `einsum`, `matmul` = contractions) |
| `lowering_hint` | string | no | Suggested lowering approach (free-form text) |
| `description` | string | no | Human-readable description of what this stage computes |
| `stage_index` | int | no | Position in stage plan (informational) |
| `code_region` | string | no | Source location in original algorithm (informational) |
| `evidence_gaps` | array | no | Known uncertainties — record, do not guess |
| _stage_spec_v1 only:_ | | | |
| `schema_version` | string | no | `"stage_spec_v1"` when present |
| `algorithm` | string | no | Algorithm name (e.g. `"naive_chunk_kda"`) |
| `source` | string | no | Source file the algorithm was extracted from |
| `stage_family` | string | no | Semantic family: `"prep"`, `"correction"`, `"recurrent"`, `"other"` |
| `stage_traits` | object | no | Additional traits; check `stage_traits.stage_subfamily` |
| `production_dimensions` | object | no | Production-scale dims — if absent, use `problem` values |
| `stage_subfamily` | string | no | Shorthand for `stage_traits.stage_subfamily` |

**Validation checklist** — before generation, verify:
- [ ] `name`, `inputs`, `outputs`, `problem` are present and non-empty
- [ ] Each input/output has `name` and `shape`; shape elements are strings or positive ints
- [ ] If `instruction_families` is present, it contains recognized PTO op names
- [ ] If `reference_source` is present and contains `@`/`matmul`/`einsum`, the stage requires Cube (see decision tree)
- [ ] If `production_dimensions` is absent, use `problem` values for buffer sizing

---

## Production Scale Awareness

Kernels must work correctly at production scale, not just at small test dimensions.
A kernel that passes validation at `HV=8` but crashes at `HV=256` is broken.

### Dimension Sources (Priority Order)

1. **`ProductionDimensions` input port** — If the workflow provides a non-empty
   dict via the `ProductionDimensions` port, use those values for scale validation.
2. **`stage.production_dimensions`** — If the StageSpec contains this field,
   use those values.
3. **`stage.problem`** — Fallback: use the problem dimensions. These may be
   test-scale values, so design the kernel with runtime-symbolic dimensions
   (per rule S4).

### How to Use Production Dimensions

When production dimensions are available:

1. **Buffer sizing**: Verify UB allocations fit at production scale.
   Example: if `HV=256` in production, ensure `BT * HV * sizeof(float)` fits in UB.

2. **Stride calculations**: Use production strides for GM offset computation.
   Example: if `H=16` and `K=128`, stride is `H * K = 2048`, not a hardcoded guess.

3. **Loop bounds**: Verify iteration counts are reasonable at production scale.
   Example: if `NT=16` and `BT=128`, total tokens = 2048.

4. **Flag/barrier arrays**: Size synchronization structures for production head counts.
   Example: if `HV=256`, flag arrays must accommodate 256 entries.

5. **Tile allocations**: Ensure tile counts and sizes work at production scale.
   Example: if `K=128` and tile width is 64, you need 2 tiles per row.

### Common Scale-Related Bugs

| Bug | Test scale (passes) | Production scale (fails) |
|-----|---------------------|--------------------------|
| Hardcoded head count | `H=8` stride works | `H=16` stride overflow |
| Fixed buffer size | `HV=8` fits in UB | `HV=256` UB overflow |
| Small flag array | 8 flags enough | 256 flags needed |
| Toy tile count | 1 tile per row | 4 tiles per row |
| Assumed alignment | `K=64` aligned | `K=128` misaligned |

### Example: Production Dimensions in StageSpec

When available (stage_spec_v1 format), `production_dimensions` provides production-scale
values that differ from test-scale `problem` values:

```json
{
  "name": "gate_prefix",
  "problem": {"BT": 64, "K": 128, "HV": 8},
  "production_dimensions": {"BT": 128, "K": 128, "HV": 256, "H": 16, "V": 128, "NT": 16},
  "instruction_families": ["TLOAD", "TADD", "TSTORE"],
  ...
}
```

In this example:
- `problem` has test-scale `HV=8` — validation runs at this scale
- `production_dimensions` has `HV=256` — agent verifies kernel works at this scale
- Agent generates kernel with runtime-symbolic `HV` (rule S4) but checks that
  `BT * HV * sizeof(float) = 128 * 256 * 4 = 131072` bytes fits in UB (192KB)

### When Production Dimensions Are Not Available

If `production_dimensions` is absent (common for stage plan entries):

1. Use `problem` values as representative dimensions for all sizing
2. Generate kernel with runtime-symbolic dimensions (per rule S4)
3. Document in banner evidence gaps that production scale is unverified
4. The workflow may provide production dimensions via a separate port

---

## Token Budget Guide

Scale generation verbosity to stage complexity. The goal is a correct,
complete kernel — not maximum detail on every sub-expression.

| Stage complexity | Indicators | Guidance |
|------------------|------------|----------|
| **Simple** | `vec_only`, ≤2 PTO ops, single input/output | ~100-200 lines. Minimal comments, straight-line code. |
| **Medium** | `vec_only` with accumulation, or single Cube contraction | ~200-350 lines. Comment the work loop structure and UB map. |
| **Complex** | `cube_vec_pipeline`, multi-phase, cross-core sync | ~350-500 lines. Comment each phase, name workspace regions, document flag protocol. |
| **Very complex** | Recurrent scan with state carry, closure + projection | ~400-600 lines. Document state semantics, closure invariants, projection sites. |

Do not pad with redundant comments. Do not repeat rule citations in code
comments unless the pattern is unusual. Prefer self-documenting variable
names over comment walls.

---

## Forbidden Patterns (Consolidated)

This is the single authoritative reject list. These patterns must never
appear in generated output. References point to the rule that forbids them.

| Pattern | Rule | Why |
|---------|------|-----|
| `ptr[idx]` on `__gm__` pointer | C1 | NPU Alarm crash |
| `#include <pto/pto_instr.hpp>` | C2 | Wrong header |
| `#include <runtime/rt_ffts.h>` or any other include besides `kernel_common.h` | C2 | Redundant, conflicts with kernel_common.h |
| `#include "acl/acl.h"` or `#include <cmath>` | C2 | Already provided by kernel_common.h |
| `using namespace pto;` outside device guard | C2 | Host compile failure |
| `WF_HAS_PTO_STAGE_IMPL` or similar guard macros | C2 | Indirection forbidden |
| `#define AICORE` redefinition | C2 | Already defined by kernel_common.h |
| `launch_*` function defined more than once | C3 | Must be defined exactly once |
| `launch_*` inside `#if` and `#else` blocks | C3 | Define once, use `#if` inside body |
| Vec intrinsics outside `#if defined(__DAV_C220_VEC__)` | C12 | Target feature not supported |
| `set_vector_mask`, `get_subblockid` without Vec guard | C12 | Must be inside Vec guard |
| Unicode characters in comments (—, –, →, ×, etc.) | C13 | Compiler rejects non-ASCII |
| Em-dashes (—) or curly quotes ("", '') | C13 | Use ASCII: `--`, `"`, `'` |
| Any non-ASCII character (code point > 127) | C13 | Bisheng compiler error |
| `VecShape`, `VecStride`, `VecGlobal`, `MakeGlobal` | C4 | Invented aliases |
| `BLayout::RowMajor, SLayout::NoneBox` on Mat | C5 | Compile failure |
| `TEXTRACT(L1Mat) → TileRight` for transposed B | C5 | Must TRESHAPE first |
| `wait_flag_dev(N)` with no prior producer | C6 | Deadlock on iteration 0 |
| `static_assert` missing for UB address map | C7 | Silent UB overflow |
| Both vids active on same UB addresses | C8 | Data corruption |
| Missing `pipe_barrier(PIPE_ALL)` after TLOAD/TSTORE | C9 | Data corruption |
| `exp()`, `expf()`, `std::exp()`, `__builtin_expf()` | S2 | Use TEXP on tiles |
| `exp_scalar()`, polynomial transcendental helpers | S2 | Use TEXP on tiles |
| Scalar `for`-loop matmul as dominant path | S3 | Use TMATMUL |
| Skipping TMATMUL because inputs are fp32 | C26, S3 | fp32 TMATMUL supported on A2/A3 |
| Cube/TMATMUL for an M=1 GEMV or rank-1 outer product | S3 | Doesn't fill a Cube fractal; use Vec TROWEXPAND/TCOLEXPAND+TMUL+TCOLSUM |
| Cube inside a loop-carried scan with tiny K/V | S3, C6 | Per-iter FFTS handshake is a deadlock hazard and unamortized; use Vec |
| `TLOAD` GM extent = compile-time CAP instead of runtime dim | C28 | Over-reads tail garbage; compounds in recurrent state |
| Single-column `[N,1]` tile (RowMajor or ColMajor) | C28 | Rejected by Tile lib / TADD / TSTORE; keep >= 8 wide |
| `TROWEXPANDMUL` with RowMajor `[N,8]` per-row scalar | C28 | Needs ColMajor `[N,1]`; else zeroes outside col 0 |
| Transposed state + `TROWSUM` when matvec reduces over the other axis | S9, C28 | Layout inconsistency; per-step error compounds over scan |
| `TMOV` from Vec tile to Left/Right operand | C26 | Use L1 Mat + TEXTRACT path |
| TEXTRACT with M not aligned to 16 | C26 | Pad M to M_PAD=16 |
| `TMUL(a, b, c)` where `b` or `c` came directly from `TLOAD` | C27 | Push to pipeline with `TMULS(x, x, 1.0f)` first |
| Any Vec op reading an operand that was only `TLOAD`'d | C27 | All Vec ops read from pipeline, not buffer |
| `o = u` copy-through in recurrent stages | S9 | Must do real contraction |
| `skeleton_only` for semantically specified stages | S5 | Must produce real math |
| `#define BT 64` from one observed case | S4 | Keep runtime-symbolic |
| Bare short all-caps constant `constexpr int BT/K/V/N = ...` | C29 | Collides with a PTO library symbol -> silently mis-sized tiles; use `kBT`/`INV_BT` |
| `TTRI<float, 0>(dst, diag)` (binds TileData=float) | C30 | First template arg is the tile type; use `TTRI<decltype(dst), 0>` |
| `extern "C"` on a templated `launch_*<N>` function | S3 | Templates cannot have C linkage; only the dispatching `call_kernel` is `extern "C"` |
| `constexpr int f(int)` helper called from `[aicore]` code | C23, S3 | `[host]` function; use a constexpr ternary or template-constant struct |
| `total_work` assumed to be elements when the harness passes rows | C18 | Derive `num_mat` from `chunk_size`; verify `num_mat > 0` |
| Vec micro-GEMM for a dense matrix-MATRIX (M,N,K >= 16) to dodge the Cube handshake | S3, C6 | Order-of-magnitude regression; use Cube (split-launch needs no handshake) |
| `TTRANS` of a value still in the Vec pipeline (not committed to buffer) | C31 | Transposes stale data -> wrong/nondeterministic; GM round-trip the source first |
| Transpose at a runtime-short tile height | C31, S3 | Unreliable; template on size or zero-pad to full static height |
| Fine-grained per-op Cube<->Vec GM relay in a loop-carried stage | S9 | Slow + coherency-fragile; use one coarse boundary or an in-kernel handshake |
| `{"outputs": {"Kernel": "..."}}` JSON wrapping | Output | Framework handles routing |
| Banner comment after first `#include` | A1 | Banner must be BEFORE all includes |
| Any text before banner comment | A1 | Banner must be first content in file |
| Prose before first `#include` | A1 | Banner comment first |
| Quoted C++ with literal `\n` escapes | Output | Raw source text only |
| `"I am inspecting local kernels..."` | Output | Return only the translation unit |

---

## Failure Policy

If you cannot implement a full high-quality stage:

- Still return compile-oriented structure with correct host/device split
- Keep compute body as minimal PTO-op path
- Record exact blockers as source comments near the relevant code
- Do not silently downgrade to scalar pointer loops
- Do not replace unresolved symbolic dimensions with guessed fixed macros
- Do not replace unresolved PTO math with custom scalar approximations
- For semantically specified stages: use the nearest real local pattern
  (COOK-§6, §8) and keep only the unresolved fragment as an evidence gap
- If runtime-symbolic dimensions are the only blocker for a TMATMUL lowering,
  keep the contraction and make the runtime bound explicit

---

## Reviewer Mode

When invoked as the second-pass reviewer/fixer over a draft kernel,
read and follow **`REVIEWER.md`** for the full repair protocol.

---

## Examples

See **`examples.md`** for:
- Annotated failure patterns with explanations (EX-§1)
- Full Vec-only example kernel (EX-§2)
- Full Cube+Vec pipeline example kernel (EX-§3)
