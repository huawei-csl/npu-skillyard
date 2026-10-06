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
7. □ Plan work distribution — COOK-§1.68/§1.69 (grid-stride or varlen).
      **C124: enumerate the grid-stride ITEM COUNT for every shape in the declared contract
      and compare it against `block_dim`.** An item count below the block dim means the kernel
      runs on a fraction of the device and NO correctness gate can see it. Detect-and-decide:
      splitting a second axis costs a per-step grid barrier, and raising a `block_dim` cap buys
      nothing while the item count binds.
     Derive elements_per_iteration from problem and input shapes.
8. □ Choose tile shapes — fixed compile-time tiles, runtime outer loops
9. □ Draft UB address map — COOK-§1.6/§3 with static_assert guard
     If Cube path: allocate addresses for L1 Mat staging tiles and L0 tiles (C26).
10. □ Plan data path to TMATMUL — GM -> L1 Mat (TLOAD) -> TEXTRACT -> L0 Left/Right -> TMATMUL (C26).
      Never use TMOV from Vec to Left/Right. Pad M to 16 if needed for TEXTRACT alignment.
11. □ Plan Vec pipeline data flow — after every TLOAD, `set_flag`/`wait_flag`
      `MTE2 -> V` before any Vec op reads the tile (C27). Do NOT add a
      `TMULS(x, x, 1.0f)` "push": it is unnecessary when the handshake is present,
      and when the handshake is MISSING it makes the kernel wrong on every run
      instead of some. Probed, see C27.
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
  │   NO  → cube_only          → COOK-§7, §8.7-§8.9, EX-§3  ·  build -cube (C33c)
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

**C1. Move BULK data between GM and UB only through MTE.**
ALL bulk data transfer between GM and UB uses `TLOAD` (GM→UB) and `TSTORE`
(UB→GM) with `GlobalTensor` descriptors — mask generation, workspace init,
output writes, everything that moves a tile. A scalar loop is not a substitute
for MTE and will be orders of magnitude slower: MTE moves a whole tile per
issue, a scalar loop moves 4 bytes.

**But a scalar `__gm__` access is legal, and is the supported way to read a
runtime scalar.** An earlier version of this rule said `ptr[idx]` "crashes the
NPU into Alarm state requiring a hardware reset". **That is false.** It was
probed directly on A2/dav-c220 (`isa_probes/probe_gmscalar.cpp`, 64/64 exact
values on every mode, device healthy before and after, each mode in its own
process):

| what was probed | result |
|---|---|
| scalar READ `p[i]` on Vec, no `dcci` | PASS |
| scalar READ `p[i]` on Vec, `dcci` first | PASS |
| scalar WRITE `out[i] = v` on Vec | PASS |
| `TSTORE` then scalar read back, no `dcci` | PASS, fresh |
| `TSTORE` then scalar read back, with `dcci` | PASS, fresh |
| scalar READ on the **Cube** core | PASS |

The library does this itself: `pto/comm/a2a3/TWait.hpp` spins on
`basePtr[idx]` of a `volatile __gm__ int32_t *`, and
`pto/comm/async_common/ccu_trigger.hpp` writes an MMIO register by scalar
store — noting that `-cce-aicore-dcci-insert-for-scalar=false` *disables*
automatic `dcci` insertion for scalar stores, i.e. the compiler inserts them by
default.

Use it for exactly one thing: **reading a runtime scalar you need before you can
build a descriptor** — a group boundary, a token count, a dynamic tile schedule.
This is what makes runtime-determined shapes implementable at 1x FLOPs instead
of padding to a worst case. Note `Tile::GetValue` cannot serve this: it is
`static_assert(Loc == TileType::Vec)` (`pto_tile.hpp:1457`), so it reads UB, not
GM, and it is unavailable on Cube.

**A scalar `__gm__` WRITE has 32-byte cache-line granularity. Never use one to
materialize an array whose words are spread across lanes.** This is the one
genuinely dangerous case, and the first version of this rule missed it because
its probe used a single lane. Measured (`isa_probes/probe_scalarscatter.cpp`,
8192 words, lanes = 2 x block_dim since both AIV sub-blocks are workers):

| store pattern | bd=1 | 2 | 4 | 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| `TSTORE`, contiguous per lane | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| scalar, contiguous per lane | 0 | 0 | 0 | 0 | 0 | 0 | 312 |
| scalar, **interleaved across lanes** | **4096** | 6132 | 7152 | 7633 | 7649 | 7662 | 7589 |

Interleaved means lane `L` writes indices `i % nlanes == L`, so adjacent words
belong to different lanes -- the shape a permutation or scatter has. At
`block_dim=1` that is **exactly half the array lost**, and `block_dim=1` is a
SINGLE core: the two AIV sub-blocks alone are enough to destroy it. A scalar store
commits a whole 32-byte line, so when two lanes hold different words of one line,
one lane's words are overwritten. No fault, no partial write -- just missing data.

So:
* **Reading a runtime scalar: fine, use it.**
* **Writing a single scalar or a contiguous run owned by one lane: fine** (clean to
  48 lanes; 96 lanes showed losses even contiguous, so do not oversubscribe).
* **Writing an array whose words interleave across lanes: NEVER.** Use `TSTORE`,
  which is exact at every lane count tested. If the natural formulation scatters,
  restructure so each worker owns a contiguous run and stores it with MTE. This is
  the same failure mode as the known `MSCATTER<Elem>` defect, so it is not
  `MSCATTER`-specific -- it is the scalar store path itself.

Two further requirements, neither of which is a crash:
* Declare the pointer `volatile __gm__ T *`. Without it the compiler may hoist
  the load out of a spin loop.
* For a value another agent may have written (another core, the host, a DMA),
  invalidate first: `dcci((__gm__ void *)(p + i), SINGLE_CACHE_LINE);`. Omitting
  it risks **stale data, not a fault**. The probe above saw no staleness in the
  same-core store-then-read case, so this is the library's cross-agent practice
  rather than a measured same-core requirement — follow it when a *different*
  agent produced the value.

→ PLAT-§Illegal

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

### C34: A COMPILE-TIME KNOB MUST CHANGE THE INSTRUCTION STREAM, NOT JUST THE ALLOCATION

If you emit a `#define` knob and name it as an optimizer target, it must be **wired end to
end**. A knob that only widens a buffer reservation is a no-op that costs capacity, and it
will be measured, lose, and retire the technique it was named after.

The instance: `grouped_matmul` emitted `PTO_DBUF_L1` / `PTO_DBUF_L0` and a header comment
naming double buffering as "the second lever". They resolved to `kL1Slots = 2`, widening an
offset and a `static_assert`. But every buffer was bound once, before the K loop:

```cpp
TASSIGN(a_l1, LM::A_L1);      // slot 0, hoisted out of the loop
for (int64_t kb = 0; kb < nk; ++kb) { ... }   // zero TASSIGN inside
```

Nothing selected `kb & 1`; no `set_flag`/`wait_flag` distance changed. The reserved slot
shifted the addresses of the buffers allocated after it, so the variants were numerically
WRONG at the no-bias config (rel err 4.9e-1 and 6.2e-2) and raised **aicore exceptions** at
larger shapes. The kernel shipped single-buffered and is **1.94x** off the vendor's
steady-state throughput -- the entire gap -- with "double buffering rejected" in its record.

**If you emit a buffering knob, it must do all three:**

1. **Bind inside the loop, indexed by the iteration** -- `TASSIGN(a_l1, LM::A_L1 + (kb & 1) * A_L1_STRIDE)`,
   not once before it.
2. **Increase the producer/consumer flag distance.** Adjacent `set_flag(P, C, e)` /
   `wait_flag(P, C, e)` is a BARRIER. Overlap requires waiting on the flag set one iteration
   earlier -- issue step `kb+1`'s TLOAD, then wait for step `kb`'s. Use distinct event ids
   per slot.
3. **Keep the allocation valid at BOTH settings.** Anything addressed after a slotted buffer
   moves when the slot count changes; `static_assert` the layout at each setting and validate
   at every contract config, including the ones that exercise different optional inputs.

If you cannot satisfy all three, **do not emit the knob**. Write the limitation in the header
as an unbuilt target instead. A named-but-unwired lever is worse than an absent one, because
the campaign will spend an attempt disproving a technique you never implemented.

### C33c: A CUBE-ONLY STAGE MUST NOT SHIP AS A MIX LAUNCH EITHER -- `dav-c220-cube` exists

> **C33-CHAIN -- RESOLVED, AND C33b/C33c WERE PRICING THE TOLL ON THE WRONG UNIT.**
>
> C33b and C33c say a single-engine stage must not ship as MIX, at a documented ~2.88 us toll. On a
> **chain** that rule forbids single-launch composition, because an FFTS kernel is one TU with one
> arch flag and three stages wanting three flags cannot be fused. A first pipeline run duly fell
> back to host-stream -- and that fallback created the `torch -> direct-launch` edge that exposed
> C60 (40/40 wrong before a sync was added).
>
> **A controlled A/B then settled it.** Same binary, same partitioning, same `block_dim`, driven
> either as one FFTS launch or one-phase-per-launch -- only the seam mechanism differs:
>
> | | host-stream (3 `.so`) | **FFTS (1 launch)** |
> |---|---|---|
> | operator score | 52.422 | **54.279** |
> | total device time, 20 cases | 16,067.8 us | **8,924.6 us** |
> | controlled A/B | -- | **wins 19/20, geomean 1.259x** |
> | vs composed vendor primitives | wins 19/20 | **wins 20/20, geomean 4.38x** |
>
> **The MIX toll is real and BIGGER than documented -- and that does not matter.** Measured with
> C33b's own recipe (same source, matched worker count, bit-identical output asserted first, all 8
> arms identical): vector-only in MIX **+2.95 to +7.51 us/call**, cube-only **+5.40 to +7.30**. Not
> 2.88, and **not Cube-specific** -- which also supports C33c's open 9-13 us discrepancy.
>
> **The error was the unit.** The toll is paid **once per launch**, not once per stage. C33b/C33c
> priced it per stage, so an N-stage chain appeared to cost N tolls; in fact single-launch pays it
> **once** while host-stream pays `1+2L` launches at ~5.5-5.8 us each plus the host syncs.
>
> **RULE: if the operator is a CHAIN of more than one stage, pick `dav-c220` (MIX) for the whole
> chain and compose with `SYNCALL<Mix>` seams.** Reserve `-vec`/`-cube` for an operator that
> genuinely ships as ONE single-engine kernel. Arch is then a **Phase-1 decomposition constraint**:
> choose it first, and require the decomposition to fit it.
>
> Three caveats that come with it: the advantage **narrows as `block_dim` grows** (a seam scales
> with core count, a launch does not -- the single A/B loss was at bd=24), so pick `block_dim`
> rather than maximising it; `block_dim` becomes a **correctness** bound rather than a knob, capped
> by the DEVICE's queried core count and not by the literal 24 this was measured at (C57a, C66);
> and pulling a stage loop in-kernel pushes cost into the contract (that run locked `numLayers <= 3`).
>
> Bonus: single-launch removes every `torch -> ours` edge, so C60 cannot arise -- **0 wrong in 400
> unfenced trials**, including the interleaved-matmul shape that went 40/40 wrong under host-stream.
>
> **SECOND DATA POINT, and it is a scoped NEGATIVE that SUPPORTS the rule as written
> (`grouped_matmul`, 78.20).** The rule is gated on *"if the operator is a CHAIN of more than one
> stage"*, and that gate earned its keep. `grouped_matmul` ships as **ONE `cube_only` kernel**
> because both would-be Vec stages disappear into hardware paths: the fp16 bias reaches the Cube
> bias table through `Mat -> TMOV -> Bias` (half->float converted **in hardware**), and the fp32
> accumulator reaches GM **already cast** through the FIXPIPE on `TSTORE`. With one stage there is
> no seam to collapse, so single-launch FFTS saves **zero** launches while MIX still charges the
> toll: same source built MIX is **bit-identical on 20/20** and **1.071x slower** (+1.40 us/call
> mean, +7.80 us worst).
>
> **So count the stages that survive lowering, not the stages in the StageSpec.** A Vec stage that
> is only a dtype conversion or a bias add is not a stage -- `TMOV`-to-bias-table and the FIXPIPE
> cast absorb it, and a StageSpec that still lists it will talk you into a MIX build that costs
> 7.1% and buys nothing. Check for a real Vec/Cube seam before applying C33-CHAIN.

### C33c: A CUBE-ONLY STAGE MUST NOT SHIP AS A MIX LAUNCH EITHER -- `dav-c220-cube` exists

**C33b's mirror, and the generator has been leaving it on the table.** A stage whose StageSpec
contains no Vec op is `cube_only`, and it must be built with `--cce-aicore-arch=dav-c220-cube`,
not `dav-c220`.

The comment "A2/A3 has no cube-only arch flag" that appeared in generated kernels is **false**.
Probed directly with `bisheng -dM -E` on this toolchain (CANN 9.1.0):

| `--cce-aicore-arch=` | macros defined |
|---|---|
| `dav-c220` | `__CCE_AICORE__` **`__CCE_AICORE_ENABLE_MIX__`** `__DAV_C220_CUBE__` `__DAV_C220_VEC__` `__DAV_CUBE__` `__DAV_VEC__` |
| `dav-c220-vec` | `__CCE_AICORE__` `__DAV_C220_VEC__` `__DAV_VEC__` |
| **`dav-c220-cube`** | `__CCE_AICORE__` **`__DAV_C220_CUBE__`** `__DAV_CUBE__` |

`dav-c220-cube` is accepted and defines the Cube macro **alone**, without
`__CCE_AICORE_ENABLE_MIX__`. It is the exact mirror of `dav-c220-vec`.

**What it is worth.** A noop-launch probe on `attention_sdpa` priced the MIX launch at
**4848.8 ns against 1172.5 ns** at `block_dim=16` -- roughly **4.08 us of pure launch overhead
on a ~9.5 us kernel**, worth **1.90x** from a compile flag alone. `quant_matmul` measured
1.04-1.19x from the same flag and validated **bit-identically** against its MIX build.

**Why this is SAFER than C33b's vector case.** C33b carries a partition hazard: rebuilding
unmodified vector source as `-vec` computes NaN on about half the output, because the work
partitioning assumes mix geometry (`lanes = 2 x block_dim`, both AIV sub-blocks as workers), so
the fix must happen at generation. **A Cube core has no sub-blocks**, so there is no equivalent
lane-count change and a cube-only rebuild is layout-neutral. Validate it anyway -- expect
bit-identical output, and treat any difference as a real defect rather than an expected
consequence.

**Unverified, and it bounds the rule.** The interaction of `dav-c220-cube` with `SYNCALL` and
cross-core flags is NOT established. A chain that mixes a `-vec` stage and a `-cube` stage is
heterogeneous-arch, and a single-launch FFTS kernel is one translation unit with ONE arch flag --
so `ffts` composition and per-stage single-engine builds are mutually exclusive. Choose
per-stage single-engine builds with `host-stream` composition unless you have measured otherwise;
`attention_sdpa` found `SYNCALL<AIVOnly>` deadlocks in that configuration.

**One open discrepancy, deliberately not folded into C33b's number.** `ffn` measured a flat
**9-13 us** for rebuilding a Cube stage inside a MIX binary, where C33b documents ~2.88 us.
7-10 us is unexplained. Do not quote 2.88 us for a Cube stage until that is probed; measure the
toll for your own stage with a noop-launch probe and report what you measured.

### C33b: A VECTOR-ONLY STAGE MUST NOT SHIP AS A MIX LAUNCH -- it costs 2.88 us per call

**Every kernel this generator has produced launches as `MIX_AIC`** (CANN
`KERNEL_TYPE_MIX_AIC_1_2`, profiler `Accelerator Core = MIX_AIC`, `Mix Block Num = 2 x Block
Num`) -- **including pure-vector elementwise ops where the Cube engine does no work at all.** The
vendor ships `AI_VECTOR_CORE` for exactly those ops.

A MIX launch carries a **flat 2.88 us of device-side dead time per call**, on top of the kernel's
reported `Duration`. Measured as a crossed 2x2, with durations fully overlapping between groups so
it is not a kernel-size effect:

| launch mode | path | n | gap median |
|---|---|---|---|
| MIX | ours (`ctypes`) | 14 | **2.880 us** |
| MIX | vendor (ACL) | 3 | **3.009 us** |
| single-engine | ours (control build) | 5 | **0.010 us** |
| single-engine | vendor (ACL) | 5 | **0.079 us** |

The toll is a property of MIX mode, not of our code or our submission path -- the vendor pays it
too on its MIX kernels. **The defect is paying it on work that needs one engine.** It is 288x the
single-engine gap, and it is **not reducible by any runtime knob**: `TASK_QUEUE_ENABLE`, K, queue
run-ahead, `block_dim` and submission API all move it by <=1.6%.

**Why this dominates small shapes.** It is flat, so it is invisible at production scale and
decisive below ~10 us of device work. On three cases whose device duration is ~2.3-2.4 us, the toll
more than doubles delivered time: `gelu`, `dynamic_quant` and `reshape_and_cache` all read as wins
on `Duration` and as **losses on delivered period** (2.05x, 1.29x and 1.25x slower respectively).

**Do NOT try to fix this with a compile flag.** Rebuilding the *unmodified* source with
`--cce-aicore-arch=dav-c220-vec` does produce `AI_CORE` with a 0.010 us gap -- and **computes the
wrong answer** (about half the output NaN), because the work partitioning assumes the mix geometry
(`lanes = 2 x block_dim`, both AIV sub-blocks as workers). **The fix belongs in generation:** when a
stage uses no Cube instruction, emit a single-engine kernel *and* partition the work for one engine,
rather than emitting mix geometry and changing the arch flag underneath it.

**Rule:** decide the launch mode from the stage's archetype. If the StageSpec contains no Cube
operation (`TMATMUL*`, `TMOV` to/from `Acc`), the stage is vector-only and must be generated,
partitioned and built single-engine. Reserve MIX for genuine Cube+Vector work, where the ~3.0 us is
a floor the vendor pays as well.

**THE PARTITION RULE, stated positively.** Measured with a direct geometry probe on this part:

| arch flag | workers per block | `get_subblockid()` |
|---|---|---|
| `dav-c220-vec` (single-engine) | **1** | identically **0** |
| `dav-c220` (mix) | 2 | 0 and 1 |

So in a single-engine kernel, **partition on `get_block_idx()` / `get_block_num()` ALONE.**
Do NOT write `lane = 2 * get_block_idx() + get_subblockid()`: under `-vec` only even lanes
would ever exist, and **half the output is never written** -- that is the exact mechanism
behind the "half the output NaN" result above. `COOK-§1.5`'s Vec preamble
`if (get_subblockid() != 0) return;` is a harmless no-op under `-vec`, but do not let it lead
you into mix-style lane arithmetic.

The failure is **direction-dependent**, and neither direction is caught by validation:
mix-geometry source built `-vec` is silently **WRONG**; single-engine source built for mix is
silently **SLOW** (both sub-blocks redundantly compute the same lanes). Only a launch-mode
check on the profiler's `Accelerator Core` column sees either.

**How to EVIDENCE the fix as a number rather than assert it.** Build the *same source* with
the mix flag at **half the `block_dim`**, so worker count and work partition are identical and
only the launch mode differs; verify the two produce bitwise-identical output, then compare
delivered device period. That control reproduced the toll at **2.72-3.03 us on eleven
independent kernels**. Compare launch modes at equal WORKER COUNT, never at equal `block_dim`
-- the latter conflates the launch mode with the partition and inflates the difference.

**How much the redundant second AIV costs depends on what the stage is bound by:** ~1-2% on a
compute-bound stage (the duplicated work is arithmetic the other sub-block was doing anyway),
but **~2x on a traffic-bound stage**, because the redundant sub-block duplicates the `TLOAD`s
and `TSTORE`s too. Measure it; do not quote either number.

**A composed chain is legitimately HETEROGENEOUS** -- MIX on the Cube stages, single-engine on
the vector ones. That is the correct outcome, not an inconsistency.

**`SYNCALL<AIVOnly>` DEADLOCKS, and the cause is NOT the participant count.** An earlier
version of this rule attributed it to a `-vec` launch presenting half the AIV sub-blocks the
FFTS barrier expects. That explanation is **falsified**: it deadlocks in a `dav-c220` (mix)
build as well, and at `block_dim` 1, where any participant-count argument is trivial. The
mechanism is unknown -- record it as an evidence gap, do not repeat the explanation.
`SYNCALL<Mix>` is not a workaround either: it reinstates the toll on the vector stages.

For a vector-only chain the options are stream-ordered launches (device-side seam measured
at **~0.02-0.03 us**, effectively free) or the library's soft GM-counter barrier (measured
**~15 us/seam** on one chain -- three orders of magnitude worse, so prefer stream ordering
unless you need an in-kernel barrier). **Measure your seam and report it**; the two choices
are not close, and picking the expensive one silently costs more than most optimisations
recover.

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

**C8. Vec subblock UB is PRIVATE (corrected).**
Each Vec sub-block has its OWN 192 KB UB (184 KB usable below TMP_UB_OFFSET).
Static TASSIGN addresses ARE private per vid, so both vids may use the same
addresses. Do NOT partition UB between vids -- that halves the budget for
nothing -- and do NOT return on nonzero vid to dodge a hazard that does not
exist, which throws away half the Vec throughput. Hardware-probed with a
positive control; see PLAT-§UB. The only shared region is the TMP_UB_OFFSET
library scratch.
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
above is correct for a Vec-ONLY kernel (C8). It is WRONG before a
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
- a 1x1 tile — violates 32-byte alignment
- any `Tile<TileType::Vec, float, R, C, ...>` where `R * C * sizeof(float) < 32` bytes
- Tiles with `Cols < 8` for float32 (minimum 8 floats = 32 bytes)

REQUIRED minimum tile sizes:
- For `float32`: minimum `1 x 8` (32 bytes)
- For `float16`: minimum `1 x 16` (32 bytes)
- For scalar values: use a `1 x 8` fp32 tile and access element 0 via `GetValue(0)`

When you need to store a single scalar result (e.g., from a reduction), use:
```cpp
// NOTE: `UbND` is NOT a pto-isa type. It is a LOCAL alias, declared in EX-3 as
//   template <typename T, int R, int C>
//   using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor, -1, -1>;
// Either declare that alias yourself or write the Tile<> form directly, as EX-2 does.
Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1> scalar_tile(1, 8);
TASSIGN(scalar_tile, SCALAR_UB_ADDR);
TROWSUM(scalar_tile, source_tile, temp_tile);
float result = scalar_tile.GetValue(0);
```

Do NOT use a 1x1 tile — it will fail compilation with alignment errors. → PLAT-§Alignment

**C15. Reduction instruction correctness.**
PTO reduction instructions have specific input/output shape requirements:

- `TROWSUM(dst, src, temp)`: Reduces each row of `src` to a single value in `dst`
  - `src`: `Tile<TileType::Vec, T, R, C, BLayout::RowMajor, -1, -1>` (R rows, C cols)
  - `dst`: `Tile<TileType::Vec, T, R, 8, BLayout::RowMajor, -1, -1>` (R rows, minimum 8 cols for alignment)
  - `temp`: **must scale with the SOURCE width, not the destination.** `src/2` is
    exact; `src/4` is silently WRONG (measured ~0.22 relative error). A `[R,8]`
    scratch is correct only for a narrow `src` -- see the corrected note below.
  - **A too-narrow `temp` can make TROWSUM write NOTHING AT ALL.** Verified in
    `a2a3/TRowReduceOps.hpp:281-285`. In the band
    `elemPerRpt < validCol < 2 * elemPerRpt` the implementation does:

    ```cpp
    if (validCol < 2 * elemPerRpt) {
        // comment in the library: present only to satisfy a ccec compile check
        if constexpr ((srcRptStride < BLOCK_MAX_PER_REPEAT) || (tmpRptStride < BLOCK_MAX_PER_REPEAT)) {
            return;                     // <-- dst is NEVER WRITTEN
        }
        ...
    ```

    `BLOCK_MAX_PER_REPEAT = 8` (32 B blocks), so a stride below it means a tile row
    narrower than 256 B = **64 floats**. `elemPerRpt = 256/sizeof(T)`, so for fp32 the
    exposed band is `64 < validCol < 128`. Because `validCol > 64` already forces the
    *source* stride above the threshold, **the trigger in practice is a narrow `temp`** --
    exactly the `[R,8]` scratch this rule already warns about.

    The consequence is different from, and worse than, a wrong value: `dst` keeps
    **whatever was in that UB region before**, so the symptom is stale or garbage data that
    *changes between runs and disappears under a debugger*. That reads exactly like a
    cross-core sync race, and it is not one -- it is a silent no-op. If a reduction result
    looks uninitialized rather than merely inaccurate, check the scratch width **before**
    adding barriers or flags.

    Give `temp` at least `src/2` columns AND at least 64 floats (256 B) per row, and
    `static_assert` both.
  - Result: `dst.GetValue(i)` contains sum of row `i` from `src`

- `TCOLSUM(dst, src)`: Reduces each column of `src` to a single value in `dst`
  - `src`: `Tile<TileType::Vec, T, R, C, BLayout::RowMajor, -1, -1>` (R rows, C cols)
  - `dst`: `Tile<TileType::Vec, T, 1, C, BLayout::RowMajor, -1, -1>` (1 row, C cols) — NOT `Tile<TileType::Vec, T, 1, 8, BLayout::RowMajor, -1, -1>`
  - Result: `dst.GetValue(j)` contains sum of column `j` from `src`

Common mistake: Using `TCOLSUM` to reduce a 1×K tile to 1×1. This is WRONG.
Correct approach for reducing 1×K to scalar:
```cpp
// WRONG: TCOLSUM(sum_1x1, g_row_1xK);  // sum_1x1 is a 1x1 tile — INVALID

// CORRECT: Use TROWSUM with proper shapes
Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1> g_row(1, k);   // Source: 1xK
Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1> sum_tile(1, 8);  // Dest: 1x8
Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1> temp_tile(1, 8); // Workspace
TASSIGN(g_row, G_ROW_ADDR);
TASSIGN(sum_tile, SUM_ADDR);
TASSIGN(temp_tile, TEMP_ADDR);
TROWSUM(sum_tile, g_row, temp_tile);           // Reduces 1×K → 1×1 (stored in element 0)
float scalar_result = sum_tile.GetValue(0);
```

**CORRECTED -- there is NO 64-lane reduction limit. The old rule here was a
misdiagnosis, and this rule's own scratch sizing was what caused the symptom.**

`TROWSUM` and `TROWMAX` are EXACT at 128-wide rows on dav-c220 / CANN 9.1.0. Three
independent hardware probes agree: a standalone `TROWMAX` sweep (correct at widths
64/128/1024, max placed at every column); a run that found `TROWMAX`'s scratch must be
src-shaped for float/half because it reduces as a binary tree; and a run that measured
`TROWSUM` exact at 128 wide with `tmp = src/2` and deterministically wrong (~0.22
relative) with `tmp = src/4`, while `TROWMAX` with that same undersized tmp stayed exact.

So the real rule is: **the scratch is proportional to the SOURCE width, and under-sizing
it is silently wrong rather than an error.** The `[R, 8]` scratch this rule used to
prescribe is under-sized for any `src` wider than 16 -- which produces exactly the
"partial sum with no error" that was then attributed to lane truncation.

Cost of the stale rule, measured: it forced a Vec chunk 2x narrower than necessary in a
generated attention kernel; removing it was that kernel's single largest speedup
(**1.617x**). Do not re-derive the 64-lane rule from older run reports.

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

**C19 (CORRECTED). `GetValue` after a Vec op needs a `V->S` / `S->V` FLAG SANDWICH,
not a GM round-trip.** The stale-read symptom is real; the "results live in pipeline
registers, not the tile buffer" explanation is **wrong**, and the GM round-trip this
rule used to prescribe **is itself broken**. Probed, 4096 rows x 5 runs, scalar
readback compared against the same tile DMA'd out:

| sync used | wrong |
|---|---|
| none | 4096 / 4096 |
| `pipe_barrier(PIPE_V)` | 4096 / 4096 |
| `V->S` flag only (half the sandwich) | **latent race** -- 0, 14, 18, 0 across four builds differing only in UNRELATED code |
| **the old C19 GM round-trip, synced `MTE2->V`** | **4048 / 4096 -- the prescribed fix is WRONG** |
| GM round-trip synced `MTE2->S` | 0 / 4096 |
| **`V->S`; `GetValue`; `S->V`** | **0 / 4096** |

**The scalar unit is a pipe.** It needs a flag like any other cross-pipe dependency, and
**both halves are load-bearing** -- the second orders the scalar result against the Vec op
that consumes it. The pinned library does exactly this in
`pto/npu/a2a3/TRowExpand.hpp:38-43`:

```cpp
set_flag(PIPE_V, PIPE_S, ev);  wait_flag(PIPE_V, PIPE_S, ev);
T tempValue = (T)(*(srcPtr + i * srcStride));      // the scalar read
set_flag(PIPE_S, PIPE_V, ev);  wait_flag(PIPE_S, PIPE_V, ev);
vector_dup(...);                                    // the Vec op consuming it
```

A generator that followed C19 literally got a kernel that was **wrong AND paid two extra
DMAs**. Using the sandwich instead removed a full H-wide `TROWEXPAND` pass per row.

This is the same shape as the C27 correction: the symptom was reported accurately and the
mechanism was invented. **Note the half-sandwich row** -- one flag makes it a latent race
that moves with unrelated codegen, which is exactly how such a bug survives review.

The original (incorrect) rationale, retained so the failure mode is recognisable: Vec ops
were believed to write to pipeline registers rather than the tile buffer, with
`pipe_barrier(PIPE_V)` ordering execution but not flushing.

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

**C1x. An `event_t` variable must NOT be `const` or `constexpr` qualified.**
`set_flag` / `wait_flag` are CCE builtins whose 3rd parameter check rejects a
cv-qualified type. Naming the type and adding `const` is the trap, because it is
the natural thing to write:

```cpp
const event_t     ev = EVENT_ID1;   // ERROR: "the 3rd parameter must be a type 'event_t'"
constexpr event_t ev = EVENT_ID1;   // ERROR: same

auto     ev = EVENT_ID1;            // OK -- deduces plain (non-const) event_t
event_t  ev = EVENT_ID1;            // OK
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);          // OK -- literal at the call site
wait_flag(PIPE_MTE3, PIPE_MTE2, (event_t)eid);   // OK -- cast of a computed id
```

All six forms above were compile-tested against `bisheng -xcce`
`--cce-aicore-arch=dav-c220`; only the two `const`/`constexpr` ones fail. `auto`
works in a loop, through a ternary, as a function parameter, and in a template.

Three separate runs reported this as "`auto _we = EVENT_ID1;` does not compile"
and repaired it by inlining the literal at every call site. **That diagnosis was
wrong** — `auto` is fine, and inlining is an unnecessary readability cost that
also makes a computed id impossible. The offending form was `const event_t`.
If you hit this error, delete the `const`; do not un-name the variable.

**C33. A BOXED tile's valid extent is honoured only within its LAST 16-row fractal.**
For `Mat` / `Left` / `Right` / `Acc` tiles, declaring `Rows` larger than the rows you
actually use and passing a smaller runtime valid extent produces **silently wrong
numbers** — no fault, no compile error, no out-of-bounds write.

The condition is exact and predictive:

> a boxed tile with declared `Rows` and valid extent `V` is correct **iff
> `ceil(V/16) == ceil(Rows/16)`**.

**C33-USE: THE RULE IS PERMISSIVE, AND READING IT CONSERVATIVELY COSTS 3-4x.** The condition says
`V` may sit **anywhere inside the last 16-row fractal** -- it does NOT say `V` must equal `Rows`.
So for `Cout = 127`, declare `Rows = align16(127) = 128`: `ceil(127/16) = 8 = ceil(128/16)`, one
tile, correct. Splitting 127 into four 32-row tiles to "stay safe" is the conservative misreading,
and it is what `conv_2d` did. Measured cost of the misreading, same op, same data:

| | conservative split | `align16(rem)`, one tile |
|---|---|---|
| case 8 GEMM | 645 us | **339 us** |
| case 7 GEMM | 248 us | **113 us** |

**So: size the last M tile as `align16(remainder)` and pass the true valid extent.** Check the
condition, do not avoid the situation.

Probed on A2/dav-c220 (`isa_probes/probe_boxvalid.cpp`), `C = A[0:V] @ B`, sweeping
**every** `V` in `1..Rows` at five declared sizes:

| declared `Rows` | `V` that are CORRECT | everything else |
|---|---|---|
| 16 | 1-16 | — |
| 32 | 17-32 | wrong |
| 64 | 49-64 | wrong |
| 96 | 81-96 | wrong |
| 128 | 113-128 | wrong (worst rel err **1.30**) |

Note what this is **not**: `V = 16, 32, 64` are all *wrong* at `Rows = 128`, so it is
not a "must be a multiple of 16" rule. Only the last fractal may be partial. Nothing
is ever written outside the valid box, so a bounds check will not catch it.

**The fix is to declare the tile at the size you use.** `Rows = 16` makes `V = 1..16`
all correct. For a runtime row count, pick the tile shape per case (template over a
small set of `Rows`, or tile the rows and let only the final tile be partial) rather
than declaring one big tile and narrowing it.

**COOK-§8.9's `TileAcc<float, M, N, DYNAMIC, DYNAMIC>` idiom invites the broken
form** — it is safe only when the runtime extent lands in the last fractal. This cost
one run 3 of its 4 repair attempts.

**C33-USE-COL: THE SAME LAST-FRACTAL RULE GOVERNS THE *COLUMN* EXTENT -- AND THERE
`CompactMode::Normal` DOES FIX IT.** Everything above is about the row extent `V`, where
`CompactMode` is no help (re-probed and reproduced exactly on `grouped_matmul`: at
`Rows = 128` only `V = 113..128` are correct at 4.8e-8, every other `V` is wrong by ~4.3e2,
and `align16(V)` is correct at every `V`). The column extent behaves the same way by
default and **unlike the row extent it has a fix**:

| tile | declared | valid extents that are CORRECT |
|---|---|---|
| plain `NT = 256` | — | only `nv = 255, 256` (everything else wrong, err ~1.3) |
| **`CompactMode::Normal`, `NT = 256`** | — | **every `nv` from 16 up, exact (4.7e-8)** |

The same holds for the **K extent of the `Left` operand**. So: for a ragged column or K
extent, declare the tile `CompactMode::Normal` and pass the true extent -- do NOT carry the
row-extent workaround (`align16` + per-case tile sizes) across to the columns, and do NOT
assume `CompactMode` rescues the rows. Probed on A2/dav-c220,
`skillyard-runs-v112/grouped_matmul/probes/run_shapes.py`.

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

# ARCH: pick per stage archetype -- see the table below. Vector-only stages MUST use
# dav-c220-vec; using dav-c220 on them costs a flat ~2.86 us per call (C33b).
ARCH=dav-c220-vec      # vector-only stage (no Cube op in the StageSpec)
# ARCH=dav-c220-cube   # CUBE-ONLY stage (no Vec op in the StageSpec) -- see C33c
# ARCH=dav-c220        # genuine Cube+Vector stage, and ONLY that

"$ASCEND_HOME_PATH/bin/bisheng" -fPIC -shared -xcce -DMEMORY_BASE -O2 \
  -std=gnu++17 --cce-aicore-arch=$ARCH \
  -Wno-macro-redefined -Wno-ignored-attributes \
  -I<kernel_dir> -I<example>/include \
  -I<pto_isa_root> -I<pto_isa_root>/include \
  -I"$ASCEND_HOME_PATH/include" \
  -I"$ASCEND_HOME_PATH/pkg_inc" \
  -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
  -I"$ASCEND_HOME_PATH/pkg_inc/profiling" \
  kernel.cpp -o kernel.so
```

**THE ARCH FLAG DEPENDS ON THE STAGE ARCHETYPE. Pick it with C33b, not from this recipe.**

| stage archetype | flag | launches as |
|---|---|---|
| **vector-only** (no Cube op in the StageSpec) | `--cce-aicore-arch=dav-c220-vec` | `AI_CORE` |
| genuine Cube+Vector | `--cce-aicore-arch=dav-c220` | `MIX_AIC` |

Verified by `-dM`: `dav-c220` defines `__CCE_AICORE_ENABLE_MIX__` and compiles a CUBE pass;
`dav-c220-vec` does not. **Using `dav-c220` on a vector-only stage costs a flat ~2.86 us of
device-side dead time per call** (C33b), which is invisible at production scale and decisive
below ~10 us of device work. Earlier versions of this rule hardcoded `dav-c220` and four
independent runs had to override it.

Changing the flag alone is NOT sufficient and NOT safe -- the work partitioning must match
the geometry. See C33b for the partition rule.

Key constraints:
- Use the active CANN toolkit (default 9.0.0 via the `/usr/local/Ascend/cann` symlink); resolve `bisheng` and includes from `$ASCEND_HOME_PATH`, never a hardcoded version path
- `-xcce` (CCE language mode, NOT `-x cce` with space)
- `--cce-aicore-arch=dav-c220` or `dav-c220-vec` per the table above — both auto-define `__CCE_AICORE__` and `__DAV_C220_VEC__`; only `dav-c220` also defines `__CCE_AICORE_ENABLE_MIX__`
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

**C27 (CORRECTED). After a `TLOAD`, SYNC `MTE2 -> V` before any Vec op reads the
tile. Do NOT "push" with `TMULS(x, x, 1.0f)` — that was a misdiagnosis.**

```cpp
// CORRECT, and sufficient:
TLOAD(k_tile, ...);
TLOAD(gate_tile, ...);
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);      // <-- THIS is the requirement
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
TEXP(gate_tile, gate_tile);
pipe_barrier(PIPE_V);
TMUL(k_tile, k_tile, gate_tile);             // no push needed
```

This rule previously claimed `TMUL`/`TADD`/`TEXP`/`TSEL`/`TROWSUM`/`TCOLSUM` read
their sources from the Vec *pipeline* while `TLOAD` writes only the *buffer*, and
mandated a `TMULS(x, x, 1.0f)` push on every `TLOAD`ed operand. **Probed and
falsified** (`isa_probes/probe_c27.cpp`, 20 repeats per cell, exact CPU reference):

| configuration | WITH the push | WITHOUT the push |
|---|---|---|
| `TLOAD` -> op, proper MTE2->V sync (`TMAX/TADD/TMUL/TSUB/TMIN`) | 20/20 exact | **20/20 exact** |
| C27's own shape: `TLOAD a,b` -> `TEXP(b)` -> `op(a,b)`, with sync | 20/20 exact | **20/20 exact** |
| same shape, **MTE2->V sync REMOVED** | **WRONG 20/20** | WRONG 4/20 |

Two things follow, and the second is why this matters:

1. **The push is pure overhead when the sync is present.** A flash_attention run
   measured removing it as bit-identical and 1.011x faster; that reproduces here as
   exactness at every op tested.
2. **The push does not substitute for the sync — it makes the failure WORSE.**
   Without the handshake the pushed form is wrong on *every* run while the unpushed
   form is wrong on 4 of 20. `TMULS` reads the tile before MTE2 has landed it and
   commits that garbage into the pipeline deterministically; without it the binary
   op sometimes wins the race. So a generator that dutifully adds the push and omits
   the handshake gets a kernel that is *always* wrong, produced by following the
   rule.

The original observation was real — a Vec op reading a `TLOAD`ed tile did return
garbage — but the cause was the missing `MTE2 -> V` handshake, and `TMULS` +
`pipe_barrier` merely perturbed the timing enough to hide it sometimes.
→ PLAT-§Pipeline

**C27-escape: `TAXPY` reads BOTH operands from the BUFFER.** `TAXPY(dst, src, a)`
computes `dst += a * src` reading `dst` and `src` from the tile buffer, so it does
NOT need the push. A broadcast-add prologue built from `TLOAD`ed operands can
therefore be written as `TAXPY(acc, addend, 1.0f)` with no `TMULS(x,x,1.0f)`
pushes and no intervening `pipe_barrier(PIPE_V)` per operand. Measured on a
masked-softmax kernel: a 5-op push-and-add prologue collapsed to 2 ops, worth
~4% end-to-end. Prefer `TAXPY` whenever the operation is `dst += a*src` and the
operands come from `TLOAD`. → COOK-§6.5

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
requires `src1` as a ColMajor `[N,1]` tile for Mode 1.

> **CORRECTION -- Mode 2 does NOT zero anything.** This rule said a RowMajor `[N,8]` `src1`
> selects Mode 2 and "ZEROES the result outside column 0". Probed: Mode 2 validated
> **216/216 exact**. The backend (`a2a3/TRowExpandAdd.hpp`) issues
> `vmul(dst, src0, src1, repeats, 1, 1, 0, 8, 8, 0)` -- the **`src1BlockStride = 0`**
> re-reads the same 32-byte block, so Mode 2 computes
> **`dst[i,j] = src0[i,j] * src1[i, j mod 8]`**: a broadcast within the 8-element block, not
> a zeroing. The original report's `src1` simply held the scalar in column 0 and **zeros
> elsewhere**, so the zeros in the output were the operand's contents, not the instruction's
> behaviour. Symptom attributed to the wrong cause -- the same failure as C19 and C27.
>
> Mode 1 still ships here: it measured **~1.28x faster** than Mode 2. Prefer it for speed,
> not for correctness. And if you feed Mode 2 deliberately, remember its real semantics --
> `j mod 8` broadcasting is only what you want when `src1`'s block genuinely repeats.

When the
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
  ~~EXCEPTION (wide free axis): `TCOLSUM` silently truncates past 64 columns.~~
  **FALSIFIED -- REMOVED.** `isa_probes/probe_colsum.cpp`, R=16 summed to `[1,C]`,
  with columns >= 64 offset by +100 so truncation could not hide:

  | C | 32 | 64 | 96 | 128 | 256 | 512 |
  |---|---|---|---|---|---|---|
  | max abs err | 0 | 0 | 0 | **0** | **0** | **0** |

  Bit-exact at every width. `TCOLSUM` is correct for any free axis; do NOT reorient
  the state to work around a limit that does not exist. (COOK-§10.5's `TROWSUM`
  orientation remains a valid *performance* choice, but it is not required for
  correctness.)

  **This was the THIRD appearance of a false "64 lanes" restriction.** C15 claimed the
  same for `TROWMAX`/`TROWSUM` -- refuted, and the symptom was C15's own under-sized
  scratch. If you meet another "> 64 lanes is broken" claim in this cookbook, probe it
  before designing around it: the real rule is that reduction scratch scales with the
  SOURCE width (C15), and an under-sized scratch produces exactly the truncation
  symptom these rules kept mis-attributing to a lane limit.
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

## C35: AN ALIASED DESTINATION CAN BE SILENTLY WRONG -- TRECIP IS

On A2/A3 (`dav-c220`), `TRECIP(dst, src)` **returns 1.0 for every input when `dst` aliases
`src`**. There is no compile error, no `PTO_ASSERT`, and no runtime signal. With a distinct
destination it is exact (probed at 0.5 / 1 / 2 / 4 / 10 / 1.368).

`TEXP`, `TMULS` and `TADDS` are all correct in place. **The restriction is specific to
TRECIP**, so "the other unary ops alias fine" is not evidence that this one does.

```cpp
TRECIP(w, w);   // WRONG: silently yields 1.0
TRECIP(o, w);   // correct; budget UB for a separate destination
```

**MECHANISM, found later and it explains the 1.0 exactly.** `TRECIP` is **not** a `vrec`
intrinsic. It expands to `TDIVS(dst, 1, src)`, which is `vector_dup(dst, 1.0)` + a barrier +
`vdiv(dst, dst, src)`. When `dst` aliases `src`, **the dup overwrites the divisor with 1.0
before the divide reads it**, so you compute `1.0 / 1.0` for every lane. That is why the wrong
answer is exactly 1.0 rather than garbage, and it predicts the rule: *any* PTO op that
internally seeds its destination before reading its source will fail the same way on an aliased
destination. Check the expansion before assuming an op is a single instruction.

Corollary, measured: **substituting a raw `vrec` for `TRECIP` is not a free speedup** -- it
measured MERE 1.10e-3 and failed 10 of 26 accuracy cases. `TDIVS` is the more accurate
expansion, and that accuracy is why it is written that way.

**Failure signature:** the kernel returns its *input unchanged*, which reads as "the kernel
never ran" or "the launch is broken" rather than "one instruction is wrong". If a vector
chain emits its input, suspect an aliased TRECIP before you suspect the launch.

**Generalisation:** for any unary tile op, do not assume in-place is legal. If a chain can
be written with a distinct destination at no UB cost, prefer that; if it cannot, probe the
aliased form against a CPU reference on a handful of scalars before building on it.

## C36: THE TUNARY FAMILY REJECTS BFLOAT16 -- ROUND-TRIP THROUGH TCVT

`TUNARY_IMPL` (the implementation behind `TEXP`, `TRECIP`, and the rest of the unary
elementwise family) `static_assert`s on **float32 or half only**. `bfloat16_t` is rejected
outright at compile time.

A bf16 elementwise kernel must therefore stage through fp32:

```cpp
TCVT(w_f32, src_bf16, RoundMode::CAST_NONE);   // widen  (exact)
// ... arithmetic in fp32 ...
TCVT(dst_bf16, o_f32, RoundMode::CAST_RINT);   // narrow (round to nearest even)
```

Both directions exist on A2/A3 (`bf16 <-> fp32`). This is a **correctness constraint, not a
precision preference** -- budget the extra fp32 tiles into the UB plan at Phase 4, not after
the first compile failure. fp16 usually wants the same treatment for accuracy reasons even
though it is accepted.

## C37: IN A VEC-ONLY BUILD THERE IS NO SECOND SUB-BLOCK

Under `--cce-aicore-arch=dav-c220-vec`, `get_subblockid()` **always returns 0** and
`get_block_num()` equals the launch `block_dim` (probed at block_dim 1 / 8 / 48). The
two-AIV-sub-block model belongs to MIX builds.

Consequently a grid-stride scheme of the form

```cpp
const int64_t lane   = get_block_idx() * 2 + get_subblockid();   // WRONG in a vec build
const int64_t nlanes = get_block_num() * 2;
```

yields only **even** lanes, so half of every multi-tile problem is never written. It passes
any test whose input fits a single tile and fails everything larger -- a size-dependent
correctness bug that a small-shape unit test cannot catch.

Index work by `get_block_idx()` / `get_block_num()` alone in vec-only builds. `EX-2`'s
`if (vid != 0) return;` is a harmless no-op here and is **required** in MIX builds.
Validate any lane-indexing change at a size spanning at least three tiles (see C33b/C33c for
when each arch flag applies).

## C38: `ValidCol` GIVES A BYTE-EXACT STORE -- RAGGED TAILS ARE SAFE

`TSTORE` on the ND path computes `lenBurst = validCol * sizeof(DType)` **in bytes**, so a
tile whose `ValidCol` is not a multiple of the 32-byte block still writes exactly
`ValidCol` elements and nothing past them. Verified with a guard-padded output buffer at
`N % 8` = 1, 3, 4, 5, 6, 7 including a prime `N = 1000003`: no overrun.

This separates two constraints that are easy to conflate:

* **`Cols`** (the tile's allocated width) must satisfy the 32-byte alignment rule --
  fp32 `Cols % 8 == 0`, fp16/bf16 `Cols % 16 == 0` (PLAT-Align, COOK alignment section).
* **`ValidCol`** (the runtime extent) is unconstrained.

**C38a -- THE GUARANTEE IS `TSTORE`-ONLY. `TLOAD` WRITES PAST `ValidCol`.** C38 was probed on
stores and its conclusion is correct for stores. The load direction is **not** symmetric: the
UB-side DMA is 32-byte granular, so a load of `validCol` elements deposits
`ceil(validCol*sizeof(T)/32)*32` bytes -- an overrun of `(-validCol) mod (32/sizeof(T))`
elements, filled with the **next real data from GM**, not garbage.

Why stores are safe and loads are not: a store rounding up would corrupt GM, so the hardware
masks the tail; a load has nothing to protect in UB, so it does not. Same 32-byte granularity
we already probed from the other side in `conv_2d` ("the UB-side address advances by
`ceil(lenBurst/32)*32 + ubGap*32` bytes per burst ... `rightPadding` does NOT change it").
`TLoad.hpp:77` passes `lenBurst = validCol * sizeof(DType)` in bytes exactly as the store does
-- the rounding is in the instruction, not the wrapper, which is why reading the wrapper does
not warn you.

**This is a silent wrong answer whenever anything downstream consumes the tile by `Cols`
rather than by `ValidCol`** -- a reduction being the obvious case. On `softmax` it took out
exactly the two fp32 cases with a ragged last axis: `D = 4097` and `D = 8193`, both
`D % 8 == 1`, so 7 extra elements entered the sum. The error signature pins it with no
ambiguity: MERE **1.711e-03** against **7/4097 = 1.709e-03**. Every 16-bit case passed at the
same time, because there the overrun landed in a staging tile that `TCVT` re-read by
`ValidCol` -- so a 16-bit-heavy sweep ships this bug.

**Rule: after a `TLOAD` with a ragged `ValidCol`, either overwrite the tail with the
operation's identity (0 for a sum, `-inf` for a max) before consuming the tile, or make every
consumer honour `ValidCol`.** Do not assume the tail is zero and do not assume it is garbage:
it is the neighbouring data, which is exactly the kind of wrong that looks plausible.

*Provenance: 13/13 exact on the overrun formula in the `softmax` run's own probe; I confirmed
the mechanism from `TLoad.hpp` and our prior `conv_2d` DMA probe and verified the 7/4097 error
signature arithmetically, but did not re-run the overrun probe myself.*

So the correct pattern for a dynamic length is a fixed, aligned `Cols` with the ragged tail
carried by a runtime `ValidCol` -- **not** a narrower tile, and **not** an overlapping
backward-shifted final tile (which breaks the *start* alignment instead).


## C39: A PRECISION TEMPLATE ARGUMENT CAN BE SILENTLY IGNORED

Several PTO instructions take an `auto PrecisionType` template parameter (`RsqrtAlgorithm`,
`ExpAlgorithm`, `RecipAlgorithm`, `LogAlgorithm`, ... each `{DEFAULT, HIGH_PRECISION}`).
**Requesting `HIGH_PRECISION` does not guarantee you get it.** Where the instruction also has a
**tmp-tile overload**, the high-precision path lives in the tmp overload only, and the
no-tmp form accepts the tag and discards it.

Measured on dav-c220, fp32, input in [0.1, 10], against the fp32 threshold 2^-13 = 1.22e-4:

| call | max rel err | |
|---|---|---|
| `TRSQRT(b, a)` | 3.047e-03 | FAIL |
| `TRSQRT<RsqrtAlgorithm::HIGH_PRECISION>(b, a)` | **3.047e-03 -- identical** | FAIL |
| `TRSQRT<RsqrtAlgorithm::HIGH_PRECISION>(b, a, tmp)` | **1.202e-07** | PASS |
| `TSQRT(t,a)` then `TDIV(b, one, t)` | 1.202e-07 | PASS |

In `pto_instr.hpp` TRSQRT declares two overloads, `(dst, src)` and `(dst, src, tmp)`; only the
second reaches `TRsqrtHighPrecision`. The first routes to `TUNARY_IMPL<RsqrtOp<...>>`, which
never sees the tag. `TROWEXPANDDIV` has the same no-tmp / tmp pair and the same exposure.

**Rules:**
1. **Never infer accuracy from the enum.** Probe the exact call you intend to emit against a
   CPU float64 reference, at the dtype threshold you must meet.
2. If an instruction has both a precision parameter and a tmp overload, **pass the tmp** when
   you want precision, and budget the tile.
3. `TRSQRT` at default precision is ~25x over the fp32 threshold **on its own**, before any
   other error source. Every fp32 normalization scale (RMSNorm, LayerNorm, any
   quantization scale) must use the 3-arg form or `TSQRT`+`TDIV`. They measure identically,
   so choose on UB and op count, not accuracy.

## C40: **RETRACTED.** THE fp32 `Cols >= 2048` ROW-REDUCE CLIFF DOES NOT EXIST -- IT WAS C49

C40 used to say: a multi-row fp32 Vec tile with `Cols >= 2048` puts the row stride beyond the
255-block repeat-stride field and the row reduction silently returns wrong values; therefore
cap chunk width at 1024 elements for any fp32 tile feeding a row reduce, and lock the cap in
`shape_contract.json`.

**Every part of that is wrong except the symptom, and the cap was costing real width.**
Re-probed on `rms_norm` over a 137-point sweep varying tmp rows, src `Cols`, tmp `Cols` and
`validCol` independently (rebuilt and re-run independently of the run that reported it):

| tmp rows | src `Cols` | tmp `Cols` | `validCol` | max rel err | |
|---|---|---|---|---|---|
| 4 | 2048 | 1024 | 2048 | **1.085e-07** | exact -- C40 predicts silent corruption |
| 2 | 4096 | 2048 | 4096 | **4.032e-08** | exact, row stride 512 blocks = 2x C40's claimed limit |
| 1 | 8192 | 4096 | 8192 | **5.119e-08** | exact |
| 8 | 512 | **64** | 256 | **2.794e-02** | WRONG at `Cols = 512`, nowhere near 2048 |

The threshold is not 2048 and it is not a property of the SRC tile at all. **The first failure
in fp32 is at `validCol = 256` with a 64-wide `tmp`**, and it moves with the tmp width exactly
as C49a's formula predicts -- 0 mispredictions in 117 points. C40's own evidence is consistent
with this: it noted the reduction output was wrong while the elementwise output from the same
tile stayed bit-exact, which is precisely what an undersized *scratch* tile does and precisely
what a *source* addressing limit would not do.

**What to do instead:** size the `tmp` tile by C49a and let the src tile be as wide as the UB
budget allows. Do NOT cap fp32 row-reduce tiles at 1024 -- on this benchmark that cap would
have halved the tile width on every fp32 case including `D = 8192`, for nothing.

**What survives, and it is the useful half:** *when a reduction output is wrong but an
elementwise output from the same tile is bit-exact, the fault is in the reduction's scratch,
not in the arithmetic and not in the source tile.* That is the diagnostic; C40 just named the
wrong culprit. Same mis-attribution shape as C15.

Related skew from the original C40 run, still valid and independently useful: `TMADD` and
`TMULA` are served by the npu-coding MCP but have **no a2a3 implementation in pto-isa
`109c9f72`**, and there is **no `TCVT` path for bf16 -> int8** (stage through fp32). Verify
against the headers you compile against, not the index.

## C65: PTO HAS HARDWARE SORT PRIMITIVES. AN ITERATIVE top-k IS O(k) PASSES AND IT SHOWS

`pto-isa` at the pinned commit ships `a2a3/TSort32.hpp` (`TSORT32`, lowering to **`vbitsort`**)
and `a2a3/TMrgSort.hpp` (`TMRGSORT`, lowering to **`vmrgsort4`** with `get_vms4_sr()`), and this
skill mentioned **neither** until now. The sort/select family is the archetype most likely to be
hand-rolled badly, because the naive form is easy to write and its cost is invisible at small `k`.

**The failure signature to look for in your own per-case table: time that scales with `k`.**
On `level3/moe_gating_top_k_softmax` the previous attempt (63.69) ran an iterative
find-max-then-mask, i.e. `k` passes over `E` experts, and the per-case ratio against the baseline
tracks `k` almost perfectly:

| k | 1 | 8 | 16 | 64 | 128 |
|---|---|---|---|---|---|
| ours / baseline | **1.17x** | 0.97x | 0.10x | 0.10x | **0.06x** |

At `k = 1` it wins. At `k = 128` it is 17x slower than a baseline that is itself a clamped proxy.
A selection algorithm whose cost is `O(E log E)` once -- or a partial/bitonic selection -- does
not have that shape.

**`TSORT32` (`TSort32.hpp:144-153`):**

| requirement | value |
|---|---|
| dst / src DType | `half` or `float` (and they must match) |
| index DType | **`uint32_t`**, no other |
| all tiles `Loc` | `TileType::Vec` |
| layout | all three row-major |

It sorts in blocks of 32 and carries the indices with the values, which is exactly what a top-k
needs -- you want `(value, original_index)` pairs, and the op's second output is the index.

**`TMRGSORT` (`TMrgSort.hpp:172-201`) merges up to FOUR sorted runs:**

| requirement | value |
|---|---|
| dst DType | `half` or `float`; tmp must match |
| Rows | **every tile must be `ONE_ROW`** |
| UB | `tmpSize + Src0::Cols*elemSize <= UB_SIZE`, and each further src `Cols*elemSize <= UB_SIZE` -- **`static_assert`ed, so an oversized merge is a compile error, not a silent truncation** |

So the standard shape is `TSORT32` to make 32-element sorted runs, then a `TMRGSORT` tree (4 runs
at a time) to merge up to the width you need, then take the first `k`. Budget the tmp tile: the
`static_assert` will tell you if you got it wrong, which is a kindness this library does not
always extend.

**Rule: for any top-k, arg-max-of-k, sort, median or selection stage, PRICE THE HARDWARE SORT
against your iterative form before shipping the iterative one, and put the `k`-scaling in the
report.** If your time grows with `k`, say so and say why you kept it.

## C64: PTO HAS A HARDWARE img2col PATH FOR CONVOLUTION. CHECK IT BEFORE HAND-ROLLING ONE

`pto-isa` at the pinned commit ships a first-class LOAD3D/img2col route and **the skill did not
mention it at all** until now -- no reference to `TIMG2COL`, `ConvTile`, `SetFmatrix`, `NC1HWC0`
or `LOAD3D` anywhere in this file or its `references/`. A conv-family run that does not at least
PRICE this path is leaving the hardware's own convolution feed unused.

The headers are `a2a3/TImg2col.hpp`, `a2a3/SetFmatrix.hpp`, `a2a3/SetImg2colPadding.hpp`,
`a2a3/SetImg2colRpt.hpp`, with `ConvTile` in `common/pto_tile.hpp`. `TIMG2COL` applies stride,
padding and dilation **in hardware** while feeding L0A.

**Its constraints, read from the library's own `static_assert`s (`TImg2col.hpp:65-73`):**

| requirement | value |
|---|---|
| source tile `Loc` | `TileType::Mat` (L1) |
| source `layout` | `NC1HWC0` **or** `NDC1HWC0` |
| destination `Loc` | `TileType::Left` -- **L0A only, never L0B** |
| destination fractal | `SLayout::RowMajor` and `isRowMajor` |
| dtypes | `int8_t`, `half`, ... (src and dst DType must match) |

**The catch, and it is the whole decision.** `TLOAD` into a `ConvTile` accepts only
`NC1HWC0 -> NC1HWC0`, `FRACTAL_Z -> FRACTAL_Z`, `FRACTAL_Z_3D`, or `NDC1HWC0` (`TLoad.hpp:665`).
So **the hardware path requires the feature map to already be 5HD in GM**. If the benchmark hands
you NCHW you must convert first, and PTO gives you the converters as Vec tile ops in
`a2a3/TTrans.hpp` (`TTransConvNCHW2NC1HWC0`, `TTransConvNC1HWC02C1HWNC0`, and the grouped
variants). That conversion is not free -- it is exactly the "layout tax" the vendor pays.

**So the rule is: price BOTH routes before committing, and say which you chose and why.**
1. hardware img2col: layout conversion + `TIMG2COL` + `TMATMUL`;
2. a channel-major formulation that keeps the activation in its given layout and expresses the
   convolution as `Kh*Kw` ordinary GEMMs with plain `ND2NZ` `TLOAD`s -- no activation transpose
   at any point.

Route 2 avoids the layout tax entirely but does more GEMM calls; route 1 gets hardware
stride/pad/dilation but pays for 5HD. Which wins is a measurement, not a principle, and it will
depend on how much of the operator's time the conversion actually is. **Measure the vendor's own
kernel list first** (`Conv2D` alongside `TransData`/`Cast` tells you what the layout tax costs
*it*), then price your own conversion at the same shapes.

**And note what `t_hw` is for the conv family: a COMPUTE roofline, not a memory one.** Fitted
across all 20 `level3/conv_2d` cases with **zero** error: `t_hw = FLOPs / 354e12` for 16-bit and
`FLOPs / 86e12` for fp32, floored at 1.0 us. So **HAP 1.0 is Cube peak** and HAP 0.5 is whatever
the vendor achieves -- on that benchmark the real baselines sit at 1.2x-2.8x of peak, i.e. the
vendor runs at 36-83% of the Cube. Budget against Cube utilisation, not bandwidth.

## C63: `TCVT` WITH THE SAME SRC AND DST ELEMENT TYPE CORRUPTS THE DESTINATION

`TCVT(dst, src, CAST_NONE)` where `DstTile::DType == SrcTile::DType` does **not** degrade to a
copy and does **not** leave `dst` alone. It writes, and what it writes is neither the source
nor the previous contents. Probed on A2/dav-c220-vec, fp32, a `[1, 512]` tile poisoned with
`-777` first:

| call | untouched (still -777) | equals src | first 3 outputs |
|---|---|---|---|
| `TCVT(fp32 <- fp32)` | **0 / 512** | **0 / 512** | `[-1.0, -1.0, 0.0]` |
| `TMULS(dst, src, 1.0f)` (control) | 0 / 512 | **512 / 512** | exact |
| `TCVT` fp32 -> half -> fp32 (control) | 0 / 512 | 0 / 512 | correct round-trip |

Both controls behave, so this is `TCVT`'s identity case specifically, not the probe.

**Rule: never emit a `TCVT` whose src and dst element types are the same.** This bites when
the dtype is a template parameter and one instantiation happens to collapse -- an
`if constexpr (!std::is_same_v<Src, Dst>)` guard around the conversion, with a `TMULS(dst, src,
1)` or a plain tile copy on the else branch, is the fix. It is a **silent** wrong answer: no
compile error, no assert.

Note how it was found and why it nearly was not: it made the fp32 `D = 1` and `D = 2` paths
wrong (MERE 7.2e-01) on an operator where **no scored case reaches that path**. The 20-case
accuracy gate was green. Only an out-of-contract stress sweep exposed it -- which is the
argument for running one (C61a).

Related skew, same run: `TMADD` and `TMULA` are served by the npu-coding MCP but have **no
a2a3 implementation in pto-isa `109c9f72`**, and there is **no `TCVT` path for bf16 -> int8**
(stage through fp32). Verify against the headers you compile against, not the index.

## C41: `pipe_barrier(PIPE_X)` ORDERS PIPE X ONLY -- CROSS-PIPE WAR NEEDS A FLAG

`pipe_barrier(PIPE_V)` orders the vector pipe against itself. It does **not** hold MTE2 back.
So a loop that re-`TLOAD`s into a tile the vector pipe is still reading has a
write-after-read hazard that no amount of `pipe_barrier(PIPE_V)` closes:

```cpp
for (c = 0; c < nchunks; ++c) {
    // WRONG if MTE2 can run ahead: the next TLOAD may land while V still reads t
    TLOAD(t, chunk(c));
    ... wait MTE2->V ...
    TMAX(acc, acc, t);
    pipe_barrier(PIPE_V);          // orders V only -- does NOT protect t from MTE2
}
```

The fix is an explicit `V -> MTE2` flag: `set_flag(PIPE_V, PIPE_MTE2, ev)` once the vector work
has consumed the tile, and `wait_flag(PIPE_V, PIPE_MTE2, ev)` before the next `TLOAD` into it.
(Equivalently, give the loop two tile slots.)

**Observed failure mode** (v0.99 run on `grouped_matmul_swiglu_quant`): a row max silently
reduced over **one chunk instead of all of them** -- 245 of 1024 rows wrong on one case, and
*which* chunk survived varied run to run. Nothing faults; the number is just wrong and
non-deterministic.

**AMENDMENT -- the prescribed remedy is not always sufficient.** A third run
(`dynamic_quant`) reports that at a **task seam**, where a Vec READ of a UB buffer must be
ordered against a later MTE2 WRITE to it, the textbook `set_flag/wait_flag(PIPE_V, PIPE_MTE2)`
**did not hold**:

| guard at the seam | first-launch corruption |
|---|---|
| `set_flag/wait_flag(PIPE_V, PIPE_MTE2, id)` | **6 of 6 fresh processes** |
| + entry-time normalisation of every event id the kernel holds | 4 of 8 |
| `set_flag/wait_flag(PIPE_MTE3, PIPE_MTE2, id)` chained behind the store's `V->MTE3` | **0 of 18** |
| `pipe_barrier(PIPE_ALL)` | **0 of 18** |

Read this carefully, because it is narrower than "the flag does not work":

* The **same kernel uses `V -> MTE2` flags successfully elsewhere** -- the failure is specific
  to the read-then-overwrite at the seam, not to the pipe pair.
* The **`MTE3 -> MTE2` row is a positive control**: cross-pipe flags *are* honoured, so this is
  not a blanket flag failure.
* The **entry-normalisation row is a falsification**: it rules out the obvious "stale event
  register" explanation.
* The bug is **invisible to a contract sweep, invisible to same-process determinism, and
  showed on only 1 run in 3 to 1 in 20** -- it took the C44 out-of-contract probe plus a
  **fresh-process** soak to find. A determinism check that does not start a new process will
  not see it.

**Status: reported by that run, not independently reproduced here** (the racy variant was
removed rather than kept behind a knob, so it could not be cheaply rebuilt). What *was*
verified in the parent session: the shipped kernel rebuilds byte-identically and shows
**0 of 12 fresh-process corruptions** on the exact geometry that flagged the bug.

**Practical rule until someone reproduces the racy arm:** at a task seam where a Vec read is
followed by an MTE2 overwrite of the same buffer, do not rely on `V -> MTE2` alone. Use
`pipe_barrier(PIPE_ALL)`, or chain `MTE3 -> MTE2` behind the store's `V -> MTE3`. The measured
cost of the drain on that op was ~6% (1.335 racy vs 1.255 safe) -- cheap for the class of bug
it removes. And **soak in fresh processes**, not just repeated launches.

**Second independent instance** (`grouped_matmul_swiglu_quant`, a different kernel and a
different pipe pair): a `y_scale` **MTE3 store** read a tile that the next group's **MTE2 load**
clobbered. Case 15 returned another unit's `x_scale` as `y_scale` on **224 of 1009 rows**,
racily, **while `y` stayed correct**. Same signature as the first instance: one output wrong,
another from the same region exact, and non-deterministic. Treat every reuse of a tile across a
pipe boundary as needing an explicit flag, in both directions -- the hazard is not specific to
MTE2-overwriting-V.

**Provenance note, stated honestly:** the general rule above is ordinary cross-pipe WAR
semantics and is not in doubt. The specific failure mode was observed by that run; a parent-session
attempt to reproduce it in isolation returned 12/12 correct for *both* variants, because the
probe placed an adjacent `set_flag`/`wait_flag(MTE2 -> V)` straight after the `TLOAD`, which
serialises the two pipes so completely that MTE2 can never run ahead. **That is a lesson about
probe design, not evidence against the rule:** a probe for a run-ahead hazard must leave the
producing pipe free to run ahead, or it tests nothing. Use >=2 slots and defer the consumer's
wait (see optimizer 3.12 for the companion discipline of proving a lever is live).

## C42: AN OVERLOAD CAN CHANGE THE REQUIRED OPERAND LAYOUT, SILENTLY

C39 established that an extra tmp argument can change which *precision* path runs. The same
mechanism changes required **layouts**, with the same silence.

Reported by the v0.99 run, found by probing: **`TQUANT`'s 4-argument overload (with an explicit
scratch tile) requires a ColMajor per-row scale.** Handed a RowMajor lane-filled `[R, 8]` scale
it is silently wrong -- max int8 difference 126 on 42% of elements. The 3-argument overload is
bit-exact on the same inputs.

So the rule from C39 generalises: **when an instruction has several overloads, the extra
argument can change the contract on the arguments you already had.** Do not assume an overload
is a superset. Probe the exact arity and layout you intend to emit against a CPU reference
before building on it. (Not independently re-probed in the parent session -- it is recorded with
its origin so the next run can confirm or correct it cheaply.)

Two more from the same run, same status -- reported by it, worth confirming when next in scope:

* **An int8 L0 operand's contraction extent is honoured only inside its last 32-deep fractal**
  (the k direction). `K = 1040` was the only failing case of 20 and the only K with
  `K % 128 != 0`; an operand-*width* explanation was ruled out by a separate probe.
* **C15's reduce-scratch is `src/2`, not src-shaped.** A `kWC/2` scratch validated 10/10, so
  src-shaped is sufficient but not necessary. `TROWMAX` takes the scratch as a required third
  argument -- `TROWMAX(dst, src, tmp)`; the 2-argument form does not exist.

## C43: `TQUANT<INT8_SYM>` ALIASES ITS STAGING TILE ONTO THE SOURCE -- WRONG ON A TAIL

`TQUANT_IMPL` in `pto/npu/a2a3/TQuant.hpp` does this (two call sites, lines ~97 and ~153):

```cpp
TASSIGN_IMPL(src_f16, reinterpret_cast<uintptr_t>(src.data()));   // fp16 staging
TASSIGN_IMPL(src_s32, reinterpret_cast<uintptr_t>(src.data()));   // int32 staging
```

Both staging tiles are placed **on top of the fp32 source buffer**. When the fp32 -> fp16
conversion splits into a full-repeat pass plus a **column tail**, the main pass's fp16 write
for row `2i+1` (byte `2*(2i+1)*RowStride`) lands on **fp32 row `i`'s tail columns**
(byte `4*i*RowStride + 4*mainCols`) *before the tail pass reads them*.

**Fingerprint, measured:** the first row of every work unit comes back saturated
(`|y - golden| = 255`) in exactly the last `w mod 64` columns of each chunk, while the
per-token `scale` -- computed from the same fp32 data **before** the conversion -- stays exact
to 1e-7. **One output garbage in a periodic column band, another from the same data perfect.**

**Rule:** do not use `TQUANT<QuantType::INT8_SYM>` on A2/A3 for a tile whose valid width is
neither a multiple of the 64-element fp32 repeat nor `<= 64`, unless you control the geometry.
Give the conversion its own staging tile instead of letting the library alias the source; on the
op where this was found that cost **zero extra UB** and measured 1.014x.

**Why this rule exists at all** is the part worth internalising: the defect had been in every
build of that kernel for 24 attempts, and **the 20-case contract sweep structurally could not
sample it** -- every benchmark `H` is a multiple of 64, `<= 64`, or leaves a 1-column tail. An
out-of-contract shape probe failed **8 of 11** shapes on the first run. See C44.

## C44: RUN AN OUT-OF-CONTRACT SHAPE PROBE -- THE SWEEP CANNOT FIND THIS CLASS

A contract sweep validates the shapes the benchmark happens to sample. For a kernel declared
dynamic-shape that is not the same as validating the kernel, and this project has now hit the
gap **three times**, each time silently:

| op | window the sweep missed | symptom |
|---|---|---|
| `grouped_matmul_swiglu_quant` | `K % 128` in (64, 128] | 24.9% of `y` wrong, bit-reproducible |
| `dequant_swiglu_quant` | valid width not a multiple of 64 and > 64 | last `w mod 64` columns saturated, `scale` exact |
| (both) | -- | one output wrong, another from the same region exact |

**So: after Phase 5 and after any change to a tiling, tail, or L0/staging layout, probe shapes
the contract does NOT contain** -- each dim just above and just below every internal blocking
constant, and at least one prime. Compare against the CPU float64 reference, not against
another run of the same kernel.

This is cheap (a handful of launches) and it is the only thing that finds this class. Record the
probed shapes and their results next to the contract sweep, and treat a *narrower* validated
range than the declared contract as a contract amendment, not a footnote.

> **C44 IS NECESSARY AND NOT SUFFICIENT -- see C67.** This probe answers "do unsupported
> shapes fail cleanly?". It does NOT answer "is the supported set wide enough?".
> `sparse_flash_attention` passed this probe (`6/6 out-of-contract shapes rejected with a
> negative rc`), shipped, and scored **0.00 on all 80 hidden cases** because its supported set
> was a ten-entry lookup table. A clean rejection is the right behaviour for a genuinely
> unsupported shape and worthless when the unsupported set is everything the scorer tests.
> Run C44 for the boundary behaviour, C67 for the boundary's *location*, and **C72 to prove
> mechanically that the supported set is as wide as `proto.yaml`/`desc.md` declare**.

## C45: THE MCP CATALOGUES INSTRUCTIONS THAT DO NOT EXIST IN THE COMPILED LIBRARY

The npu-coding MCP indexes a **different** pto-isa than the one we compile against
(`$PTO_LIB_PATH`, github/hw-native-sys). Instructions it lists and describes may
have **no implementation at all** in your checkout. Verified absent so far:

| instruction | MCP | pto-isa `109c9f72` |
|---|---|---|
| `TMADD`, `TMULA` | catalogued | no a2a3 implementation |
| `TINTERLEAVE`, `TDEINTERLEAVE` | catalogued under Data Movement / Layout | **absent from `include/` entirely** |

`TINTERLEAVE`/`TDEINTERLEAVE` matter because they are the obvious primitives for an
interleaved RoPE rotation, and a design that assumes them has to be thrown away. (One run
planned around them, found them missing, and had to build a permutation from an index
`TGATHER` instead -- which then took three iterations to get fast.)

**Rule: before an instruction enters a design, `grep` it in the headers you compile against.**
The MCP is a search aid, not the API. `get_cpp_intrinsic` returning a signature proves the
*index* has it, nothing more. This is the same source-skew already recorded for constraints
and dtype support -- treat existence the same way.

## C46: A UB-TO-UB COPY RUNS ON THE VECTOR PIPE, NOT MTE3

`pto_copy_ubuf_to_ubuf` -- the primitive under `TINSERT`, `TCOLEXPAND` and `TCONCAT` -- is
**not** an MTE3 operation. A run that guarded it with an MTE3 barrier was **silently wrong on
8 of 20 cases**; the correct pipe is `PIPE_V`.

The header does not annotate the pipe (`common/arch_cce_intrinsic.hpp` just declares the
intrinsic), so this is a *behavioural* finding, established by measurement rather than by
reading -- which is exactly why it is easy to get wrong: "copy" reads like data movement, and
data movement reads like MTE.

**Rule:** the pipe an instruction runs on is a property to **verify**, not to infer from its
name or its category in the docs. When a barrier or flag around a data-movement helper does
not behave, test the other pipe before assuming a hardware bug. Family-level guidance:
anything that moves data **within** UB is Vector; only GM<->UB traffic is MTE2/MTE3.

## C47: `rtGetL2CacheOffset` TAKES A DEVICE ID AND SILENTLY RETURNS 0 FOR DEVICE 0

`rtGetL2CacheOffset(deviceId, &offset)` returns **success with `offset == 0`** when asked
about device 0. The offset is per-device, and a hardcoded `0` is the value most callers write.
The consequence is an **L2-bypass alias that is silently not applied** -- the kernel runs
uncached-in-name-only and the measurement looks like "the alias does nothing".

**Reported independently by two runs in the same session**, both of which had already banked
measurements before noticing: one found three alias measurements invalid, the other found its
first alias probe entirely invalid.

```cpp
int32_t d = 0; rtGetDevice(&d);            // ask about the device you are actually on
uint64_t off = 0;
if (rtGetL2CacheOffset(d, &off) != 0 || off == 0) return -4;   // fail loudly
```

**Rule:** never pass a literal device id to a per-device runtime query, and treat
`offset == 0` as failure rather than as "no offset needed". More generally: when a lever
measures as *exactly* neutral, check that it is wired before concluding it does not work --
this is the v0.97 unwired-lever rule with a specific, recurring instance.

## C48: A BACK-TO-BACK RAW ON THE SAME PIPE IS NOT COVERED BY PROGRAM ORDER

Two vector instructions where the second reads what the first wrote, issued back to back on
`PIPE_V`, can execute **out of order**. Measured on a ramp built by doubling
(`ramp[len..2len) = ramp[0..len) + len`): **two runs of the same `.so` corrupted different
32-byte blocks.** One `pipe_barrier(PIPE_V)` between them fixes it -- 6/6 clean.

This is distinct from C41, which is about *cross-pipe* WAR. Here both instructions are on the
**same** pipe and program order still does not guarantee the read sees the write.

**Why it is easy to ship:** it was invisible below ~22-row runs, so **19 of 20 contract cases
passed** while one failed on exactly 256 of 2048 index entries -- and non-deterministically, so
a single re-run could clear it.

**Rule:** when one vector op consumes another's output *in place or into an overlapping range*,
put a `pipe_barrier(PIPE_V)` between them unless you have measured that you can omit it. The
in-order intuition for a single pipe is not reliable on this part. Combined with C41's
amendment, the safe default for any producer/consumer pair sharing a buffer -- same pipe or
not -- is an explicit barrier, and the optimisation is removing the ones you can prove
unnecessary, not adding the ones you find you need.

## C49: THE ROW-REDUCE `tmp` TILE MUST BE `validCol/2` WIDE, NOT 64 -- AND NOTHING CHECKS IT

`TROWSUM` / `TROWMAX` / `TROWARGMAX` take a scratch tile. Its required width is **a function
of `validCol`**, and an undersized one produces a **silently wrong reduction** -- no compile
error, no runtime error, no assert.

`pto/npu/a2a3/TRowSum.hpp::FillTmp` writes `tmp + i * ElemPerRpt` for
`i` in `[0, srcRptPerRow / 2)`, where `srcRptPerRow = ceil(validCol / ElemPerRpt)` and
`ElemPerRpt` is 64 for fp32. So per row:

```
tmp_cols_needed = max(ElemPerRpt, (ceil(validCol / ElemPerRpt) / 2) * ElemPerRpt)
```

`TRowReduceCheck` has five `static_assert`s -- Loc, row-major, dst layout, dtype, dtype
consistency. **None of them mentions `TileDataTmp::Cols`.** `tmpRptStride` is computed from
`TileDataTmp::Cols`, so a narrow tmp just walks into the next row's scratch.

**C49a -- THE FORMULA ABOVE IS CONSERVATIVE, THE EXACT ONE USES `floor`, AND A ONE-ROW `tmp`
HIDES THE BUG INSTEAD OF AVOIDING IT.** Re-probed on `rms_norm` over a 137-point sweep of
(tmp rows, src Cols, tmp Cols, validCol), fp32, A2/dav-c220-vec, rebuilt and re-run
independently. Three results, in order of importance:

1. **The requirement has THREE regimes, and the middle one fails differently.** My first
   correction here said "`floor` twice, no minimum" -- the `floor` half is right and **the "no
   minimum" half was wrong**, caught on the next operator (`softmax`) and re-probed:

   ```
   validCol <= ElemPerRpt            -> tmp unused; any width works (OneRepeatProc branch)
   ElemPerRpt < validCol < 2*ElemPerRpt -> tmp MUST be >= ElemPerRpt, or the whole reduction is
                                        a SILENT NO-OP: dst is never written at all
   validCol >= 2*ElemPerRpt          -> ElemPerRpt * floor(floor(validCol/ElemPerRpt)/2)
   ```

   which collapses back to the shipped form, minimum included:

   ```
   tmp_cols_needed = 0 if validCol <= ElemPerRpt
                     else max(ElemPerRpt, ElemPerRpt * floor(floor(validCol/ElemPerRpt)/2))
   ```

   **The middle regime is the dangerous one and it is not "wrong numbers", it is "no numbers".**
   `TRowReduceOps.hpp:281-285`: inside the `validCol < 2*elemPerRpt` branch there is an
   `if constexpr ((srcRptStride < 8) || (tmpRptStride < 8)) return;` -- placed there, by its own
   comment, to dodge a ccec compile check on `pto_copy_ubuf_to_ubuf`. It returns **before**
   `FillTmp` and before anything writes `dst`. Measured, fp32, dst poisoned with `-31337` first:

   | tmp Cols | blocks | validCol 67 / 100 / 127 | validCol 64 | validCol 128 | validCol 256 |
   |---|---|---|---|---|---|
   | 8 | 1 | **NO-OP, poison intact** | OK | WRONG 9.8e-02 | WRONG 5.8e-02 |
   | 32 | 4 | **NO-OP, poison intact** | OK | WRONG 3.2e-02 | WRONG 1.7e-02 |
   | 64 | 8 | OK | OK | OK | WRONG 1.8e-02 |
   | 128 | 16 | OK | OK | OK | OK |

   Note the `validCol >= 128` columns confirm the `floor` formula exactly (128 needs 64, 256
   needs 128) -- so both halves of the rule are now measured, not inferred. And because the
   guard is `if constexpr`, a tmp tile declared narrower than `ElemPerRpt` makes that
   instantiation **always** no-op. `dst` keeps whatever it held, which in a real kernel is a
   stale value from the previous iteration -- it reads as plausible data, not as an error.

2. **C49 IS TROWSUM-ONLY.** The binary-tree `FillTmp`/`TmpProc` the formula describes is an
   **override on `TRowSumOp`**. `TRowMaxOp` (`a2a3/TRowMax.hpp`) does not define `FillTmp` at
   all -- it inherits `TRowReduceOp`'s base version (`TRowReduceOps.hpp:151`), which writes
   exactly ONE repeat. So `TROWMAX` / `TROWMIN` / `TROWPROD` need only `ElemPerRpt` columns of
   scratch whatever `validCol` is, while `TROWSUM` needs the full formula. Sizing a max's
   scratch by the sum's formula only wastes UB; sizing a **sum's** scratch by the max's rule is
   silently wrong. Check which op you are emitting.

3. **C49 as written is never UNSAFE** -- across all 137 points there is **no case where the
   rule says "wide enough" and the answer is wrong** (0/137). Every one of its 29
   mispredictions is in the safe direction. So keep using it as the design rule; the exact
   formula is for when you are fighting for UB.

4. **A `tmp` tile with ONE ROW passes every correctness check and is still writing out of
   bounds.** `FillTmp` addresses `tmp + i * ElemPerRpt` **linearly from the tile base** -- the
   offset does not involve `TileDataTmp::Cols` at all, which is *why* a narrow multi-row tmp
   corrupts row 1. With `Rows == 1` there is no row 1 inside the tile to corrupt, so the
   reduction reads back what it wrote and the answer comes out right: **20/20 one-row cases
   correct, including `validCol = 8192` with a 64-wide tmp**, while the same `(Cols, validCol)`
   at `Rows > 1` is wrong by up to **4.8 relative**. That write is 16 KB past the end of a
   256-byte tile. It only looked correct because nothing else in the probe owned that UB yet.
   **`Rows == 1` is not an exemption from C49 -- it is C49 failing silently one level further
   out.** Size the tmp by the formula whatever its row count.

**This also RETIRES the claim "fp32 Vec tile `Cols >= 2048` breaks row reduce".** It does not.
The first failure in fp32 is at **`validCol = 256` with a 64-wide tmp**, the threshold moves
with the tmp width exactly as the formula says, and `validCol = 2048` and `4096` are **exact
(1.1e-7)** once the tmp is sized correctly. The original observation was a correctly-sized-tile
/ undersized-scratch confusion -- the same mis-attribution C15 had.

Measured directly (fp32, 16 rows, `reports/probe_rowsum_tmpwidth/` in
`skillyard-runs-v101/moe_gating_top_k_softmax`), max relative error vs a CPU float64 row sum:

| validCol | tmp cols | needed | max rel err | |
|---|---|---|---|---|
| 64 | 64 | 64 | 9.6e-08 | ok |
| 128 | 64 | 64 | 9.2e-08 | ok |
| 256 | **64** | 128 | **6.9e-02** | WRONG |
| 256 | 128 | 128 | 1.2e-07 | ok |
| 512 | **64** | 256 | **1.09e+00** | WRONG |
| 512 | **128** | 256 | **2.8e-01** | WRONG |
| 512 | 256 | 256 | 5.2e-08 | ok |

The formula predicts every row. Note the error is **not** a clean factor -- it depends on what
the overrun lands on -- so it does not look like a scaling bug and will not be recognised as
one.

**Why `validCol = 128` is the trap.** At 128 columns the requirement is exactly 64, so the
"obvious" 64-wide tmp is correct, and a kernel validated only at small widths passes. The
first wider case then fails and reads as an *accuracy* problem in the surrounding math. In the
`moe_gating_top_k_softmax` run this took out 5 of 20 cases (every `E >= 256`) and was chased
as a softmax precision issue.

**Rule:** size the row-reduce scratch from `validCol` with the formula above and
`static_assert` it against the tile's `Cols`. If `validCol` is dynamic, size the tile for the
contract's **largest** `validCol`, not the one you are testing. A row reduction that is right
at 64 and 128 columns has told you nothing about 256.

## C50: `TCOLEXPAND*` NARROWS `validRow` TO `uint8_t` -- BUT ONLY ON THE STATIC-`ValidCol` PATH

`ColExpandBinInstr` takes `uint8_t repeats`, and `TColExpandBinaryNormMode`
(`pto/npu/a2a3/TColExpandBinOp.hpp:33`) passes `unsigned validRow` straight into it. No clamp,
no assert -- `TCOLEXPANDOP_IMPL`'s two `static_assert`s cover dtype and row-major only. So on
that path the op computes `validRow % 256` rows and silently leaves the rest untouched.

**Which path you get is a COMPILE-TIME property of the tile declaration.**
`ColExpandBinaryInstr` dispatches on `TileData::Cols == TileData::ValidCol || TileData::Rows == 1`:

| tile | dispatch | behaviour |
|---|---|---|
| `Tile<..., R, C, RowMajor, -1, -1>` (dynamic ValidCol) | `TColExpandBinaryCountMode` -- a per-row `for` loop | **correct at any R** |
| `Tile<..., R, C, RowMajor, -1, C>` (static `ValidCol == Cols`) | `TColExpandBinaryNormMode` -- one call, `validRow` as `uint8_t` | **`validRow % 256` rows** |

Measured, `TCOLEXPANDDIV`, fp32, 64 columns, sentinel-filled dst, counting rows actually
written (`reports/probe_colexpand_uint8/` in `skillyard-runs-v101/cross_entropy_loss`):

| rows | dynamic ValidCol | static ValidCol |
|---|---|---|
| 128 | 128 | 128 |
| 255 | 255 | 255 |
| **256** | **256** | **0** |
| **300** | **300** | **44** |

At 256 rows the static-`ValidCol` form computes **nothing at all** and reports success.

**Note `ValidRow` must stay `-1` regardless** -- `SetValidRow` carries
`static_assert(ValidRow == DYNAMIC, "Only Dynamic Valid Row Support Set Value.")`.

### The other half: what the dynamic form COSTS, and when that matters

CountMode is a C-level `for (i < validRow)` issuing **one instruction per row**; NormMode
issues **one instruction with `validRow` hardware repeats**. A later run reported this as a
large penalty -- switching its destinations to static `ValidCol` took `aiv_scalar` from
**92.0 to 66.4 us, -28%**, in a kernel where scalar was co-binding.

**Measured in isolation, the two paths are the same speed.** 200 back-to-back
`TCOLEXPANDDIV` calls per launch (so the ~2 us launch floor cannot bury the difference),
fp32, 64 columns, arms alternated (`reports/probe_colexpand_dispatch/` in
`skillyard-runs-v102/engram_gate_fusion`):

| rows | dynamic (CountMode) | static (NormMode) |
|---|---|---|
| 128 | 48.221 / 48.221 us | 47.781 / 47.961 us |
| 255 | 91.102 / 91.062 us | 90.902 / 90.642 us |

0.4-0.9% apart, and both scale linearly with rows. So "48 instructions instead of 1" is not
by itself a cost: the hardware repeat and the issued instructions do the same work at the
same rate **when the vector unit is the bottleneck**.

**Reconcile the two with C52.** Instruction count is free when you are throughput-bound and
expensive when you are issue-bound -- and C52 gives the signature of issue-bound: nonzero
`aiv_icache_miss_rate`, a pipe-ratio sum below 1.0, scalar dominating Duration. My probe is a
tight loop of full-width vector ops, i.e. throughput-bound, which is exactly why it shows
nothing. The reporting kernel was scalar-co-bound, which is exactly why it showed 28%.

**Rule, both halves:**

* **Default to `-1, -1`.** It is correct at any row count and costs nothing measurable unless
  you are issue-bound.
* **Only if you have MEASURED that you are issue-bound** (C52's signature) is static
  `ValidCol` worth reaching for -- and then you must **chunk rows to <= 255 per call** and
  `static_assert` the chunk height, or you will silently compute `validRow % 256` rows.
* Do not switch tile forms on the theory that fewer instructions must be faster. Two probes
  in two days say that theory is regime-dependent, and the regime is measurable.

This rule was reported to me twice, in opposite directions -- once as an unconditional
">255 rows computes nothing" and once as an unconditional "the dynamic form costs 28%".
Probing narrowed both. A kernel that uses both tile forms will see one hazard on one and the
other on the other.

## C51: GUARDING THE `__global__` DECLARATION WITH `__CCE_AICORE__` DELETES THE LAUNCH, SILENTLY

`bisheng -xcce` runs **two passes**: a device pass with `__CCE_AICORE__` defined and a **host
pass with it undefined**. The `<<<...>>>` launch is lowered in the HOST pass. So if the
`__global__` entry point's **declaration** sits inside `#if defined(__CCE_AICORE__)`, the host
pass sees no declaration, the launch statement compiles to nothing, and you get:

* a clean compile, no warning;
* a `.so` that loads;
* `call_kernel` returning **0**;
* the output buffer **completely untouched**.

Reproduced minimally -- one source, one `-D` flip, `reports/probe_launch_guard/` in
`skillyard-runs-v102/adaptive_avg_pool_3d`:

| build | rc | elements written | |
|---|---|---|---|
| declaration unguarded, body guarded | 0 | **256 / 256** | launch ran |
| declaration inside `#if defined(__CCE_AICORE__)` | 0 | **0 / 256** | **silently dropped** |

**Rule:** the `__global__` entry point's *signature* and the `<<<>>>` call site are **host-visible
code and must never be inside a `__CCE_AICORE__` guard**. Guard the **body**:

```cpp
extern "C" __global__ AICORE void launch_stage(__gm__ uint8_t* out, ...) {
#if defined(__DAV_C220_VEC__)          // guard the BODY
  ...
#endif
}

extern "C" void call_kernel(uint32_t block_dim, void* stream, uint8_t* out, ...) {
  launch_stage<<<block_dim, nullptr, stream>>>(out, ...);   // NEVER guarded
}
```

`BUILD-§` already puts the guard inside the body; `COOK-§1`'s skeleton does not make the
consequence explicit, and that omission cost two separate runs their longest debug of the week
-- both found it independently, both by output poisoning.

**Detection:** this is invisible to any test that does not check the output actually changed.
**Poison every output buffer before every launch** (fill with a sentinel, assert it is gone).
A zero return code from `call_kernel` means nothing at all here.

> **C51a -- THE POISON DETECTOR ITSELF CAN LIE, for the C60 reason.** `y.fill_(poison)` is a torch
> op, and a direct `<<<>>>` launch can overtake it through torch_npu's host task queue. Measured:
> four identical launches, and trial 2 came back **fully poisoned** -- i.e. the detector reported
> "the kernel wrote nothing" when the kernel had in fact run and the *fill* had landed after it.
> A false "wrote nothing" is as expensive as the bug it was built to find.
> **Sync between the poison fill and the launch** (`torch.npu.synchronize()`), the same edge C60
> requires. A detector built out of torch ops inherits every torch-ordering hazard.

## C52: FOR A SMALL ELEMENTWISE KERNEL, PTO'S GENERIC WRAPPERS ARE THE BOTTLENECK -- CHECK `aiv_icache_miss_rate`

> **C52-BARRIER -- READ THIS BEFORE YOU APPLY C52. THE WRAPPERS WERE ISSUING YOUR INTRA-PIPE
> BARRIERS. WHEN YOU REPLACE THEM WITH RAW INTRINSICS YOU OWN THAT ORDERING, AND THE FAILURE IS
> SIZE-DEPENDENT, SO IT WILL PASS EVERY GATE YOU HAVE.**
>
> C52 tells you to drop `TLOAD`/`TSTORE`/`TUnaryOp` for raw intrinsics and shows a 3.04x win. It
> does **not** follow that the wrappers were doing nothing else. They insert the
> `pipe_barrier(PIPE_V)` between dependent vector issues that C48 requires. Raw intrinsics do not.
>
> On `mish`, a chain of 7 dependent raw vector issues with no barriers **validated 20/20 on the
> contract sweep AND 72/72 on the out-of-contract stress sweep, and was wrong.** One source, one
> `-D` flip, re-run independently on a fresh build:
>
> | chunk | repeats per issue | barriers ON | barriers OFF |
> |---|---|---|---|
> | 7680 (the shipped size) | 120 | 20/20 + 72/72 | **20/20 + 72/72 -- bug invisible** |
> | 512 | 8 | 20/20 | **0/20** |
> | 256 | 4 | 20/20 | **0/20** |
>
> A long issue hides the hazard: by the time the next instruction reads the destination, the
> previous one has retired anyway. Shrink the issue and **every case** is wrong -- not a few
> elements, not a tail.
>
> **This is why it is dangerous.** Your contract sweep runs at the production chunk, which is the
> large one, so it reports green. The out-of-contract stress sweep varies SHAPE, which does not
> change repeats-per-issue, so it reports green too. Nothing in the pipeline varies the chunk.
>
> **Rules:**
> 1. Every dependent raw-vector pair gets a `pipe_barrier(PIPE_V)`. Budget it: on `mish` the
>    barriers cost **4.3%**, which is the price of the kernel being right.
> 2. **Ship the barriers behind a compile-time knob** (`-DMISH_VBAR=0/1`, or `sigmoid`'s
>    `PTO_BARMASK` bitmask, one bit per barrier) so the claim stays reproducible and so a future
>    run can bisect rather than remove them all at once.
> 3. **Validate at a SMALL chunk as well as the production one.** A raw-intrinsic kernel that has
>    only been validated at its production tile size has not been validated. Add at least one
>    `chunk <= 512` run to the gate.
>
> `sigmoid` already had this right -- `PTO_BARMASK` defaults to `0x3F`, all six on, with
> "all-off is MEASURED INCORRECT, 25/26" in its source. That knowledge sat in one kernel's
> comments for the whole campaign instead of in this rule, and `mish` paid to rediscover it.


`level1/sigmoid` was the worst op in our cann-bench campaign for weeks: score 58.40, **0 of 20
cases beating the published baseline**, on the simplest op in the benchmark. The store said
"scalar-bound, cause unresolved."

**The cause is instruction-fetch, and the tell is a profiler column nobody was reading.**

| | ours (v0.97) | vendor `aclnnSigmoid` |
|---|---|---|
| `aiv_scalar` share of Duration | **43-81%** | 2-5% |
| **`aiv_icache_miss_rate`** | **6.1-10.8%** | **exactly 0.000** at every size |
| pipe ratio sum, small cases | **0.84** (every pipe idle 16%) | 1.34 |

`PTO`'s generic `TLOAD`/`TSTORE`/`TUnaryOp` wrappers emit **21,180 bytes of device `.text` for a
seven-instruction inner loop**: ~2,400 scalar cycles per `TLOAD` and ~320 extra per vector op.
`TLoadGm2ubNd2nd` wraps one `copy_gm_to_ubuf_align_bXX` in a **runtime triple loop over
`gShape0/1/2`** -- for a `Shape<1,1,1,1,C>` tensor, where all three trip-count 1. Issuing the
same MTE and vector instructions directly takes `.text` to 6,076 B and icache to **0.000%**.

Measured end to end, same harness, same idle device, arms alternated in one session
(`skillyard-runs-v102/sigmoid_rootcause`, re-verified independently by the parent):

| | score | mean HAP | beats baseline | |
|---|---|---|---|---|
| PTO wrappers | 58.39 | 0.168 | 0/20 | |
| raw intrinsics | **72.72** | **0.454** | **5/20** | geomean **3.04x** per case |

**The gain is a function of how issue-bound the kernel is**, so scope it:

* case 1 (1.05 M fp16, small): 23.90 -> 5.98 us, **4.00x**
* case 5 (67.1 M fp32, bandwidth-bound): 414.16 -> 343.44 us, **1.21x**

An ablation confirms the arithmetic was never the problem: with the sigmoid body replaced by a
pure copy the kernel still ran **356.68 us against the vendor's complete sigmoid at 337.90** --
**zeroing the transcendental entirely left us slower than the vendor.** `TEXP` + `TRECIP`
together are 7.2% of the time; the whole arithmetic chain is 11.6%.

### C52a AMENDMENT -- IT IS A CONSTRUCTION RULE, NOT A REPAIR, AND IT DOES NOT BUY 3x TWICE

`level1/gelu`, a longer transcendental chain than sigmoid, written with raw intrinsics **from line
one** rather than converted:

| | gelu (raw from the start) | sigmoid before C52 |
|---|---|---|
| fixed cost vs vendor | **1.2x** | 8.4x |
| `aiv_icache_miss_rate` | **0.000 on all 20 cases** | 6.1-10.8% |
| device `.text` per kernel | 3.7-4.7 KB | 21.2 KB |
| pipe-ratio sum | never below 1.0 | 0.84 |

So the C52 signature was **absent from the start** -- the rule bought its win at construction
time, and there was no 3x left lying around to collect. **Do not expect C52 to be worth 3x on
every elementwise op.** Sigmoid's 3.04x was the cost of *repairing* a kernel built on the
wrappers; a kernel built raw simply never pays it. Where gelu and the vendor compute the **same**
function (the 9 `tanh` cases) our slope is **1.04x theirs** -- essentially parity.

Corollary worth internalising: once C52 is applied, the remaining gap is **algorithmic**, and you
should go look for it there. gelu's residual on its other 11 cases is 33 vector cycles per repeat
against the vendor's 19 -- and the reason turned out to be that the vendor was computing a
different, cheaper function (see the `level1/gelu` benchmark defect). Two of its largest cases sit
at a **DMA floor**: a load+store-only probe ran 38.44 and 41.60 us against the vendor's *complete*
kernel at 39.22 and 37.98, so no arithmetic change could have helped them at all.

**Rules:**

1. **Add `aiv_icache_miss_rate` to every `PipeUtilization` read.** A nonzero value on a small
   kernel means the inner loop does not fit, and no amount of pipe rebalancing will fix it.
   A pipe-ratio sum **below 1.0** says the same thing from the other side: nothing is
   overlapping because the core is starved of instructions, not of work.
2. **For an elementwise kernel whose inner loop is a handful of instructions, do not pay for
   the generic tile wrappers.** Issue the MTE and vector intrinsics directly. The wrappers earn
   their size on shaped, strided, multi-dimensional traffic; on a flat 1-D surface they are
   pure overhead.
3. **Suspect this whenever fixed cost dominates.** The intercept of the size sweep was **17.08
   us against the vendor's 2.03 us (8.4x)** while the slope was only 3.1x off -- fixed cost,
   not throughput, and that ratio is the signature.

## C53: THE WRONG `--cce-aicore-arch` IS SILENT -- A MIX KERNEL BUILT `-vec` RUNS AND DOES NOTHING

`--cce-aicore-arch` selects which engines exist. Build a kernel that uses Cube with
`dav-c220-vec` and:

* it **compiles** -- the Cube code is inside `#if defined(__DAV_C220_CUBE__)`, which is simply
  false, so it vanishes;
* it **links**;
* it **launches**;
* `call_kernel` returns **0**;
* every Cube instruction is **gone**, and the output is whatever the Vec half alone produced.

This is the same shape as C51 (a guarded `__global__` declaration deleting the launch): the
guard does its job, the build is clean, and the only evidence is an output that never changed
or changed wrongly. Here it is worse than C51, because the kernel does *some* work -- a partially
correct output is harder to spot than an untouched one.

**Where it bites in practice:** any build system with ONE tree-wide arch setting. The cann-bench
submission trees had exactly that, hard-coded to `dav-c220-vec` because the first four ops
packaged were vector-only; the first MIX op to go in would have been silently gutted. The fix was
to make arch **per source** (`register_cce_kernel_arch(<src> <suffix>)`, keyed by
`MAKE_C_IDENTIFIER` of the source path, family prefix from a variable so the tree stays
portable), so an op declares its **engine** -- `""` = MIX, `"-vec"`, `"-cube"` -- and sources that
do not register keep the default.

**Rules:**

1. **Arch is a per-kernel property, never a per-tree one.** The moment a second kernel joins a
   build, the shared arch flag is a bug waiting for the first kernel that disagrees.
2. **`static_assert` the engine you require**, so the mistake becomes a compile error instead of
   a silent one:
   ```cpp
   #if !defined(__DAV_C220_CUBE__)
   #error "this kernel requires Cube: build with --cce-aicore-arch=dav-c220 (MIX) or -cube"
   #endif
   ```
   That one line converts the entire failure mode from "wrong numbers in production" to "build
   stops". Put it in every kernel that names an engine.
3. **Verify the engine on device, not from the build log.** The profiler reports `Accelerator
   Core` per kernel (`AI_CORE` vs `MIX_AIC`); check it says what you intended.

## C54: THE HARDWARE AIV-ONLY `SYNCALL` HANGS IN A PLAIN VEC-ONLY LAUNCH -- USE A SOFTWARE BARRIER

Any kernel that reduces ACROSS cores in ONE launch -- a norm, a sum, a max, anything whose
output is smaller than one lane's share -- needs a cross-core barrier between "publish my
partial" and "combine everybody's". `pto/npu/a2a3/SyncAll.hpp` offers one:

```cpp
pipe_barrier(PIPE_ALL);
ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x0, SYNC_AIV_ONLY_ALL));  // == 3585
wait_flag_dev(SYNC_AIV_ONLY_ALL);                                     // == 14
```

**It deadlocks the vector core** when the kernel is launched as
`<<<blockdim, nullptr, stream>>>` and built `--cce-aicore-arch=dav-c220-vec` -- at **every**
block_dim including **1**, and with `set_ffts_base_addr(rtGetC2cCtrlAddr(...))` already done at
entry. It presumably needs an FFTS+ task context that only a MIX/FFTS dispatch establishes;
pointing the core at the C2C control area by hand is not enough.

Measured (`skillyard-runs-v104/foreach_norm/src/probes/probe_sync.cpp`, dav-c220-vec,
CANN 9.1.0). Each lane writes `lane+1` to a GM slot, barriers, lane 0 sums the slots;
correct answer `bn*(bn+1)/2`. Mode 0 is the sensitivity control and must be wrong:

| mode | bd 1 | bd 8 | bd 24 | bd 48 |
|---|---|---|---|---|
| 0 -- no barrier (control) | 1 / 1 | 10 / 36 | 293 / 300 | 29 / 1176 |
| 1 -- `ffts_cross_core_sync` + `wait_flag_dev` | **hang** | **hang** | **hang** | **hang** |
| 2 -- software GM barrier | 1 / 1 | 36 / 36 | 300 / 300 | **1176 / 1176** |

**The software barrier that works, and the three things that make it cheap.** It started at
**6.6-7.6 us** -- half the duration of every case under 2 MiB -- and six measured variants took
it to **~2.8 us**:

1. **The generation comes from the HOST, not from the slot.** A read-modify-write
   (`cur = slot + 1`, then wait for all slots `>= cur`) costs a full GM read round trip before
   the poll can start, AND it deadlocks the moment two launches share one workspace at
   different `block_dim` -- a lane that sat out the previous launch is permanently one
   generation behind. Pass a monotone `int32 gen` as a kernel argument and just WRITE it.
2. **Slots 4 B apart, not 32 B.** The poll then reads 192 B / 3 cache lines at block_dim 48
   instead of 1536 B / 48. Store with `copy_ubuf_to_gm_align_b32` (the non-align form's minimum
   burst is one 32 B block and would clobber seven neighbours). **But only if step 3 is also
   done**: with ALL 48 lanes polling those 3 lines the contention makes it **1.8x WORSE**
   (24.0 us against 13.7 us) than the 48-separate-lines layout.
3. **Only the lanes that CONSUME the partials poll.** Everyone else publishes and leaves;
   nothing waits on them.

Dropping `dcci` from the poll measured inside the null band -- keep it. Dropping
`pipe_barrier(PIPE_ALL)` from inside the poll loop is worth ~0.7 us. **Dedicating lanes to the
combine** so they are already polling when the last producer arrives measured **2.7%-8.9%
WORSE** on 8 of 9 cases: the residual is the poll's GM round trip, not the arrival skew, and
the lost compute width costs more than it saves.

**Do it in ONE launch anyway.** A second kernel for the combine is correct and simple, and it
costs another ~4.5 us launch floor -- on `level1/foreach_norm` four of the twenty published
baselines are under 14 us, so that is the whole margin.

**And it needs the device.** A spin barrier requires all of the launch's blocks to be
co-resident. Sharing the device with a second process **times it out** (reproduced: the same
wheel validated 20/20 idle and timed out in the poll loop under a concurrent benchmark).
Official evaluation gives the operator the device; a shared box does not.

## C55: A DMA BURST SHORTER THAN ONE 32 B BLOCK REPLICATES, AND AN MTE STORE'S **UB** SOURCE MUST BE 32 B ALIGNED

Two separate hazards on the same code path -- moving a handful of fp32 partials between UB and
GM -- and both are silent in the way that matters: the first is a wrong number, the second is
list-length dependent.

**(A) `copy_gm_to_ubuf_align_b32` with `lenBurst < 32 B` fills the whole destination block with
COPIES of the value.** A 4-byte load came back as **eight** copies, and the 64-lane `vcadd`
that followed summed it eight times: the kernel read **exactly 8x its own sum** on every
summing case while the GM it read from was bit-exact (`foreach_norm`, case 8 gave 37.084
against a golden 22.050, and `37.084^4 / 22.050^4 = 8.000`). 8 = 32 B / 4 B. The **store** side
is exact at the same length -- a 4-byte `copy_ubuf_to_gm_align_b32` writes 4 bytes and leaves
the next seven words alone (verified by dumping the destination) -- so it is the **load** that
broadens.

*Rule:* size any GM<->UB partial array in whole 32 B blocks -- give it a FIXED stride that does
not depend on the lane count -- and mask the reduction to the live lanes with COUNT mode
instead of shortening the burst. This is invisible at block_dim 8 and above, because
`nlanes * 4` is then a whole number of blocks; it appears at block_dim 1-7, which is exactly
what you reach for while debugging.

**(B) The UB address an MTE store reads FROM must be 32 B aligned.** The GM destination may be
4-byte aligned; the UB source may not.

```cpp
vcadd(SCR, ACC, LT, 1, 1, 8, false);                       // LT results, 4 BYTES apart
fn_store<float>(gm + ..., SCR + t, 4u);                    // faults for every t > 0
```

`errorStr: The access address of the MTE instruction is not aligned with the data type bit
width`, `subErrType:4`. Every single-tensor case passed and every list of two or more faulted.
*Fix:* `vcadd(SCR, ACC, LT, 8, 1, 8, false)` -- a destination repeat stride of 8 elements, so
each result lands on its own 32 B boundary.

## C56: MASK **COUNT** MODE REMOVES THE RAGGED-TAIL PATH ENTIRELY, AND IT IS EXACT

`set_mask_count(); set_vector_mask(0, n);` with every instruction issued at `repeat = 0` makes
each op process exactly `n` elements. Probed against a CPU float64 reference at
n = 64, 65, 100, 128, 192, 1000, 4096, 12032 (`foreach_norm/src/probes/probe_reduce.cpp`):
`vcadd`, `vcmax`, `vabs` and `vmul` (including `dst == src0 == src1`) are **exact at every
length**, ragged ones included, and `vcadd` writes `ceil(n/64)` contiguous partials with
`dstRptStride = 1`. NORM mode with `repeat = ceil(n/64)` is correctly wrong on the dead lanes of
the final repeat -- at n=65 it returned `-1.5e+38` for the second repeat where COUNT mode
returned the exact value.

So a tiled kernel needs **no tail path at all**: run every tile, full or ragged, in COUNT mode
with the tile's own element count. That is strictly simpler than a masked `vector_dup` over the
dead lanes and strictly safer than padding the input (there is no pad value that is the
identity for a negative norm order -- `0^-1` is `inf`).

`vabs` has **no `bfloat16_t` overload** (`error: the 1st parameter maybe need a type
'__ubuf__ half *'`). Taking `|x|` in the input's own 16-bit width is still worth it -- one b16
issue covers 128 elements against the fp32 form's 64 -- but do it with a **`vand` against
`0x7fff` on a `__ubuf__ uint16_t*` view**, which is a bitwise sign clear: exact, and identical
for float16 and bfloat16. Reinterpreting a bfloat16 buffer as `half*` to borrow the fp16
`vabs` is NOT provably exact (fp16 denormal flush, NaN canonicalisation).

## C57: A 2-D GM VIEW COSTS ONE MTE2 BURST **PER ROW** -- FLATTEN IT WHEN THE ROWS ARE CONTIGUOUS

A `GlobalTensor` with a row stride makes `TLOAD` issue **one MTE2 burst per row**. With 48 lanes
each issuing 48 rows that is 2304 sub-cache-line requests at once, and the cost is **linear in
lane count**.

Measured two ways that agree. A micro-probe (`reports/probe_2dview_parent/`, 64 `TLOAD`s of the
**same 1536 contiguous bytes**, 2-D `[48,8]` view against a flat `[1,384]` view, arms alternated):

| lanes | flat | 2-D | ratio |
|---|---|---|---|
| 1 | 9.66 us | 53.89 us | **5.58x** |
| 16 | 46.13 | 701.59 | **15.21x** |
| 48 | 136.30 | 2103.00 | **15.43x** |

Per `TLOAD` at 48 lanes that is 2.13 us flat against **32.86 us** 2-D. Independently, bisecting a
shipped kernel put **30.5 us** on one such `TLOAD` of a `[48,8]` fp32 tile -- 1536 bytes. The two
numbers were obtained separately and match.

**It is the 2-D-ness, not the `DYNAMIC` extent:** static extents measured 0.344 vs 0.333 us, no
difference. And the marginal cost is flat per request, ~6.1 ns.

**Rule: if a GM region's rows are contiguous, view it as `[1, rows*cols]`, not `[rows, cols]`.**
The tile that receives it can be reshaped just as freely. This costs nothing to do and is worth
more than most schedule changes: flattening **two call sites** in one shipped kernel was worth
**1.231x geomean** and **+2.93 operator score points**.

**Why it hides:** a 1536-byte load looks trivially cheap, so it is the last thing anyone bisects.
It does not show up as a scalar bottleneck and it does not raise `aiv_icache_miss_rate`. Look for
it whenever a phase costs far more than its byte count can justify.

**C57b: THE PAYOFF IS BOUNDED -- IT NEEDS SMALL ROWS, CONTIGUOUS ROWS, AND MANY LANES.**
Flattening is not a general win, and a sweep for 2-D views is not a to-do list. Measured on
`mha`, which has **more 2-D GM views than any other shipped kernel** (19 across its two shipped
kernels): flattening every site that is provably contiguous is worth **nothing** -- geomean
**0.9984**, all 20 cases inside their own 3-replicate null band (median 3.5%, worst 9.6%),
operator score **-0.01**. Three reasons, and they are the checklist to apply BEFORE bisecting:

1. **Most 2-D views are genuinely strided and cannot be flattened at all.** A BSND attention
   layout puts consecutive tokens `N*D` apart, and `N` is 8-32 across the whole sweep, so every
   Q/K/V/y access is a real sub-block of a larger matrix. 16 of `mha`'s 19 sites are like this.
   The `wqbmm` win came from a **private workspace plane** the kernel laid out itself, which is
   exactly where contiguous-and-small rows come from.
2. **The row must be SMALLER THAN A CACHE LINE for the per-request cost to dominate.** The
   probe's marginal cost is ~6.1 ns per 32-byte request, against ~95 GB/s once a burst is long,
   so the crossover is around **500-600 bytes per row**. `wqbmm`'s row was **32 bytes** (8
   floats) -- deep in the pathological regime. Every flattenable `mha` row was **256-1024
   bytes** -- already at or past the crossover, so one burst per row was near-optimal and there
   was nothing to recover.
3. **A runtime `if` to pick the flat path can cost more than the flatten saves.** The three
   `mha` cases where a conditional flat path fired (`dimSkv == kT`) came out at geomean
   **0.9844**, against **1.0009** for the 17 cases that took the unconditional flatten only.
   If contiguity is not a compile-time property, leave it alone.

**So the rule to apply is:** flatten when the region is a kernel-private workspace plane whose
rows are contiguous AND under ~512 bytes AND read by many lanes at once. Outside that box,
measure before believing it, and expect zero.

### C66 vs C57a: A `SYNCALL` KERNEL IS EXACTLY WHERE QUERYING THE CORE CAP IS **MANDATORY**  🔴 **CRITICAL**

> **CORRECTED 2026-09-30, and the previous version of this section is what cost a real run.** It said
> *"if a kernel contains `SYNCALL<*>`, `block_dim <= 24` is a correctness guard and querying the cap is
> a REGRESSION."* **That is backwards.** `SYNCALL` is precisely the case where the query is
> **mandatory**, because over-subscribing a barrier is a **deadlock**, not merely idle cores.
>
> **The real rule: `SYNCALL` deadlocks when `block_dim > the part's cube core count`** -- and **24 was
> never architectural. It is the A2's core count.** C57a was measured on a 24-core A2 and the number
> got generalised into a constant.
>
> ```
> Ascend910B2.ini    (A2, local)   ai_core 24   cube 24   vector 48
> Ascend910_9362.ini (A3, RUNNER)  ai_core 20   cube 20   vector 40
> ```
>
> Both files are under
> `/usr/local/Ascend/cann-*/aarch64-linux/data/platform_config/` and readable **with no device at
> all** -- so this is a Preflight check, not an experiment.
>
> **What it cost:** an operator shipped `block_dim = 24` into an all-core `SYNCALL<Mix>` on an A3 die
> with 20 cube cores. Twenty blocks arrive, wait for twenty-four, never retire; blocks 20-23 can never
> be scheduled. Guaranteed deadlock -> `retCode=0x25 [aicore timeout]`, 507014, on **two shared A3
> dies**. **13 of its 20 cases launch above 20.** Measured: `bd = 8, 10` PASS; `bd = 24` FAULT, twice,
> on two different dies.
>
> **`24` is a ceiling to clamp WITH the query, never INSTEAD of it:**
> ```python
> cap = 20                                    # A3 FLOOR, not 24 -- a failed query must not re-hang
> try:  cap = int(torch.npu.get_device_limit(dev).get("cube_core_num", 0)) or 20
> except Exception: pass
> block_dim = max(2, min(24, cap, need))      # query AND ceiling
> ```
> **The fallback must be 20, not 24** -- every remote runner is an A3, so a failed query on the default
> path is the same hang. Seven other trees in this fleet query correctly but fall back to **24**; that
> is A3-unsafe and should be 20.
>
> **And it is consistent with the neighbouring record rather than contradicting it:** an operator that
> over-subscribed `block_dim` to 2x the physical count "launched and computed correctly" -- because it
> has **no `SYNCALL`**, so its blocks simply serialise. The hazard is the barrier, not the grid.
>
> **What the old verdict got right, and keep:** do not **delete** the `return -N` guard. Clamping
> downward is safe; removing the refusal is not, because a workspace sized `NBLK` and indexed by
> `get_block_idx()` still needs its own bound.

**The two rules do not actually conflict once the cap is queried rather than assumed.** C66 says never hardcode a core count -- query it, keep the
literal only as a fallback. But in a kernel containing `SYNCALL<*>`, `block_dim <= 24` is a
**correctness guard**, and replacing it with a queried cap is a **regression from a clean rejection to a
silent hang** on any part that reports more than 24 cube cores.

This was nearly landed on a real operator. Its `call_kernel` had
`if (block_dim == 0 || block_dim > 24) return -6;`, a C66 pass deleted it, and the change tested
*clean* -- `block_dim` 25 / 64 / 1024 / 4096 all "accepted" with bit-identical outputs. **The test was
vacuous:** the patched `call_kernel` clamps to `min(queried, maxBlk)` and the measuring card reports
`cube_core_num = 24`, so every one of those runs executed with **24** blocks. It proved the clamp
works; it never ran `SYNCALL` above 24, which is the only regime that matters.

**And there was nothing to fix.** The rejection was unreachable from the scored path, because the host
`pick_block_dim` already returned `min(24, need)`. The change removed a correct guard to address a
rejection nothing could trigger.

**The discriminator, and it is the general form of this trap:** an **architectural** cap and a
**fitted** cap look identical in a grep. Ask what exceeding it produces:

| exceeding it yields | class | action |
|---|---|---|
| a smaller tile and a correct answer | fitted / tuning | **widen, or add a general path** |
| idle cores and a slower run | portability (C66) | **query the cap** |
| a wrong answer, a hang, or a device fault | **architectural** | **keep the guard, in BOTH layers** |

So before applying C66 to any kernel: **grep for `SYNCALL`**. If it is present, the core cap stays,
guarded host-side *and* refused device-side with a negative code -- neither layer being the only guard.
The same question applies to any cap whose comment claims an architectural reason: verify the claim
against measured evidence (C57a's is measured -- with one barrier, `block_dim` 25 and 32 never return),
then leave it alone.

### C57a: `SYNCALL<*>` DEADLOCKS SILENTLY ABOVE THE DEVICE'S CORE COUNT

Separate finding from the same investigation, and it is a correctness hazard rather than a
performance one. With **0** barriers, `block_dim` 25 / 32 / 48 all launch and all 75 / 96 / 144
participants report. With **1** barrier, `block_dim` 25 and 32 **never return** -- no error, no
diagnostic, `npu-smi` Health OK throughout.

**The threshold in that measurement was 24 because the part was a 24-cube-core A2 -- it is NOT the
number 24.** A barrier waits for every participant, so the bound is whatever the device actually
has. Shipping a literal `24` to a 20-cube-core A3 hung two shared dies (`retCode=0x25
[aicore timeout]`, 2026-09-29).

**Rule: any kernel containing a `SYNCALL<*>` must clamp `block_dim` host-side to the QUERIED core
count** (C66), and should `TORCH_CHECK` it rather than relying on a schedule that happens to stay
under. The kernel-side literal is a ceiling, not the guard -- keep it, but it cannot be the only
layer. Three details that have each cost a run:

- **Match the fallback to the barrier's core type.** A `Mix` or Cube barrier falls back to the A3's
  **20**, a Vec-only one to **40**. Setting a vector cap to 20 needlessly halves a grid; leaving it
  at 48 keeps the over-subscription.
- **Never cache a FAILED query.** `cached = 0` (or `cached = fallback`) with no retry freezes the
  A2-era value for the whole process and hands the decision back to the kernel's literal.
- **A `SYNCALL` grep is not an audit until you confirm the hits are code.** One tree's three hits
  were comments explaining why it *avoids* a barrier; its cap is a plain grid cap, where
  over-subscription is perf-only.

### C114: A 16-BIT STRIDE DESCRIPTOR **TRUNCATES SILENTLY** -- NO FAULT, WRONG ROWS  🔴 **CRITICAL**

`TLOAD` Mat ND->NZ requires `1 <= Stride3 <= 65535` because the descriptor field is **16 bits**.
Exceed it and the hardware does **not** fault -- it takes the value **mod 65536** and reads the wrong
rows. Measured: a GM row stride of `Nh*(D+Dr) = 128*576 = 73728` becomes `73728 mod 65536 = 8192`,
and the GEMM returned **16,355,107 of 16,776,905 output elements over the gate** (`mare 1.37e5`,
normal band `16355107/0`) while **the kernel exited cleanly**.

**Attribution was clean because the switch was host-side:** both arms' `.aicore_binary` were
**byte-identical**, so the same device bytes produced right and wrong answers depending only on a
runtime branch. A control at a legal stride (`40960`) passed in the same build, and the outputs that
never read the oversized operand passed in every arm.

**So a stride guard of this kind is LOAD-BEARING, and "we have never seen the hardware fail here"
is not evidence that it would not** -- our own guard was refusing the shape first, which is exactly
why the hardware had never been asked. **Falsify a guard by forcing the hardware past it before
concluding the guard is unnecessary.** Removing this one would have shipped silent wrong answers on
a declared shape with no diagnostic anywhere.

**How to handle an operand whose row stride can exceed 65535:** a one-time GM->GM repack into a
layout whose stride fits (here `[Nh][Hcq][D+Dr]`, stride 576), or single-row ND->NZ loads where
`Shape3 = 1` means no second row is addressed and `Stride3` becomes irrelevant. Gate the repack on
the stride actually exceeding the limit so the common path is untouched -- and then **force the
repack's own unexercised branches** (multi-block, ragged tail), because the shape that triggers it
in practice usually lands on the easiest one.

### C119: A `PIPE_S` INSTRUCTION NEEDS AN EXPLICIT `PIPE_S -> PIPE_V` SYNC -- `pipe_barrier(PIPE_V)` IS NOT IT  🔴 **CRITICAL**

**`pto-isa`'s own event table assigns `TCI` to the SCALAR pipe.** `pto/common/event.hpp:154` reads
`PIPE_S /* TCI */`; the only other `PIPE_S` entries are `SCALAR`, `TRESHAPE`, `SETFMATRIX`,
`SET_IMG2COL_RPT` and `SET_IMG2COL_PADDING`.

**And it depends on which overload you call:**

| form | template / runtime args | implementation |
|---|---|---|
| `TCI<Tile, T, desc>(dst, start)` | 3 / 2 | **scalar store loop** -- `for (i < validCol) *(dstPtr+i) = start +/- i;` (the source even comments it `// scalar`) |
| `TCI<Tile, Tmp, T, desc>(dst, start, tmp)` | 4 / 3 | the **vector** path (`TCI_b32_repeat`) |

`pipe_barrier(PIPE_V)` orders V against V. **It does not order the scalar pipe's UB writes against
the vector pipe's reads.** So the 2-argument form followed only by `pipe_barrier(PIPE_V)` is a
**race**, and it is worst when `validCol` is large (a 256-iteration scalar fill).

**It cost a device fault, and the fault blamed the wrong thing.** On `cross_entropy_loss` the filled
tile was the byte-offset operand of a `TGATHER`; a stale lane became an out-of-range offset and the
device reported:

```
errorStr: The address for the VEC instruction to read/write UB is out of bounds.   (507035)
```

3 faults in 6 evaluator runs, against an **interleaved control at 0 of 6 on the same card**. A
mechanical slot-overlap audit of every instantiation was **clean**, and the gather index was provably
in range -- so the layout audit cannot find this. **And a stale lane that happens to land IN range is
a silent wrong value, not a fault**, which makes this a live candidate for accuracy failures in a
build that is already scoring.

**The sanctioned fix** -- once per kernel, never inside a loop:

```c
AICORE inline void scalar_to_vec()        // after any PIPE_S instruction whose result a Vec op reads
{
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}
```

`PtoSetWaitFlag<PIPE_S, PIPE_V>()` is the library equivalent and is what a correct kernel in this
repo already uses. Verified: 0 faults in 9 consecutive runs against a measured 50% null (p = 0.2%).

**Why it survives every normal gate:** it is **launch-history dependent** -- invisible in a fresh
process and invisible to single-shot sweeps (1463 configs missed a sibling instance of the same class
of bug). Catching it needs an **allocator-interleaving determinism probe**: many reps in one process,
no `del` between them, hashing the output bits each rep.

**Audit recipe.** Grep for a `PIPE_S` instruction followed only by a `PIPE_V` barrier:

```bash
grep -rnE '(^|[^A-Za-z_])(TCI|TRESHAPE|SETFMATRIX|SET_IMG2COL[A-Z_]*) *[<(]' --include='*.cpp' --include='*.h' .
```

then **check the overload** (a 2-argument `TCI` is scalar; a 3-argument one is not) and **check what
reads the tile**. A `TRESHAPE` whose result is consumed by a Cube/MTE1 op is a different pipe pair and
is not covered by this rule -- confirm the consumer before converting anything.

### C123: fp16 / bf16 GO INTO THE CUBE **NATIVELY** -- UPCASTING TO fp32 FIRST IS A DEFECT  🟡

**Verified in the pinned library**, `a2a3/TMatmul.hpp` `CheckStaticMad`'s `static_assert` accepts:

```
(int32_t, int8_t,      int8_t)
(float,   half,        half)          <-- fp16 operands, fp32 ACCUMULATOR
(float,   float,       float)
(float,   bfloat16_t,  bfloat16_t)    <-- bf16 operands, fp32 ACCUMULATOR
```

So a 16-bit operand already accumulates in fp32. **Converting it to fp32 before the Cube buys no
accuracy and costs throughput**, and the library's own cost model says how much
(`costmodel/a2a3/cce_costmodel/cce_costmodel_cube.hpp`, `mad()`):

```c
constexpr uint64_t kCyclePerRepeatForFloat = 2;
int cycle_per_repeat = 1;
if (std::is_same_v<dtype_a, float>) cycle_per_repeat = 2;   // float is penalised
const uint64_t kTiles = CeilDiv(k, 32 / sizeof(dtype_a));   // and gets a SMALLER k-tile
```

fp32 pays `cpr = 2` with `kTile = 8`; fp16 pays `cpr = 1` with `kTile = 16`. Cycles scale as
`cpr * kTiles`, i.e. `k/4` against `k/16` -- **a 4x predicted difference for the same shape.**
Cross-check: int8 gets `cpr = 1`, `kTile = 32`, predicting 8x over fp32, which matches an
independently measured int8/fp32 ratio on another operator.

**This figure is a COST-MODEL PREDICTION, not a device measurement.** Before quoting any speedup,
run a two-arm on-device microbenchmark at identical shape -- fp16 operands against fp32 operands,
same fp32 accumulator, with a null control -- and confirm the fp16 arm still clears the operator's
accuracy gate.

**Worked defect:** `kernel_lstm.cpp` routes every weight and input through **9 `TCVT` sites** into
fp32 buffers in its PREP phase and declares **every** Cube tile as `float`
(`L1Mat<float,...>`, `TileLeftF<float,...>`, `TileRightF<float,...>`), so it instantiates **only**
the all-fp32 triple. **8 of its 20 cases are fp16 or bf16** and pay the 4x for nothing -- and its
golden is an fp32 CPU computation anyway, so there is no accuracy argument for the upcast.

**Second-order effect, and it couples to COOK-24:** fp32 operands **double the weight footprint**
versus fp16, which can push a loop-invariant weight that would fit L1 out of it. So an upcast can
convert a residency win into a per-iteration GM reload.

### C124: A GRID-STRIDE LOOP WHOSE **ITEM COUNT** IS BELOW THE BLOCK DIM RUNS ON A FRACTION OF THE DEVICE  🔴 **CRITICAL**

**No correctness gate can see this**, and no `block_dim` change fixes it.

A grid-stride loop of the form

```c
for (int32_t it = lane; it < nItem; it += lanes) { ... }
```

executes on `min(nItem, lanes)` lanes. If `nItem < lanes`, the surplus cores are **idle by
construction** -- every case passes, the score simply never reflects the hardware.

**Worked defect:** `kernel_lstm.cpp:422` computes `nItemW = q.NU * nRB` (directions x
`ceil(batch/16)`). Across the 20 declared cases that product is **1 for fifteen of them and 2 for
the bidirectional ones, against a `block_dim` of up to 24** -- measured independently as *all
recurrence work running on `vid == 0` of 1-4 active cores out of 24*. The four-gate axis, up to 1024
rows, is a **serial loop inside the single active core**.

**MAKE IT A MECHANICAL DETECTOR**, in the same spirit as the accepted-set-equals-exercised-set rule:
**enumerate the item count the loop produces for every shape in the declared contract and compare it
against the block dimension.** A declared shape whose item count is below the block dim is a finding.

**And record the counter-pressure, because this is DETECT-AND-DECIDE, not always-split.** Splitting a
second axis across blocks when the consumer needs all partials costs a **per-step grid barrier**,
which collides with the cookbook's guidance against grid-barrelling a per-tile seam and with the
`SYNCALL` block-dim cap (**C66**/**C57a**: a barrier over-subscribed past the device's core count
deadlocks). So the detector reports; the decision is separate.

**Corollary: raising a `block_dim` cap buys NOTHING while the item count is the binding constraint.**
Measured on lstm -- `block_dim` forced to 2/4/8/16 all *lose*, best per-case +0.071, i.e. the existing
value is already optimal because the item count, not the cap, is what limits occupancy.

### C122: PTO's **b32 index gather** ALIASES ITS OFFSET TABLE ONTO THE DESTINATION  🔴 **CRITICAL**

`pto/npu/a2a3/TGather.hpp` has two paths, and **only one of them is safe**:

```c
// b32 (lines 58-61): offset table written INTO dst, then gathered FROM dst -- SAME ADDRESS
vmuls  ((__ubuf__ int32_t  *)(dstPtr + i*TShape1), ...);   pipe_barrier(PIPE_V);
vgather((__ubuf__ uint32_t *)(dstPtr + i*TShape1), (__ubuf__ uint32_t *)(dstPtr + i*TShape1), ...);

// b16 (lines 73-76): offset table written into tmpPtr, gathered FROM tmpPtr -- distinct buffers
vmuls  ((__ubuf__ int32_t  *)(tmpPtr + i*TShape1), ...);   pipe_barrier(PIPE_V);
vgather((__ubuf__ uint16_t *)(dstPtr + i*TShape1), (__ubuf__ uint32_t *)(tmpPtr + i*TShape1), ...);
```

**`pipe_barrier(PIPE_V)` is not sufficient on the b32 path**, and it fails two ways:

1. An **intermittent** `"The address for the VEC instruction to read/write UB is out of bounds"`
   vector-core trap (`subErrType:4`).
2. **SILENTLY WRONG ANSWERS.** On `engram_gate_fusion` the decode path was wrong on every item
   carrying more than one hyper-connection channel: at `HG=2` exactly `hc in {1,3}`, at `HG=4`
   exactly `hc in {1,2,3}` -- **never `h == 0`**. A shape the kernel *accepted* returned
   `MERE 1.60 / MARE 7.2e2`, and at `B=256`, `MARE 2.2e4`.

**CORRECTED 2026-10-06 by direct measurement on A2. Two things I had wrong:**

**(a) The BARRIER is the main effect, not the aliasing.** 2x2 arms, `validCol=64`, `validRow=1`,
40 launches each, bit-exact against an fp64 oracle with positive and negative controls:

| | `pipe_barrier(PIPE_V)` | `pipe_barrier(PIPE_ALL)` |
|---|---|---|
| **aliased dst** (the library's b32 path) | **37/40 wrong** | 0/40 |
| **distinct buffer** | **8/40 wrong** | 0/40 |

`PIPE_V` is insufficient **even to a distinct buffer**; aliasing only amplifies it. **So the b16
branch is NOT proven safe** -- it uses a distinct `tmpPtr` but still only `PIPE_V`, and remains
**unmeasured**.

**(b) `PIPE_ALL` ALONE HAS A WIDTH CLIFF.** Aliased + `PIPE_ALL`, `validRow=1`, 30 launches per width:

| validCol | 128 | 192 | 256 | 320 | 384 | 512 | 640 | 768 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| wrong launches | 0 | 0 | 0 | 1 | 1 | 0 | 4 | **17** | **30/30** |

**Banding restores it**: band-to-256 and band-to-64 are **0/30** at 640, 768 **and** 1024. And the
library shows banding was intended and abandoned -- `TGather.hpp:45-48` computes `numRepeatPerLine`,
`numLoop` and `remainAfterLoop`, then **never uses them**.

**THE VALIDATED RECIPE** (for `validRow == 1`; measured **0 wrong in 490 launches, widths 1-1024**):

```c
set_mask_count();
for (each band of <= 256 b32 elements) {
    vmuls  (dst_band, idx_band, elem_bytes);   // offset table into dst
    pipe_barrier(PIPE_ALL);
    vgather(dst_band, dst_band, src);
    pipe_barrier(PIPE_ALL);                    // after EACH band
}
set_mask_norm(); set_vector_mask(-1, -1);
```

plus a **2 KB zeroed tail** past the end of the UB map, zeroed **once at kernel entry before the work
loop** and never written after. **Both halves are necessary**: band256 + `PIPE_ALL` with the tail
ablated gives **9/200 wrong** (first failure at `validCol=31`) against **0/200** with it.
`dst`, `src` and `idx` must be **three distinct regions** -- `dst` is destroyed mid-call.

**Residual failures above the band trace ENTIRELY to `validRow > 1`** (widths >=64 at vr 1-4 give
26/200; the same widths at vr=1 give 0/200). A `validRow > 1` gather is a separate, unmitigated
hazard.

**Caveats:** the 256 threshold is measured at vr=1 on A2/910B2 only and the boundary is
**probabilistic, not a hard cliff** (512 read 0/30 while 384 read 1/30) -- treat 256 as a
measured-safe working value. A barrier-only fix suffices **only** where every gather is <= 256 b32
elements wide.

**NO CALLER-SIDE STATIC CHECK CAN FIND THIS CLASS**, because the aliasing lives inside the library:
`tools/scan_tile_aliasing.py` reports **0 sites across all of `submissions/`** while 48 exist. The
only static signal is **"Form A call + b32 payload"**. Note Form A branches on the **payload** width
(`src0`/`dst`), never the index width -- `src1` is b32 by `static_assert` -- and on the b32 branch
`tmpPtr` is computed and **never dereferenced**, so passing a distinct tmp buys nothing.

**THE TEST THAT FINDS IT, because every normal gate misses it:** a **same-shape repeat loop cannot
see this class** -- the stale residue must come from a *different* address map. A 150x repeat of the
faulting shape ran clean while the wheel faulted on its first call. **Use an INTERLEAVED multi-shape
soak** (many distinct shapes, cycled, in one process) and poison/zero-check the tail. It also passes
every single-shot sweep, which is why it survived three scored 20/20 runs.

**When writing a gather:** prefer the **tmp-taking overload with a tile you own** so the offset table
never aliases the destination, verify the tmp is actually distinct from `dst`, and treat any b32
in-place gather as requiring the `PIPE_ALL` + zeroed-tail treatment above. Related: **C119** (the
other `PIPE_V`-is-not-enough case, there a `PIPE_S` producer) and **C121** (an aliasing hazard whose
"disjoint" claim was really about timing). The pinned pto-isa has now produced **three**
silent-wrong-answer defects -- see also **C113** and **C118** -- so **read the header for any
library op you rely on.**

### C121: A UB MAP'S "DISJOINT" COMMENT IS A **TIMING** CLAIM, NOT AN ADDRESS CLAIM  🔴 **CRITICAL**

Two UB slot `#define`s can hold the **same address** and still work -- for as long as no single
kernel has both live at once. The source comment then records a *timing* invariant while reading
like an *address* invariant, and the next edit silently breaks it.

Measured on `unique`: `FA_S1` and `FA_OGW` are **both 131072**. The UB map's own comment said a tile
was "disjoint from FA_S1/FA_S2 users" -- **false of `bst`**, which is both an `FA_S1` user (the
`m > F32CAP` single-valued probe) and a tail-block user in phase E. They had only ever avoided each
other because the two phases never overlapped. Widening a cap made them overlap, and the device
reported:

```
MTE accesses an invalid GM address        (aivec core 20)
```

**It faulted only because one visible case happened to drive the oversized branch. With a different
case mix it would have shipped as a silent wrong answer.**

**The guard: tie every aliased slot to the extent that must not reach it, with a `static_assert`**,
so a later edit cannot re-create the overlap:

```c
// FA_OGW aliases FA_S1 at 131072.  Anything living in FA_S1 while a tail block is live
// must fit below the alias point -- assert it rather than asserting it in a comment.
static_assert(FA_S1_OFFSET + BST_BYTES <= FA_OGW_OFFSET,
              "bst in FA_S1 would overlap the FA_OGW tail block");
```

**And when you widen any cap, re-derive which slots are simultaneously live** -- a cap increase is
exactly the edit that turns a timing-safe alias into an overlap. Related: **C119** (the other class
of bug a static audit cannot see -- there a synchronisation gap, here an aliasing gap that only a
*liveness* analysis finds). Note a mechanical slot-overlap audit **passes** both: it compares
declared extents, and these are declared not to overlap.

### C120: A PRECISION-RECOVERY **SPLIT** BREAKS NON-FINITE SEMANTICS  🔴 **CRITICAL**

**`TAXPY` binds its scalar to the SOURCE TILE's dtype** -- `pto/npu/a2a3/TAxpy.hpp` declares
`TAXPY_IMPL(..., typename TileDataSrc::DType)`, so a `half` source **forces a `half` scalar** and you
cannot pass an fp32 weight against an fp16 tile.

The standard workaround is to recover the lost precision by **splitting the scalar and accumulating
twice**: `w = wh + wl`, then `acc += wh*v; acc += wl*v`. **That is algebraically equal for finite `v`
and WRONG for `v = +/-Inf`:**

| golden | ours (split) | |
|---|---|---|
| `w*Inf` = `Inf` | `wh*Inf + wl*Inf` | |
| | **NaN when `wl < 0`** | `Inf + -Inf` |
| | **NaN when `wl == 0`** | `0*Inf` |

Measured on `roi_align`: a minimal `[2,16,16,16]`, `oh=ow=1`, `samp=1`, single grid point, no
merging -- golden **0 NaN / 14 Inf**, ours **14 NaN / 0 Inf**. *Every* position where the reference
returns a signed Inf returns NaN instead. The class is **fp16 x any Inf**, independent of
`sampling_ratio` and `aligned`: on a 144-config non-finite surface, `[-inf,inf]` 36/36 DIFF,
`[0,inf]` 36/36, `[-inf,0]` 36/36, and `[nan,nan]` **0/36** (NaN absorbs, so a NaN-only probe
cannot see it).

**Guarding only `wl == 0` is NOT the fix** -- built and measured, still failing, because `wl < 0` is
the larger half. Either handle the non-finite case before the split, or do not split.

**Why it hides.** It needs a **16-bit** tile AND an **Inf** input AND a split-accumulate path. A
visible case with fp32+Inf never splits; a visible case with fp16+NaN absorbs. **So check any
split/compensated accumulation (Kahan, Veltkamp, two-pass, `wh+wl`) against +/-Inf explicitly** --
`static_assert`s and finite-data batteries are all blind to it. Related: **C106** (the Vec pipe does
not fuse, which is *why* these splits get written) and **C115** (judge the fix by whether its failure
set is a strict subset).

### C118: `TSUBS` ON AN int32 TILE ROUNDS ITS SCALAR THROUGH fp32 -- A LIBRARY BUG  🔴 **CRITICAL**

`pto/npu/a2a3/TSubS.hpp:22` implements scalar subtract as an **add of the negated scalar, negated
through `float`**, regardless of `T`:

```cpp
vadds(dst, src0, (T)(-(float)src1), repeats, 1, 1, 8, 8);   // (T) == int32_t here
```

So on an `int32_t` tile **any `|scalar| > 2^24` is silently rounded to fp32 precision**. No error, no
fault. The threshold is exact and was measured: a scalar of **16777217 subtracts 16777216**.

| scalar | `(int32)(-(float)s)` | error |
|---|---|---|
| 16777216 | exact | 0 |
| **16777217** | -16777216 | **+1** |
| 1294967298 | -1294967296 | **+2** |
| 1294934303 | -1294934272 | **+31** |

**`TADDS` is EXACT** -- `TAddS.hpp:22` passes `src1` straight through (`vadds(dst, src0, src1, ...)`).
**The two are therefore NOT interchangeable on integer tiles**, which is the whole trap: the obvious
reformulation is the fix.

**The sanctioned fix -- add the WRAPPING negation:**

```c
// EXACT int32 scalar subtract.  Never TSUBS on an int32 tile.
AICORE inline void subs_i32_exact(/*tile*/ dst, /*tile*/ src, int32_t s)
{
    // unsigned so INT32_MIN is not signed-overflow UB; modulo-2^32 is what a key map wants
    TADDS(dst, src, (int32_t)(0u - (uint32_t)s));
}
```

**Carry it as a kernel-wide invariant -- "no `TSUBS` on an int32 tile"** -- and state it at every call
site, because a later reader will "simplify" it back.

**Why this hides.** It needs an integer input whose **minimum exceeds 2^24**, which no low-magnitude
test range reaches, and its signature is a **uniformly shifted result** (every value off by the same
small amount) rather than noise -- which reads like an indexing-base bug, not an arithmetic one. On
`unique` it also produced a **device fault**: the shifted value became a negative index into an
unclamped gather, giving `"The address for the VEC instruction to read/write UB is out of bounds"`
plus a vector-core exception. **A silent arithmetic defect can present as a hardware fault.**

**How it was localised, which is the reusable method:** instrument the kernel to **publish its own
measured key space** and compare across passes. Both passes reported identical, correct bases
(`fin_vmin == p1_vmin`), which ruled out the measurement and left the subtraction itself; host
arithmetic then reproduced both observed shifts (+2 and +31) exactly from the measured bases.

**`TDIVS` is NOT the same bug.** `TDivS.hpp:104,132` does contain
`T inv = (T)((float)1 / (float)src1)`, but it sits inside
`if constexpr (std::is_same<T, float> || std::is_same<T, half>)`, so it never applies to an integer
tile and going through `float` there is correct. Checked first-hand against the pinned library; do
not propagate it as a suspect. Related: **C113** (`SetContinuousMask`) -- the pto-isa library has now
produced two silent-wrong-answer defects, so **read the library source for any scalar-taking op you
use on an integer tile.**

### C116: THE SORT FAMILY IS ONLY **POSITIONALLY STABLE** -- THE COMPANION IS **INERT** IN THE COMPARE  🔴 **CRITICAL**

**The PTO doc for `TSORT32` states "ties broken by smaller index first". Measured with positive and
negative controls: FALSE as stated.** The whole chain -- `vbitsort` **and** the batched 4-way
`vmrgsort4` merge tree -- is merely **positionally stable**, and the uint32 companion **takes no part
in the comparison**:

- 32 tied keys carrying **descending** companions come back in **input order**, not companion order.
- A 4-way merge of four all-tied runs **drains list 0 entirely before list 1**.

The doc's claim is an artifact of the companion normally being an **ascending iota**, which makes
positional stability *look* like an index tie-break. (8 tests.)

**Consequence for design: a tie-break rank cannot be PACKED into the companion -- it can only be
CARRIED.** Any ordering you need among equal keys must be imposed by your own pass after the sort
(e.g. adjacent compare-exchange to convergence on `(key DESC, companion ASC)`), not delegated to the
sort. A kernel that relies on the documented tie-break is wrong on every tied key, and tied keys are
exactly what a `top_k`-style hidden set probes.

**Verify a tie-break claim before building on it**: feed tied keys with *descending* companions and
check the output order, and keep a negative control that proves your own re-order pass actually fired
(an `#ifdef` that leaves the object byte-identical proves the guard is inert).

### C117: `TAND` / `TANDS` ARE **1- AND 2-BYTE ONLY** ON A2/A3 -- A 32-BIT MASK WILL NOT COMPILE  🟡

A 32-bit bitwise AND has no vector form on `dav-c220`: `TAND`/`TANDS` **`static_assert` on 1- or
2-byte element types**. To extract a low bit-field from a b32 lane, use shift arithmetic instead:

```c
// low b bits of x, without a 32-bit AND
x - ((x >> b) << b)
```

Related shape-of-problem: an **arithmetic** right shift (`TSHRS` by 31) yields a lane-wise sign mask,
which is how to turn a non-overflowing int32 difference into a `>` predicate without `TCMP` and
without any mask register.

### C115: A **PARTIAL** ROUNDING MATCH IS NOT MONOTONE -- JUDGE BY **SUBSET**, NOT BY COUNT  🔴 **CRITICAL**

When a case is scored against a **lossy** reference on a near-zero denominator, passing it means
**reproducing the reference's exact rounding sequence**. That makes the match **all-or-nothing per
output**: fix one rounding and leave a second mismatched, and the result lands on a **third** value
that can be **further** from the reference than the original was.

So a candidate arm with **fewer** failures can be the **worse** arm. Measured on `grid_sampler_3d`
over 2016 configurations:

| arm | failures | failure set vs control | **new** |
|---|---:|---|---:|
| control | 40 | -- | -- |
| **one rounding matched (SHIPPED)** | **21** | strict **subset** | **0** |
| + a second rounding matched | **4** | **not** a subset | **4** |

The 4-failure arm introduced four failures that did not previously exist, all in the parameter
region where the first rounding was *already* exact. The same shape appeared on `roi_align`.

**The acceptance test is a CENSUS, not a total.** For every candidate, emit `fail->pass` and
`pass->fail` counts against the full prior battery. **Ship only an arm whose failure set is a strict
subset of the current one.** `pass->fail > 0` rejects the arm even if its total fell.

**Two corollaries that cost real sessions:**

1. **Enumerate every rounding the reference performs before matching any of them, from its SOURCE.**
   Include roundings the *compiler* introduces: `aarch64` **contracts** `a*b - c` into a single
   `fmadd`, so `((coord+1)*size - 1)/2` is **one** rounding in the reference and **two** in a
   `TMULS`-then-`TADDS` kernel. The formula is identical; the rounding count is not.
2. **Look for an exactness CONDITION, not an exactness LIST.** Here the 2-rounding chain equals the
   true fma **iff `size` is a power of two** -- 0 mismatches at sizes 1,2,4..4096 and 0.24% of
   coordinates otherwise, over 3.69e9 evaluations. A list of observed-good values would have hidden
   that and mispredicted every unlisted size.

**Being MORE accurate is not the axis.** An exactly-rounded sum with the wrong unnormalize still
failed ~4%. See **C106** -- the Vec pipe has no FMA (`vmla`/`TMULA` is byte-identical to a separate
`vmul` then `vadd` and rounds the product first), so an exactly-rounded fma must be **emulated**: a
Veltkamp split plus a **Knuth 2Sum**, 17 vector ops, 0 mismatches in 3.77e9 evaluations. Dropping
the 2Sum fails 965 of 3.69e9 -- rare enough to read as clean on any small probe.

### C113: `pto::SetContinuousMask(128)` RETURNS A **64-LANE** MASK -- A LIBRARY BUG  🔴 **CRITICAL**

`pto-isa/include/pto/common/utils.hpp:32` builds the high mask word as

```c
(n > MASK_LEN) ? (((uint64_t)1 << (uint32_t)(n - MASK_LEN)) - 1) : 0      // MASK_LEN == 64
```

At `n == 128` the shift amount is **64** -- undefined behaviour -- and AArch64's variable shift takes
the amount **mod 64**, so it evaluates `1ULL << 0 == 1`, subtracts 1, and the high word is **0**.
**`SetContinuousMask(128)` enables only lanes 0..63.** Measured: every `n` in 1..127 is correct and
`n == 128` **alone** is wrong.

**Where it bites:** any op that asks for 128 lanes. On one kernel a duplicate-index gather broadcast
used `lanes = (sizeof(T)==2) ? 128 : 64`, so only **fp16** reached the bug; lanes 64..127 of every
128-lane group were never written, the broadcast left **stale UB**, and the consumer read it. Silent
wrong answers, no fault, 72 of 1116 configs.

**The boundaries are a function of the REPLICATE WIDTH, not of a single constant** -- on that kernel
they were `C >= 17` for the bilinear path (width `4*CPT >= 128`) and `C >= 65` for nearest (width
`CPT`). Two different boundaries, and `C=17`/`C=32` had been called clean by every prior sweep.

**Rule: never pass 128 to `SetContinuousMask`.** Use a UB-free local setter for any `n` that can
reach it:

```c
AICORE inline void safe_mask(int n)          // n in 1..128
{
    const uint64_t all = 0xffffffffffffffffULL;
    if (n >= 128)      set_vector_mask(all, all);
    else if (n > 64)   set_vector_mask((static_cast<uint64_t>(1) << (n - 64)) - 1, all);
    else if (n == 64)  set_vector_mask(static_cast<uint64_t>(0), all);
    else               set_vector_mask(static_cast<uint64_t>(0), (static_cast<uint64_t>(1) << n) - 1);
}
```

**Audit every call site for its reachable maximum `n`, and do NOT patch the library.** ~10 other PTO
ops call `SetContinuousMask`, so a library edit widens the blast radius and breaks every per-kernel
bit-identity proof. A local setter at the one exposed site is the sanctioned fix. For reference,
`TCvt` can never reach 128: `ComputeTCvtRepeatConfig` gives `elementsPerRepeat <= 128` and
`TCVT_IMPL` passes `validCol % elementsPerRepeat <= 127`.

**The signature to recognise:** element 0 of each sub-block correct and 1..n-1 wrong is a *broadcast*
signature, and a defect that tracks a **replicate width** rather than a tile size or a cost-model
branch points at the mask, not the arithmetic.

### C111: A MASK TILE'S WRITE GRANULE IS A **64-LANE REPEAT**, NOT `ceil(validCol/8)` BYTES  🔴 **CRITICAL**

A compare writes its mask LSB-first per byte, so the naive size for `validCol` lanes is
`ceil(validCol/8)` bytes. **That is wrong and it silently overruns.** Measured over an `0xAA` poison:

| validCol | bytes actually written |
|---|---|
| 8 / 16 / 32 / 63 / 64 | **8** |
| 65 / 128 | **16** |
| 192 | 24 |
| 256 | 32 |

So the requirement is **`8 * ceil(validCol/64)` bytes** -- up to **8x** the naive figure at small
widths. A mask tile sized `ceil(validCol/8)` leaves whatever follows it in UB to be overwritten,
with no fault and no diagnostic. Size every mask slot by the 64-lane rule, and keep the mask's GM
transfer on a raw `*_align_b8` so exactly one Tile names that address (C62).

**The dst dtype is not free either:** `vcmp_dispatch`'s destination is `__ubuf__ uint8_t*`
(`TCmps.hpp:23`), so a mask tile **must** have `DType uint8_t` -- a `uint16` view does not compile.
That one is a compile error rather than a silent bug, which is the only reason it is cheap.

**Worked error worth repeating:** the first probe of this asserted `ceil(n/8)` and reported a
"FAILURE" at n=8 and n=16. **The probe was at fault, not the hardware** -- the third time in one
campaign that a probe rather than a mechanism was wrong. A probe is an instrument; validate it with
a poison pattern and a known-size control before believing a negative.

### C112: COMPARES ARE **ORDERED** (NaN -> FALSE IN EVERY MODE), AND `TSELS` SELECTS SRC ON A SET BIT

Both measured with positive and negative controls, on two unrelated operators, and they agree.

**`vcmpvs_*` is ordered.** On `[nan,1,-1,inf,-inf,0,nan,2]` vs `0.0`, **every** mode -- EQ, NE, LT,
LE, GT, GE -- returns **FALSE** for the NaN lanes. Controls: `finite GE 0.0` alternates as expected,
`GE +INF` is all-0, `GE -INF` is all-1. This reproduces the independently measured top_k result that
**`CmpMode::NE` returns FALSE for NaN**, so `mask = (x != x)` **detects nothing** and is never a NaN
test on this part.

Two idioms that follow, both verified:
- **`GE -INFINITY` is 1 for every finite value AND both infinities, 0 for NaN only** -- the NaN test.
- **`(GT -INF) AND (LT +INF)` is 1 iff finite** -- the finite test.

**`TSELS(dst, mask, src, tmp, scalar)` is `dst[i] = maskbit(i) ? src[i] : scalar`** -- the mask
selects true-lanes from the **first source**, confirmed from a host-built mask (so the test is not
circular), with all-0 -> all scalar, all-1 -> all src, and a single set bit selecting exactly its one
lane. **In-place `TSELS(c, mask, c, tmp, k)` is SAFE** -- identical to out-of-place. Do not
generalise from `TRECIP`, which IS silently wrong in place; aliasing has to be probed per
instruction, not assumed either way.

## C58: A PREPROCESSOR `#if` ON A C++ TEMPLATE PARAMETER IS SILENT AND WRONG -- USE `if constexpr`

The preprocessor runs before templates exist. A template parameter is **not a macro**, so the
preprocessor reads it as `0`:

```cpp
#elif MHC_COLBCAST && (MP >= BLK)     // MP and BLK are TEMPLATE PARAMETERS
```

evaluates as `0 && (0 >= 0)` -- and since `0 >= 0` is **true**, the branch fires for every
instantiation. Inside it, `MP/BLK` then computes `0`, so a divisor was never written.

**Why it is dangerous rather than merely broken:** it validated **4 of 20**, and the two cases the
change was written to target showed a clean **1.18x speedup**. A wrong branch that is fast on the
cases you are looking at, and NaN on the ones you are not, is the worst possible failure shape --
cases 1-8 came back all-NaN while the change looked like a win.

**Rule: never put a template parameter inside a `#if` / `#elif`. Use `if constexpr`.** If you need
a compile-time branch on a template value, `if constexpr (MP >= BLK)` is the construct; the
preprocessor form compiles, links, runs and is silently wrong.

This is a member of the same family as C51 (a guarded `__global__` declaration deletes the launch)
and C53 (the wrong arch flag deletes every Cube instruction): **a preprocessor condition that is
false for a reason you did not intend, producing a clean build and wrong output.** When a
`#if`-guarded change behaves strangely, print the macro's value with `#warning` before debugging
the code inside it.

## C59: `TEXTRACT` LAYS L0A OUT AT THE **STATIC** `Cols` UNLESS `CompactMode::Normal` -- AND THE GATE CAN PASS ANYWAY

`TEXTRACT` writes its L0A fractal using the tile's **declared** `Cols`, while `TMATMUL` issues
`k = GetValidCol()`. If those disagree and the tile is not `CompactMode::Normal`, the contraction
reads the wrong stride. Silent: no fault, no compile error.

Measured relative error against a CPU reference, by output width:

| `Wo` | rel err |
|---|---|
| 16, 32, 48, 64 (multiples of 16) | 2.9e-04 |
| 17, 20, 30, 31, 33, 61 | **6.8e-02 … 3.8e-01** |

**The part that matters beyond the bug.** On `level3/conv_3d_backprop_filter`, case 8 **PASSED the
official accuracy gate carrying this defect** -- MERE 2.99e-04 against a 9.77e-04 threshold. With
the defect fixed the same case reads **6.08e-07: a 490x accuracy improvement entirely invisible to
pass/fail.**

That is a failure mode beyond rule 3.23's. 3.23 says a scored case may not *reach* your bug. This
is worse: **the case reaches the bug, computes a materially wrong answer, and still passes**,
because the threshold is loose enough to absorb it. A green gate is not evidence that the
contraction is right.

**Rules:**

1. **Put `CompactMode::Normal` on any boxed tile whose `ValidCol` can differ from `Cols`**, and
   `static_assert` the pairing rather than relying on it.
2. **Probe with data that can actually fail.** The run's own first probe used **all-ones operands**
   and could not detect a stride error at all -- every wrong element summed to the same value. Use
   distinct, position-dependent values (a ramp, or random) so a mislaid fractal shows up.
3. **When accuracy is comfortably inside threshold but not near machine epsilon, ask why.** A
   contraction that should be exact to ~1e-7 sitting at 3e-4 is a signal, not a pass.

## C60: A DIRECT `<<<>>>` LAUNCH CAN OVERTAKE A PRECEDING TORCH OP -- THE HOST TASK QUEUE IS NOT A STREAM

torch_npu dispatches through a **host task queue**. A `ctypes`/direct `<<<blocks, nullptr,
stream>>>` launch issued after a torch op does **not** reliably observe that op's writes, even on
the same stream.

Measured on a three-kernel composed operator:

| edge | wrong results |
|---|---|
| `torch -> ours` | **40 / 40** |
| `ours -> ours` | 0 / 40 |
| `ours -> torch` | 0 / 40 |

It presents as **order-dependent NaN across cases that each pass in isolation**, it survives
`torch.npu.empty_cache()` and a device change, and it **vanishes under
`ASCEND_LAUNCH_BLOCKING=1`** -- which is the diagnostic, because that is a host-side serialisation
knob.

**Rule: put a device sync on every `torch -> direct-launch` edge.** The cost is host-side only, and
cann-bench scores the sum of kernel `Duration`, so it does not enter the measurement at all.

**Why it went 27 ops without being seen:** every previous operator in this campaign was a single
kernel, so there was never a torch op between two of our launches. It appears the moment an
operator is composed from more than one launch -- which is exactly what a staged pipeline
produces.

### C60a: `TCVT` float -> float IS A SILENT NO-OP

A same-type `TCVT` does not copy. Every fp32 configuration of one operator returned an **exactly
zero** output while every fp16/bf16 configuration passed -- because the fp32 path used `TCVT` where
the narrow paths needed a real conversion. No compile error, no fault, output simply never written.
Use an explicit move/copy when source and destination types are the same, and never let a dtype
template instantiate a conversion that is an identity.

## C61: MTE2 BURST COST IS A FUNCTION OF **BASE ALIGNMENT AND STRIDE**, NOT JUST LENGTH -- AND THE TERMS INTERACT

Three measured effects on the same operator, and the third is why a naive "round the burst up"
optimisation can go backwards:

| change | effect |
|---|---|
| `stride % 32 != 0` | **2.6x** slower |
| round `lenBurst` up to 32 B, **over a 32 B-aligned base** | **3.7x** faster |
| the same rounding, **over a MISALIGNED base** | **2.2x WORSE** |

So length, stride and base alignment are not independent knobs. Rounding the burst length is only
a win once the base is aligned; applied blind it is a regression.

**Rule: fix base alignment first, then stride, then length.** And when you build a cost model for
DMA, separate those three terms explicitly -- the run that found this lost an attempt (0.79-0.90x)
to a first probe that **conflated length with stride**.

**Related hard constraints on the same path:** a misaligned UB base for a Vec op is a **hard
fault** at 32-byte granularity regardless of dtype, and there is **no `copy_ubuf_to_ubuf_align`** --
the GM->UB load is the only byte-granular shifter on the part, so a misaligned UB layout cannot be
repaired in place.

### C61a: `pipe_barrier(PIPE_ALL)` DOES NOT ORDER A VEC/MTE2 UB WRITE AGAINST A SCALAR READ

C19 is stronger than it is written. `pipe_barrier(PIPE_ALL)` is not sufficient for a scalar read of
a UB location that a Vec or MTE2 instruction wrote: removing the explicit flag sandwich made two
cases **deterministically** wrong, 3 runs out of 3. Deterministic, not flaky -- so it will survive
a re-run and look like a logic bug.

Use the explicit `set_flag`/`wait_flag` pair for that edge; do not substitute a barrier.

## C62: TWO TILES `TASSIGN`ED TO THE SAME UB ADDRESS -- THE COMPILER REORDERS THEM, AND `pipe_barrier` DOES NOT STOP IT

`pipe_barrier(PIPE_V)` is a **hardware** barrier. It is not a **compiler** barrier. bisheng's tile
dependence analysis treats operations naming **different `Tile` objects** as independent -- so if
two objects are `TASSIGN`ed to the **same UB address** (a sub-tile overlay), the optimiser will
reorder across them.

Bisected on a real kernel, gate relative error:

| build | result |
|---|---|
| `-O0` + `pipe_barrier(PIPE_V)` | exact |
| `-O1` + `PIPE_V` | **wrong (~1e0)** |
| `-O2` + `PIPE_V` | **wrong (~1e0)** |
| `-O2` + `pipe_barrier(PIPE_ALL)` | exact |

**Three things make this worse than an ordinary aliasing bug:**

1. **It survives `-O1`**, so it is not something you find by turning optimisation down one notch.
2. **It is silent** -- no fault, no warning, just a wrong answer.
3. **It is LATENT.** The run that found it had a `PIPE_V` build passing 20/20 twice -- but only
   because an unrelated optimisation attempt had changed *which objects alias*. Passing today is
   not evidence the kernel is safe tomorrow.

And the construct is not exotic: **sub-tile overlay is an idiom this cookbook actively
encourages** for UB budgeting.

**Rule: when two `Tile` objects can name the same UB address, use `pipe_barrier(PIPE_ALL)` between
their uses**, not `PIPE_V`. Cost measured at **1.0-2.4%**. If you want the cheaper barrier, first
prove the objects cannot overlap -- and re-prove it whenever the UB map changes, because this is
exactly the kind of invariant a later optimisation attempt breaks without touching the barrier.

**Related but distinct:** C61a says `pipe_barrier(PIPE_ALL)` is *insufficient* for a scalar read of
a Vec/MTE2 write (use an explicit `set_flag`/`wait_flag`). So `PIPE_ALL` is necessary here and not
sufficient there -- the two rules constrain different edges and neither substitutes for the other.

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
   - **An EMPTY constraints list means NOT DOCUMENTED, never "unconstrained".** The
     response carries `documented: true|false` — read it. When `false`, the real
     constraints may still live in the prose of `get_instruction()`, in
     `get_cpp_intrinsic()`, or as `static_assert` checks in the pto-isa headers,
     which are the authority. Treat it as an evidence gap and expect the compiler to
     find what the tool did not.
     (Historical note: before the MCP fix pinned in `.mcp.json`, this list was empty
     for 98 of 161 instructions including `TEXTRACT`, `TMOV` and `TCVT` — four
     generation runs read that as "unconstrained" and two lost repair attempts to it.
     It is now 5 of 161, but the rule above still holds for those five.)
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
| `const event_t e = EVENT_ID1;` or `constexpr event_t` | C1x | "the 3rd parameter must be a type 'event_t'" -- drop the `const`, or use `auto` |
| Scalar loop instead of `TLOAD`/`TSTORE` to move a TILE | C1 | Orders of magnitude slower (4 B per issue vs a whole tile). NOT a crash -- a scalar `__gm__` read of a runtime scalar is legal, see C1 |
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


## STRONG-FORM TRIGGER: every operand read exactly once -> TRY THE L2 ALIAS FIRST

`PLAT-§L2Bypass` was written around a streamed weight in a grouped matmul and is hard to reach
from a plain Vec kernel. That is a reachability defect, because across this campaign the L2
alias has been **the single largest optimization on nearly every memory-bound kernel**:

| case | alias effect |
|---|---|
| `rms_norm_backward` | **1.697x** |
| `gemma_rms_norm` | 1.473x fp32 / 1.518x fp16 |
| `deep_norm` | 1.472x |
| `rotary_mul` | 1.451x |
| `dynamic_quant` | 1.389x |

In several of these the alias is **the entire result**. `rms_norm_backward` on the cached path
measures 228.3 us against the vendor's 226.0 -- **1.010x SLOWER** -- and ships at 1.648x
FASTER purely from the alias. `dynamic_quant` is 0.9932 (parity) without it.

**So make it the FIRST thing you try, not something you discover.** Trigger:

> **If a GM operand is read exactly once per kernel invocation and never re-read by another
> lane, probe the L2 alias on it before any other optimization.**

Then apply the qualifiers that took this campaign several runs to learn:

* **"Read once" means CACHE LINES, not elements** (`PLAT-SS-LineReuse`). A row narrower than
  the line shares it with a neighbour read on the next iteration -- aliasing there measured
  **1.35x SLOWER**. Check the access granule against the 128 B line.
* **Only ACROSS-lane reuse needs L2.** Within-lane reuse is served by L1/UB, so a small hot
  table can still be a good alias candidate (measured +2.5%).
* **Writes can go either way** -- +3.4% in one case, a loss in another. Probe both directions.
* **Probe in the FINAL configuration; isolated readings do not compose.** One case measured a
  tensor 1.018x faster alone and 1.015x slower on top of the winning alias.
* **Always carry an alias-off control** built from the same binary via a runtime flag, and
  verify the output is **bit-identical**.

**Report the cached-path number alongside the aliased one.** If the kernel is at parity cached
and fast aliased, say so -- that is the honest description of what was achieved, and it is a
statement about the memory system rather than about the code.


### ALIAS FIRST, THEN TUNE -- cached-path tuning decisions do NOT transfer

A corollary of the L2-alias trigger, and it changes the *order* of the optimization campaign.

`quantize` measured its shipped configuration on the **cached** path as indistinguishable from
baseline (49.04 us vs 48.69 us -- nothing). The **same changes on the aliased path** were worth
**1.078x** and **1.035x**. The tuning was invisible until the alias was in place.

The mechanism is simple once stated: while the kernel is stalled on L2 miss-and-allocate, it is
not limited by anything you are tuning, so every knob reads as noise. Remove that stall and the
real bottleneck -- packing width, saturation path, tile shape -- becomes measurable.

**Therefore: land the alias FIRST, then run the rest of the campaign on top of it.** A campaign
that tunes on the cached path will conclude, correctly and uselessly, that none of its knobs
matter.

This is the same lesson as "probe in the final configuration; isolated readings do not
compose", sharpened into an ordering rule. Two decisions in that run flipped on it: an
output-alias read **1.002x** in isolation and **1.036-1.044x** in the final configuration, and
a fast-saturation path read **1.4%** at pack=1 and **3.4%** at pack=2.


### A kernel must DERIVE its work count on-device, never take it from the host

Extension of **C18**. A host-computed `total_work` with a hard-coded tile count silently
disagreed with the kernel's actual geometry the moment a compile-time knob changed: every
`NB=1` variant computed **160 of 288 work items** and reported success.

**The invariant C18 asks for cannot survive a compile-time geometry knob if the count is
supplied from outside.** Whenever the work count is derivable from the shapes and the tiling
constants, the kernel must compute it itself. Export it (`kernel_ntile()`) so the harness can
*assert* agreement rather than *supply* the value.

Pair it with output poisoning on the harness side -- the two together turn this from a silent
wrong answer into an immediate failure.

### Macro-blocking is DTYPE-DEPENDENT: it is net-negative for fp32 Cube

`COOK-§8.7B`'s macro-blocking is worth **3.2x** on the fp16 kernel it was written from. On an
**fp32** Cube kernel it is **net-negative**: fp32 doubles every L0 tile, so a 2-tile
macro-block starves the double-buffering that actually pays. Measured: `NB=2` gives 159.36 us;
`NB=1`, with the freed L0 spent on an extra buffer, gives 156.59 us and then buys a further
**1.060x**.

**On A2, L0 capacity is the binding constraint, and fp32 halves how many tiles fit.** Re-derive
the blocking factor from the *actual* tile footprint in the kernel's dtype rather than
inheriting a factor tuned at fp16.

### Padding to alignment can beat saving the arithmetic

Removing a **12.3% arithmetic overhead** from padding (1026 -> 1152 columns) made the kernel
**1.43-1.96x SLOWER**. The padded shape keeps every tile on its aligned fast path; the
"efficient" unpadded shape pays a ragged-tail penalty far larger than the MACs it saves.

**Do not remove padding on an arithmetic-count argument alone -- measure it.** Charge the
padding overhead honestly to your own numbers (this run did), but expect removing it to lose.


### C28(c) EXTENDED -- it is the whole TROWEXPAND* family, and the dispatch is on src1 LAYOUT

`C28(c)` was written about `TROWEXPANDMUL`. It applies to the **entire `TROWEXPAND*` family**,
and the mechanism is a **silent dispatch on the src1 layout**:

> `TRowExpandBin` selects its path from `src1`'s layout. A **RowMajor** `src1` silently takes a
> **32 B/row elementwise** path instead of the broadcast you asked for -- and the library's own
> `PTO_ASSERT` **accepts it**, so nothing fires.

Cost when it happened: `y = inf` everywhere, on the first kernel of the run.

Two further undocumented constraints found alongside it, both hard `static_assert`s or silent
failures:

* **ColMajor / NoneBox requires `Rows * sizeof(T)` to be 32 B aligned.**
* **`SetValidShape` requires BOTH valid dims to be DYNAMIC.**

And one performance cliff: **crossing the 64-wide single-repeat threshold costs 2.03x.** Keep
the reduction width inside one repeat where the shape allows it.

### TROWARGMAX returns the LOWEST index on ties (probed, undocumented)

Probed 8/8, including across the internal group boundaries of a two-stage reduction. This is
load-bearing whenever a selection must be order-preserving: it is what makes a two-level top-k
provably match a flat one. It is not documented anywhere -- verify it again if the pto-isa pin
moves.


### SHAPE CONTRACT: kernels are DYNAMIC, but an out-of-contract shape must FAIL LOUDLY

Generated kernels take the problem dimensions as **runtime arguments** -- one `.so` serves the
whole sweep (measured: a single `reshape_and_cache` binary covers T=14 to T=65536, a 4680x
range). Do not specialize a binary per shape.

But dimensions fall into **three tiers**, and the tier must be recorded in the contract:

| tier | example | why |
|---|---|---|
| **free** | rows, tokens, batch, sequence count | swept at runtime, no constraint |
| **constrained** | `S % 128 == 0`, `N % 512 == 0`, `K*N % 64 == 0`, `HD % 16 == 0` | MTE burst granularity, or a tile multiple |
| **equal to a tile constant** | `D == 128` | Cube L1/L0 fractal tiles are **template parameters** (`Tile<..., Rows_, Cols_, ...>`), so the tile SHAPE is compile-time even though its *valid extent* can be DYNAMIC |
| **capped** | `N <= 24576` | the reduction row must fit one UB tile |

**The defect to avoid:** most generated kernels guard with a bare early return --

```cpp
if (dimS <= 0 || dimD != kTile || (dimS % kTile) != 0) return;   // SILENT
```

-- and their host wrapper returns `void`. **An out-of-contract shape therefore produces a
silent no-op**, and the output buffer keeps whatever was in it. Combined with a recycled
allocator block (see the harness rule on output poisoning) that is precisely the false-pass
that once reported *"OK, 1.941x FASTER"* for a kernel computing 55% of its work.

**Required:**

1. **The host wrapper returns `int`**, not `void`, with a distinct negative code per violated
   constraint. `dynamic_quant` does this (`-1`/`-2`) and is the pattern to copy.
2. **The harness checks the return code** and fails the case, rather than proceeding to compare
   an untouched buffer.
3. **Every constraint is recorded in the contract as a locked/amended dim** with its reason --
   never silently rounded or substituted.
4. **Validate at least one out-of-contract shape** and assert the kernel *rejects* it. A guard
   nothing tests is a guard you do not have.


## CORRECTION to "ALIAS FIRST, THEN TUNE": it is LAYOUT FIRST, THEN ALIAS

The alias-first ordering rule was derived from a fused attention kernel with **contiguous**
scratch. It is **wrong for a strided operand**, and the failure is a sign flip, not a magnitude
error:

| operand layout | burst | L2 alias effect |
|---|---|---|
| native ND, 32 B bursts | 32 B | **0.283x — a 3.5x REGRESSION** |
| native ND (strided) | 128 B | **0.916x — a REGRESSION** |
| native ND (strided) | 256 B | **1.313x — a win** |
| repacked (contiguous) | -- | **1.410x — the best** |

Same alias, same kernel, same data: **the sign of an alias result depends on the burst length it
was measured at.** An alias probed on a badly-laid-out operand can send you away from the alias
*and* leave the layout unfixed.

**Corrected ordering:**

1. **LAYOUT first** -- fix the operand's memory layout (see the optimizer skill's layout axis).
2. **ALIAS second** -- re-probe the cache-bypass alias *on the fixed layout*.
3. **TUNE third** -- tile widths, macro-blocking, block_dim, buffering.

**They are not independent knobs -- the effects are SUPERADDITIVE.** Measured on the same case:
layout alone **1.111x**, alias alone **1.313x**, both together **1.652x**. Probing either in
isolation understates it and can invert its sign.

**The regression is larger than first recorded.** A later probe measured **0.283x at 32-byte
bursts** -- a 3.5x loss, not the 0.916x first reported. Probing the alias before fixing the layout
would have measured 0.283x and abandoned an axis that is worth **1.5x** once the burst is right.

**Reporting requirement:** every alias measurement must state **the burst length it was run at**.
An alias number without a burst length is not reproducible and, as above, is not even
sign-stable.


## CORRECTION: the alias sign tracks SCHEDULE REDUNDANCY, not burst length

An earlier rule here claimed the L2-bypass alias' sign depends on the **burst length** it is
measured at. A controlled follow-up shows **burst length and redundancy were confounded** in that
finding, and redundancy is the real variable:

**Burst held constant is not the discriminator.** At a fixed schedule redundancy of 2.0x, a
**128x change in burst length** (256 B -> 32,768 B) moved the alias by **0.6%** and **never
flipped its sign** (1.0222x vs 1.0286x).

**Redundancy is.** Across operands in one kernel:

| operand | schedule redundancy | alias effect |
|---|---|---|
| `a` (intermediate) | 2.0x | **1.064x — win** |
| `x` | 5.0x | 0.944x |
| `W1` | 32x | 0.884x |
| `W2` | **64x** | **0.655x — heavy loss** |

**The rule: the alias pays when the schedule re-reads an operand FEW times, and costs when it
re-reads it MANY times.** That is the same principle as `PLAT-SS-LineReuse` (a bypassed fill is
wasted only if nothing reads it again) — now with a measured knob. A useful crossover from
another case: the alias regressed at 2.56x reuse and only helped once reuse fell below ~1.9x.

**So: report SCHEDULE REDUNDANCY with every alias measurement** (burst length remains worth
recording, but it is not what decides the sign). And probe the alias **per operand** -- in the
kernel above the correct answer was to alias exactly one of four.

### C66 -- NEVER HARDCODE A CORE COUNT IN A HOST PLANNER; QUERY THE DEVICE

`torch.npu.get_device_limit(dev)` is a public torch_npu export (backed by
`_npu_get_device_res_limit`) returning the part's core counts. On the A2 910B2 these
kernels are developed against it returns:

    {'cube_core_num': 24, 'vector_core_num': 48}

Those two numbers are **exactly** the literals that host planners keep hardcoding -- a
block_dim capped at `min(48, ...)` for Vec work, `min(24, ...)` for Cube work. They were
never chosen by measurement; they are the development part, written down.

**What it costs.** `masked_scale`, kernel object byte-identical
(`035b3f9268c1df84d8144289cbf25712`), submitted twice to cannbench's A3 910c:

| planner cap | score | geomean speedup |
|---|---|---|
| hardcoded `48`            | 83.35, 83.45 | 1.771 |
| `get_device_limit(...)`   | **87.95**    | **2.340** |

**+4.50 points from one constant.** The A3-vs-A2 slowdown fell 1.434x -> 1.089x and 9 of
20 cases became FASTER on A3 than on the part the kernel was tuned for. Three operators
had each lost ~5 points on A3 (masked_scale -5.20, exp -4.77, moe_gating -5.25) and that
suspicious uniformity across a memory-bound elementwise op, a transcendental one and a
sort/selection op was the tell: a shared constant, not three schedule artifacts.

**The rule.** Derive the cap from the device, cache it once per process, and fall back to
the historical constant if the API is unavailable so a missing query is never worse than
hardcoding:

```python
_CORES = {}
def _vector_cores(fallback=48):
    n = _CORES.get("v")
    if n is None:
        n = fallback
        try:
            v = int(torch.npu.get_device_limit(
                torch.npu.current_device()).get("vector_core_num", 0))
            if v > 0:
                n = v
        except Exception:
            pass
        _CORES["v"] = n
    return n
```

Use `cube_core_num` for Cube-side caps and `vector_core_num` for Vec-side ones. A MIX or
multi-stage operator needs BOTH -- `conv_2d` caps its Cube stage at 24 and its two Vec
stages at 48, and all three are this property.

On the development part the query returns the constant it replaces, so the change is
provably a no-op there: same cap, same block_dim, same measurements. It only ever differs
where the constant was already wrong.

**THREE WAYS THE QUERY SILENTLY DOES NOTHING.** Rolling C66 across 13 operators found three
distinct failure modes, and a correct query defeated by any of them looks identical to "C66
does not help for this operator":

1. **Dead code (the entry-point trap).** A `TORCH_LIBRARY` registration makes the Python
   driver unreachable -- cann-bench's `load_ai_operator` tries `torch.ops.cann_bench` FIRST.
   A Python-side query on such an operator never runs. Caught by a null result:
   `moe_gating_top_k_softmax` moved **+0.12** while module-path operators moved +3.29 and
   +4.50. Decide per operator which path is live and put the query there.
2. **Unreachable (compiled into device code).** A `#define` baked into the kernel object,
   used inside the in-binary planner, cannot be reached from any driver. That was `gather`'s
   **-3.61**: `#define GK_MAXBLK 48` used four times inside `gather_plan()`, with the driver
   passing `block_dim = 0`. Requires a kernel rebuild, not a host patch.
3. **CLAMPED AWAY (the one that looks most like success).** The host query runs, returns the
   right number, passes it in -- and an unconditional clamp inside `call_kernel` reverts it:

       if (blocks > 24) blocks = 24;      // silently undoes any larger host value

   `grouped_matmul` and `grouped_matmul_swiglu_quant` both had this. Even an explicit
   `block_dim = 32` came back as 24. A driver-only fix is invisible here **and leaves no
   trace** -- no error, no warning, correct results, unchanged score.

**The fix for (3) is a named setter, not a repurposed hint slot:** keep
`static uint32_t <op>_core_cap = <historical>;` in the kernel TU, export
`extern "C" void <op>_set_cube_cores(uint32_t)` that ignores 0, have every clamp site read
the variable, and push the queried value once from the live path (`std::call_once` in the
plugin, or lazily in the Python driver's kernel-wrapper constructor).

**And every clamp site must read it, including the workspace sizer.** `kernel_gmsq.cpp` held
the cap TWICE -- once in `call_kernel` and once in `kernel_workspace_bytes`, which sizes
`bd * per-core scratch`. Raising only the launch clamp would overrun the workspace on a part
with more AIC cores: a memory-safety bug, not a performance miss. Push the value BEFORE the
workspace is sized, from a single query authority, so host and device agree by construction.

**Verify the query is actually linked, not just written.** `nm -D --defined-only` on the built
`_C.abi3.so` should show your setter as `T`, and the object that queries should show
`U c10_npu::GetDeviceResLimit(int, int)`. One tree's "ABI anchor" TU compiled to a 768-byte
object with **no symbols at all** -- its anchor pointers had internal linkage and were
discarded -- so the exports came from elsewhere and the anchor was decorative.

**A host-only change must leave the kernel object byte-identical; a CCE-TU change will not,
and that is not evidence of a device-code change.** `call_kernel` is host code living inside
the CCE translation unit, so editing it changes the object. Compare the embedded
`.aicore_binary` section instead -- it was byte-identical across all three operators here.
(Recompiling a reconstructed source under a DIFFERENT filename shifts that section by 8 bytes,
because the device ELF embeds the source name; compare under the original filename.)

## C67-C71: SURVIVING THE CASES YOU WERE NOT SHOWN  🔴 **CRITICAL**

These five rules all come from one discovery: the scorer runs a **hidden case set 4x the
size of the visible one**, and it deliberately probes the edges the visible set omits.
Twelve operators have now been through it. Six passed cleanly; six failed, and **every
failure was one of the five classes below** -- none was a novel numerical problem.

**Why any of this matters enough to be a rule: the scoring is multiplicative in the pass
fraction.** It is not "19 of 20 cases score and one scores zero". Every term scales:

    score = (20 + 30) * k/N + 50 * (sum HAP_i)/N

So a single case that REJECTS or faults costs up to **5 points**, and `strided_slice` shows
the compounding: 6 real faults cascaded into 29 skipped cases and took an 84.32 operator
to **49.94**. Conversely `conv_2d` sits at 33.09 because 43 of 80 cases never produced a
number. Correctness robustness is not hygiene here; it is the largest available score lever.

**Read the failure prefix FIRST -- it says whose fault it is:**

| prefix | meaning |
|---|---|
| `Golden执行失败` | the BENCHMARK's reference crashed. Not your bug. |
| `AI算子执行失败` | your kernel or driver failed. |
| `精度不达标` | both ran; the comparison rejected. |

`moe_finalize_routing` lost 6 cases to the first kind (the hidden generator shaped
`skip1`/`skip2`/`bias` for a different `num_rows` than `expanded_permuted_rows`, so our
kernel was never invoked) and it is worth ~87 rather than 80.72 for reasons nobody can fix
from the kernel side. Diagnosing that as our defect would have burned a whole repair budget.

### C67 -- DERIVE THE SUPPORTED SET; NEVER ENUMERATE IT, AND NEVER FIT ITS GRANULARITY

A kernel whose supported shapes are a **lookup table** is not a dynamic-shape kernel, it is
a shape-specialised one with a rejection path -- and it will reject almost everything the
hidden set asks for.

`sparse_flash_attention` scored **70.66 on the visible 20 and 0.00 on the hidden 80.** All
80 failed identically: `shape out of kernel contract`, raised by our own driver. The cause
was `sfa_config()`, a hardcoded table of exactly ten `(dk, dv, g, topk, bf16)` tuples
returning -1 for anything else. The visible 20 contain exactly ten distinct configs. The
contract had been **fitted to the cases it was shown**. Deriving the tiling from the dims
instead moved it 0.00 -> 51.77 for 1.35 points of visible score.

`conv_2d` is the same class one size down: `if (khkw != 1 && khkw != 9 && khkw != 25)
return -1;` -- only 1x1, 3x3 and 5x5 kernels exist as far as that kernel is concerned.

**And deriving a bound is not sufficient -- the bound's own granularity must be derived
too.** After the sfa rewrite, `Dk=96` still rejected with rc=-6, because the agent had
bounded Dk to "64-aligned, in [64,768]" and probe-confirmed that 96 rejects, recording that
as correct behaviour. The 64-alignment was itself fitted: every visible case happened to use
a multiple of 64. The hidden set uses 96. Same mistake, one level down, after the lesson.

> **C67 IS THE PRINCIPLE; C72 IS THE CHECK.** This rule has been violated *after being
> followed* -- sfa derived its tiling and then fitted the very next bound (`Dk` 64-aligned) from
> the visible cases. Do not rely on remembering C67: run the C72 declared-surface gate and put
> its combination/rejection counts in the report.

**The test that catches this is not "do out-of-contract shapes reject cleanly".** sfa's
generating run verified exactly that and reported `6/6 out-of-contract shapes rejected with
a negative rc`. Rejecting cleanly is correct behaviour for a genuinely unsupported shape and
worthless when the unsupported set is everything the scorer will test. **Ask instead: is the
SUPPORTED set wide enough, and does every bound in it trace to the declared family in
`desc.md`/`proto.yaml` rather than to an observed case?** Enumerate the declared family, not
the cases.

**Corollary for performance switches.** `strided_slice` picked its DMA mode on
`s_in*esize <= 64`. That constant was not a correctness bound, it was a fitted performance
cutoff -- measured crossover is 256-512 B, so it was 4-8x too low and cost 1.18x-2.9x across
the whole 66-256 B band. When a cutoff must exist, derive it from a cost model
(`nburst*A + rbytes*B`, argmin) so the boundary is a **consequence** of measured constants,
and prefer the form whose error is symmetric: a better-FITTING model was rejected there
because one grid step early cost 9-97% while one step late cost 1-17%.

### C68 -- NO TORCH COMPUTE OP ON THE KERNEL PATH; THE IMAGE DOES NOT SERVE EVERY aclnn OP

The evaluation image is not your dev box. These calls each returned
`JSON configuration file ... cannot be found` (error 561103) on the runner while working
locally: `ZerosLike`/`aclnnInplaceZero`, `aclnnCast`, `aclnnMuls`,
`aclnnInplaceCopy_1_Broadcast`.

Confirmed cost: `nms` and `foreach_norm` each scored **0/20** on a `torch.zeros(device='npu')`
in the driver. `exp` lost 3 of 80 hidden cases to a tiny-tensor fallback that used
`torch.exp`/`.to(dtype)`/`*scale`. `gcd` lost hidden case 100 to a `.contiguous()` that
lowered to a broadcast copy. `grouped_matmul` had `gl.to(torch.int64)` on a device tensor.

**The safe pattern** is `torch.empty` plus one H2D `.copy_()` from a CPU tensor, or padding
to a legal size and slicing the result. Never a broadcast copy -- `copy_(x.expand(...))`
lowers to the very op that killed gcd, so an obvious-looking fix can trade one missing-op
failure for another.

**Two traps around this rule:**

1. **A `TORCH_LIBRARY` registration makes your Python driver dead code.** cann-bench's
   `load_ai_operator` tries `torch.ops.cann_bench` FIRST. If the plugin registers a torch op,
   everything in `cann_bench/__init__.py` never executes, so a Python-side fix is invisible.
   Decide per operator which path is live, and put host logic in C++ when the torch op wins.
   This is why a ZerosLike sweep of the Python drivers came back clean while three operators
   still had the bug in their C++ plugins.
2. **"No visible case reaches it" is not a reason to leave a fragile fallback.** exp's
   fallback carried the comment *"never reached by any benchmark case -- the smallest is
   1 MB"*, true of the visible 20 and false of the hidden set. That exact reasoning, written
   as an operator property, has now cost five separate operators.

### C69 -- DEGENERATE SCALAR ATTRS: A FOLDED CONSTANT CAN BE NaN WHERE THE GOLDEN IS FINITE

Hidden cases vary the **scalar attributes** too, not just tensor shapes and magnitudes, and a
degenerate scalar puts algebraically-valid folding in an IEEE corner.

`apply_adam_w` folded the AdamW update into `k1*m + k2*g`. Valid while the bias correction
is finite. At `beta1 == 1.0` the correction `1 - beta1^step` is **exactly zero**, so
`ihat1 = +Inf` and `(1 - beta1) = 0`, giving `k2 = 0 * Inf = NaN` -- and one NaN constant
poisons every output element. The golden never distributes: it forms the numerator first
(`m + 0*g == m`, finite) and divides once, yielding +-Inf.

**The signature is unmistakable and invisible to any magnitude-based gate:**
`MERE = MARE = 0.000000` with the comparator failing on `NaN位置不匹配` (NaN positions).

> **CORRECTED 2026-09-26 -- do NOT read those zeros as "every finite element is bit-exact".**
> `compare.py:390-398` returns a `CompareResult` carrying the DATACLASS DEFAULTS the moment the
> NaN masks differ; the metrics are never computed. So `MERE = MARE = 0.000000` here means
> *"the comparison exited before measuring"*, not *"the error was zero"*. The same applies to
> `total_count: 0` on that output -- it is an artifact of the early exit, NOT an empty tensor.
> I misread both on `dequant_swiglu_quant` and briefly concluded its `scale` output was empty.
> (Inf-position mismatches behave differently: they are saturated to max-finite and the
> comparison continues, so only a NaN mismatch produces this hard early exit.)
> The practical consequence is that this signature tells you the non-finite PATTERN differs and
> tells you NOTHING about the finite values -- you must measure those separately before claiming
> the arithmetic is fine.

`sparse_flash_attention` and `conv_2d` each have a case in this class too.

**NOT EVERY INSTANCE IS A FOLD.** `conv_2d`'s bf16 NaN cases were a silent RANGE loss: prep
re-encoded bf16 as fp16 on the reasoning that bf16's 8-bit significand fits fp16's 11. The
significand fits; **the exponent does not** -- bf16 carries fp32's 8-bit exponent (max ~3.4e38)
against fp16's 5-bit (max 65504). Every `|value| > 65504` became an fp16 Inf, and `Inf + -Inf`
is NaN where the golden is finite. The measured threshold sat between 1e4 and 1e5, i.e. exactly
65504. Running bf16 natively (`(float, bfloat16_t, bfloat16_t)` is a documented a2a3 TMATMUL
triple, and TIMG2COL takes `bfloat16_t`) fixed it -- and the original "exact re-encode" premise
was falsified outright: the two paths are BIT-IDENTICAL on in-range data and the native path is
also faster, because it deletes an fp32 staging pass. So when you see this signature, check for
a narrowing conversion as well as a fold.

**So: before folding a constant, ask what it evaluates to at each endpoint of every scalar
attribute's declared range** -- `beta1/beta2 in {0,1}`, `epsilon = 0`, `step = 0`, `lr = 0`,
`scale = 0`. Where the folded coefficient is an indeterminate form, the correct value is
whatever the UNFOLDED expression gives (here the gradient term is absent from the golden's
numerator, so the coefficient is 0, not NaN):

```c
const double g_coef = (1.0 - beta1);
c.k2 = (g_coef == 0.0 || !std::isfinite(ihat1))
           ? 0.0f
           : static_cast<float>(u * g_coef * ihat1 / s2);
```

### C70 -- A DEVICE FAULT INSIDE AN ACCEPTED SHAPE IS NOT A REJECTION, AND IT COSTS ~30 CASES

`507035` is `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION` and `507015` is its neighbour; torch_npu
surfaces them with **"timeout" wording, which is not what they are** -- they are an AIV
exception, not a hang. The device names the real cause if you ask it: *"The UB address
accessed by the VEC instruction is not aligned"*, subErrType 4.

**Two structural facts about how this appears in a hidden report:**

- **A cluster of 3 consecutive failing cases means a shape FAMILY, not scattered numerical
  error.** strided_slice faulted at 32-34 and again at 95-97; conv_2d at 52-54 and 72-74.
- **Everything after a fault is collateral.** strided_slice reported 35 failures of which
  only **6** were real; the other 29 read `device unrecoverable, case skipped` and never
  executed. Count the real faults before sizing the defect, and never report a cascade
  length as a failure count.

The concrete ISA cause found here, previously undocumented: **`vreducev2` requires a
32-byte-aligned UB destination.** Its compaction loop chunked the repeat at 255 (the hardware
field is 8 bits) and re-based the destination by `done*outper` UNITS per chunk; with
`outper == 1` and b16, `done = 255` lands 510 bytes past a 32 B base. Note the fix is NOT a
255 cap -- that is a fitted bound (C67). Chunk in quanta that keep the destination aligned
for ANY mode and repeat:

```c
const int q    = 32 / gcd(outper * ubytes, 32);
const int rmax = (255 / q) * q;
```

Also: `vreducev2`'s repeat is declared `uint16_t` in the headers but the hardware field is
**8 bits**. A wide declaration is not a licence to pass > 255.

### C71 -- PROBE THE DECLARED DTYPE'S EXTREMES, NOT THE OBSERVED DATA'S

`gcd` concluded that an int32 overflow path was unreachable because *"INT32_MIN occurs zero
times in the generated data for all eight int32 cases"*. True of the visible 20. The hidden
set drove it to ~1e18 and to INT_MIN, and the operator went 20/20 -> 70/80.

Two distinct findings there, both worth carrying:

- **Magnitude.** int64 was computed in 32-bit lanes because `vconv_s642s32` saturates.
  Visible cases top out at |value| 100000; hidden cases reach 1e18. Supporting the declared
  dtype means real 64-bit arithmetic -- and note `int32 TADDS` **saturates** rather than
  wrapping, which kills the unsigned-compare-via-bias idiom, so 4x16-bit limbs in int32
  lanes is the practical representation.
- **The golden can be wrong at the extreme and it is still the spec.** `torch.gcd(INT_MIN, b)`
  returns the correct magnitude with a sometimes-NEGATIVE sign, an overflow artifact of its
  Euclid loop. Our kernel returned the mathematically correct non-negative value and FAILED.
  Reproducing an artifact is legitimate work when the artifact is the scored reference;
  verify it is deterministic first (C-Euclid emulation matched it exactly for b=1..128).

**Operationally:** sweep each input to the limits of its **declared** dtype, and each scalar
attr to the endpoints of its declared range, before claiming any path unreachable. A local
probe driving cann-bench's own `DataGenerator` and `compare_tensors` at those limits
reproduced all three gcd families for **zero credits**, and found two the paid hidden run did
not reveal. Build the probe before spending a submission.

**Finally, if your kernel writes a status word, your driver MUST read it.** gcd's kernel set
a sticky GM flag when a tile exceeded INT32_MAX and neither the harness nor the wheel driver
ever looked, so it returned silently wrong answers instead of rejecting -- the exact class
C51/C53 exist to prevent. Reading it is three lines and converts silent-wrong into loud-fail.

## C72: THE DECLARED-SURFACE GATE -- ASSERT YOU ACCEPT EVERYTHING THE SPEC DECLARES  🔴 **CRITICAL**

C67 tells you to derive the supported set instead of enumerating it. That is a *principle*, and
a principle is something a run has to remember. **C72 is the mechanical check**, and it exists
because C67 alone did not hold: `sparse_flash_attention` was rewritten to derive its tiling and
its very next bound (`Dk` "64-aligned") was fitted again, one level down.

**Every hidden-set failure in this campaign that was OUR fault has had the same shape: the
implementation covers the VISIBLE cases, and the scorer tests the DECLARED surface.**

| operator | narrowed to | the spec declares | hidden result |
|---|---|---|---|
| `sparse_flash_attention` | a 10-tuple config lookup | derived from dims | **0 / 80** |
| `conv_2d` | `khkw in {1,9,25}` | general convolution | 37 / 80 |
| `cross_entropy_loss` | `target` int64 only | int32, int64, fp32, fp16, bf16 | 63 / 80 |
| `gcd` | 32-bit lanes | int64 | 70 / 80 |

### None of the other gates can see this class

- A **contract sweep** samples the cases the benchmark happens to ship. It cannot know what the
  document allows.
- **C44's out-of-contract probe asks the wrong question.** It verifies that *unsupported* shapes
  reject cleanly. `sparse_flash_attention` passed it **6/6** and scored **0.00** on all 80
  hidden cases. Rejecting cleanly is correct behaviour for a genuinely unsupported shape and
  worthless when the unsupported set is everything the scorer will test.
- **C67 is judgement.** It has already been violated *after being followed*, in the same run.

### The gate

`skillyard-cannbench/tools/declared_surface_probe.py`. It parses `proto.yaml`'s per-input
`dtype` lists and attr enums/defaults (plus `desc.md`'s dtype matrix, which is often stricter
and more explicit), forms the **declared cross product**, and calls the operator once per
combination on minimal tensors. It checks only one thing:

> **does the operator ACCEPT every combination its own spec declares?**

No golden, no accuracy comparison, no benchmark run, no credits -- so it is cheap enough to be
unconditional. A combination that RAISES is a finding with exactly two legitimate outcomes:
implement it, or write down why the spec is wrong. "No visible case uses it" is not one of them.

    python3 declared_surface_probe.py <task_dir> <op_name> --install <isolated_install_dir>

**IT MUST VARY ONE AXIS OFF A KNOWN-GOOD BASELINE, AND THE FIRST VERSION OF THIS TOOL DID NOT.**
That version synthesised minimal tensors (a fixed 4x8) and passed only the attrs that declare an
enum or default. Swept across 45 operators it reported **~100% rejection**, and every one was an
artifact: `TypeError: missing 1 required positional argument` for attrs like `dim`, `strides`,
`perm`, `num_groups`, or a rank error from the invented shapes. A 100% rejection rate reads like
a catastrophic finding and was pure noise.

The working design takes a **real case from the task's own `cases.yaml`** -- valid shapes, every
required attr present -- builds its tensors with cann-bench's own `DataGenerator`, **asserts that
baseline is ACCEPTED first** (printing `SETUP_FAILED` and claiming nothing if not), and then
varies ONLY the declared dtype axis on top of it. Every rejection is then attributable to the
dtype, because everything else is held at a configuration the operator demonstrably accepts.

**Both controls, on the rewritten tool:**

    POSITIVE  cross_entropy_loss  baseline case 1 ['float16','int64'] ACCEPTED
                                  15 combinations, 12 REJECTED
                                  -> target must be int64 hard labels, got Int
    NEGATIVE  softmax             baseline case 1 ['float16'] ACCEPTED
                                  3 combinations, 0 REJECTED

**Never trust a gate you have not run against a known positive AND a known negative.** Three
scans in this campaign returned falsely clean results for want of a positive control, and this
tool's first version returned falsely alarming ones for want of a valid baseline.

### The dtype lists are allowed SETS, not independent axes

`proto.yaml` gives a dtype list per input. Cross-producting them independently generates
combinations **the spec forbids**, and their rejection is correct behaviour. Most operators
additionally require the float inputs to share one dtype, stated only in `desc.md` prose:

    apply_adam_w   "var、grad、m、v 四个张量的 shape 和 dtype 必须完全一致"
    maximum        "两个输入张量的 dtype 必须一致"
    rms_norm       "gamma 的 dtype 需与 x 一致"

The first fleet sweep reported **16 operators with rejections and 15 of them were this** --
`apply_adam_w` alone showed 78/81 "rejected", every one a mixed-dtype combination the spec never
declared. The probe now reads `desc.md` for that constraint and skips mixed-float combinations
when it is present, while still varying INTEGER inputs independently -- which is what keeps
`cross_entropy_loss`'s int32/soft-label hole visible (its matrix deliberately mixes
`float32 | int32`). After the fix: `apply_adam_w` 3 combinations, **0 rejected**, 78 skipped;
`cross_entropy_loss` unchanged at 12/15.

**The lesson generalises past this tool: a "declared" surface is the DOCUMENT'S cross product,
not the one your parser finds easiest to build.** Read the prose constraints and the dtype
matrix, not just the per-field lists.

### What the corrected fleet sweep actually found

45 operators, credit-free: **26 CLEAN**, **3 could not establish a baseline** (claim nothing --
`foreach_addcdiv_scalar`, `foreach_norm`, `lstm`), and **one confirmed finding**,
`cross_entropy_loss`. The remaining rejections were spec-conformant mixed-dtype combinations or
artifacts of the probe's own data generation (out-of-range values for `int8`/`uint8`, an
optional-scale argument the int32 path requires). So the class is real and expensive but it is
NOT widespread -- worth knowing before a fleet-wide rewrite is proposed on the strength of a
scary-looking table.

**Its positive control is the failure that motivated it.** On `cross_entropy_loss`, seconds of
runtime, zero credits:

    declared: input {float32,float16,bfloat16} x target {int32,int64,float32,float16,bfloat16}
    15 declared combinations tried, 12 REJECTED
      {'input': 'float32', 'target': 'int32'}
         -> RuntimeError: target must be int64 hard labels, got Int

That is verbatim the message four hidden cases returned in `job_349a3591d562`. The paid run
found it at a cost of one credit and a 62.62; the gate finds it before packaging.

### The rule

1. **Run the gate before shipping, and again after any change to a dispatch, a `TORCH_CHECK`, or
   a supported-set predicate.** Record the combination count and the rejection count next to the
   contract sweep.
2. **Zero rejections, or a written justification per rejection citing `desc.md`/`proto.yaml`.**
   A `TORCH_CHECK` that refuses a dtype the spec allows is a self-inflicted zero, not a safety
   check: every scoring term scales by the pass fraction, so each rejected case costs up to 5
   points.
3. **A declared dtype you have not implemented is a MISSING FEATURE, not an unsupported input.**
   `cross_entropy_loss` never implemented soft labels (`target` float = a probability
   distribution, `-sum(p * log_softmax(x))` rather than a gather) because no visible case used
   them. Read the dtype matrix in `desc.md` as a feature list.
4. **Widen the gate wherever the spec enumerates anything** -- reduction modes, layouts,
   `approximate`, padding modes, `is_causal`. Any attr with a declared enum belongs in the cross
   product, not just dtypes.
5. **Casting a declared integer dtype is not free.** `int32 -> int64` on a device tensor lowers
   to `aclnnCast`, which the evaluation image does not serve (C68). Round-trip through the host.

**Why this is CRITICAL and not STANDARD:** the failure mode is a total or near-total loss on an
operator that passes every local gate at 20/20, and it has now cost this campaign four
operators, one of them entirely.

## C73: A LEGAL NO-OP MUST STILL LAUNCH -- ZERO KERNELS ZEROES THE OPERATOR  🔴 **CRITICAL**

**An operator can be numerically perfect on every case and still score exactly 0.**

`strided_slice` v127, hidden set: `accuracy_passed_cases: 80` of 80, `genuine_pass_rate: 1`,
`result_contract.valid: true`, geomean speedup **2.85x** -- final score **0**. One case did it:

    case 96   status: success          accuracy.passed: true
              total_count: 0           <- the output has ZERO elements
              elapsed_us: 0            op_times: {}
              failure_type: anti_cheat anti_cheat_invalidated: True
              performance_error_msg: 未检测到 NPU 算子执行，疑似 CPU fallback，反作弊触发
                                     ("no NPU operator execution detected, suspected
                                      CPU fallback, anti-cheat triggered")

Hidden case 96 has an **empty output**, so the driver did the obviously correct thing:

```python
y = torch.empty(oshape, dtype=x.dtype, device=x.device)
if y.numel() == 0:
    return y            # nothing to compute -> nothing launched
```

No launch means no NPU kernel in the profile, which the anti-cheat reads as CPU fallback. It
does not merely void that case -- `score_error_code: no_npu_kernel_detected` zeroes
**compilation_score, function_score AND performance_score for the whole operator**. An ~86-point
operator became 0 because one degenerate case correctly did nothing.

### The rule

**Never return from the host without launching.** If the work is empty, launch anyway with
`block_dim = 1` and let the kernel return immediately on-device. One ~4 us launch on a case that
computes nothing is the entire cost, and it is the difference between ~86 and 0.

```python
y = torch.empty(oshape, dtype=x.dtype, device=x.device)
# C73: do NOT early-return on an empty output. A profiled window with no NPU kernel trips
# cann-bench's anti-cheat (no_npu_kernel_detected) and zeroes EVERY score for the operator,
# even though accuracy passes trivially. Launch with a degenerate grid instead.
rc = lib.call_kernel(0 if y.numel() else 1, stream, ...)
```

and on the device side, make the no-work path a launched no-op rather than a skipped launch:

```c
    if (nelem_out == 0) { /* launched, writes nothing */ return 0; }   // NOT a host-side skip
```

### Why no existing gate catches it, and why "it passed" is not evidence of safety

- **Accuracy gates pass it trivially.** `mismatch_count: 0` of `total_count: 0`. The case is
  recorded as `passed: true`; only the *performance* stage flags it.
- **Our own out-of-contract probes treat a clean empty-output return as CORRECT** -- the
  generating run documented "call_kernel returns 0 on success (including the legal empty-output
  case)" as a feature. By every standard except the scorer's, it was.
- **The visible 20 contain no zero-element case,** so this is invisible until a hidden run. Same
  shape as C67/C72: the declared surface includes degenerate extents that the shipped cases omit.
- **A passing score is luck, not safety.** TEN operators in this collection had a
  launch-skipping path -- and the two most exposed are the two BEST results in the campaign:

      exp/cann_bench/_exp.py:149              if n == 0: return torch.empty_like(x)
      masked_scale/_masked_scale.py:141       if n == 0: return torch.empty_like(x)
      gather/_gather.py:80                    if y.numel() == 0: return y
      transpose/_transpose.py:71              if y.numel() == 0: return y
      mish/_mish.py:118 + kernel_mish.cpp     (BOTH layers)
      strided_slice driver + call_kernel x2   (BOTH layers)
      gcd kernel_gcd.cpp:1529                 (via pl.rc == 1)
      foreach_addcdiv_scalar kernel_fa.cpp:441  if (ntot == 0) return 0;
      maximum kernel_maximum.cpp:760            if (n <= 0) return 0;
      arg_max argmax_plugin.cpp:131             rank-0 input returns with no launch

  `exp` is the campaign's only rank-1 operator (93.38, 80/80 hidden) and `masked_scale` its
  highest hidden score (94.55, 80/80). Both passed only because no hidden case had a
  zero-element input. **Fix it in every operator, including the ones that are passing.**

- **A LEADERBOARD ENTRY ALREADY BANKED IS SAFE; A RE-RUN IS NOT.** Because an entry is the best
  FULLY-PASSING run, a later anti-cheat zero cannot displace a posted score. What the defect
  costs is the ability to ever re-run or resubmit that operator, which is exactly what you must
  do to improve it.

- **FIX AT EVERY LAYER.** A driver-side fix alone is usually insufficient: `mish` and
  `strided_slice` each skipped the launch a SECOND time inside `call_kernel`, which is host code
  living in the CCE TU. Removing only the Python early-return leaves the bug.

- **Grep for this pattern across two lines, and use a control.** The condition and the `return`
  sit on separate lines, so a single-line `grep ... | grep return` finds nothing and reports a
  clean fleet. Two scans of mine did exactly that and reported 3 operators instead of 10; one of
  them additionally lost operators to a zsh `nomatch` glob that kills the whole `grep`. Verify any
  such scan against a site you already know exists.

### How to check it locally, for free

Add a zero-element configuration to the contract sweep for **every** operator, and assert that a
kernel actually launched -- not that the result was correct. Profile it, or read back a
launch counter the kernel increments; a correct empty output proves nothing. Any degenerate
extent the declared shape space allows (a zero dim, an empty slice, `k = 0`, an empty segment)
belongs in that sweep.

**Related:** [C51/C53] a status word nobody reads; [C72] the declared surface includes degenerate
extents. The common thread is that **the scorer measures what ran, not only what was returned.**

### Measuring "no regression": the stored baseline is not a control

When you claim a fix does not regress an operator, compare against the **old artifact re-measured
in the same session**, not against a number in the campaign store. Cross-session drift on this
host is ~0.4 points -- comparable to or larger than most effects, and larger than several
operators' own replicate spreads.

The C73 batch is the worked example. Six operators got a change that is provably inert on every
visible case (a guard on a degenerate path no scored case reaches). Five came back **0.16-0.37
below their recorded baselines**, reading as five regressions. A matched same-session control on
`masked_scale`, backed-up pre-fix wheel against the fixed one, three replicates each:

    pre-fix  88.24, 88.25, 88.30   median 88.25
    fixed    88.14, 88.28, 88.60   median 88.28      matched delta +0.02
    recorded baseline (earlier session)   88.65

The old wheel scores 88.25 today as well. So: **back up the pre-change artifact before editing,
and score it beside the new one, serialized, same device, same session.** Report the matched
delta. And when several independent changes all move the same direction by a similar small
amount, suspect one shared cause before believing N separate regressions.

## C74: `TQUANT` GUARDS ITS ALIASING ON THE STATIC WIDTH, NOT THE RUNTIME ONE  🔴 **CRITICAL**

`pto/npu/a2a3/TQuant.hpp` reuses **one buffer** for the fp32 source, the s32 intermediate and the
fp16 intermediate (`TASSIGN_IMPL(src_f16, src.data()); TASSIGN_IMPL(src_s32, src.data())`). The
s32 -> fp16 step therefore aliases a 4-byte-per-element array onto a 2-byte-per-element one **with
different row strides**. The library knows this is dangerous and guards it:

```cpp
constexpr bool kHasTail = (TileDataCvtS32::Cols % kS32ElemsPerRepeat != 0);   // STATIC Cols
if constexpr (kHasTail) { if (TQuantBuffersOverlap(...)) { /* safe row-by-row path */ } }
else { TCVT_IMPL(src_f16, src_s32, ...); }                                     // fast, aliased
```

**`Cols` is the TEMPLATE width. `validCol` is never consulted.** So a tile with a 64-aligned
static width and a non-aligned RUNTIME width takes the unsafe fast path, and the write of one
fp16 block overlaps the s32 read of another.

**Predicate, derived from the stride arithmetic and then confirmed 60/60 on device:**

    corrupt  iff  vc > 128  AND  vc % 64 != 0
    bad columns: [64*floor(vc/64), +32)      i.e. the first 32 of the last partial 64-block
    bad rows:    the LOWER HALF of each row tile

Worked example: `N = 2018 -> N/2 = 1009 -> chunks 256, 256, 256, 241`. Bad columns are exactly
global 960-991 on rows 0-15 and 32-47. This was predicted analytically *before* it was measured,
which is why the predicate is trustworthy rather than curve-fitted.

**Fix at the call site** (do not wait for the library): round the valid column count up to a
multiple of 64, quantise the rounded width, and `TSTORE` only the real columns.

```c
const int vq = ((vc + 63) / 64) * 64;      // static tile width must be 64-aligned
TQUANT(q8, tile(vr, vq), scale);
TSTORE(..., vc);                           // pad lanes are stale UB and are discarded
static_assert(kWC % 64 == 0, "TQUANT aliasing guard needs a 64-aligned static width");
```

**Do NOT pre-zero the pad lanes with a sub-tile `TASSIGN` at `base + vc*esize`.** A `TASSIGN`
byte offset that is not 32-byte aligned faults the Vec: at `vc = 1` that is offset 4 and the
kernel dies with `507015 ... The UB address accessed by the VEC instruction is not aligned`.
The 32-byte requirement applies to the **TASSIGN offset**, not only to tile geometry. The pad
lanes are discarded by the store, so they never need zeroing.

**Why this is CRITICAL:** it is silently wrong on a shape the contract accepts, it hides in the
last partial chunk, and it only touches half the rows -- so a spot check of row 0 passes. It cost
`grouped_matmul_swiglu_quant` 62 of 260 randomised in-contract shapes.

## C74a: `TRowSumOp` HAS A WIDE `FillTmp`; `TRowMaxOp` DOES NOT -- SO ONLY THE SUM OVER-RUNS

The row-reduce scratch rule (scratch width scales with `validCol`, and a ONE-ROW tmp hides the
bug as an out-of-bounds write rather than a wrong answer) has a specific and non-obvious shape:

* `pto/npu/a2a3/TRowSum.hpp :: TRowSumOp::FillTmp` writes `floor(k/2)` repeats at
  `tmp + i*ElemPerRpt`, i.e. it needs `64*floor(floor(validCol/64)/2)` floats -- **not 64**.
* `TRowMaxOp` inherits the **generic** `FillTmp`, which writes one repeat only.

So an identical `[1,64]` scratch tile is safe for `TROWMAX` and overflows for `TROWSUM`. On
`add_rms_norm_dynamic_quant` the fold exits at `64*k` with **k odd** (`sw=7168 -> 448, k=7`;
`sw=8064 -> 4032, k=63`), and the over-run of `(floor(k/2)-1)*256` bytes walked past the scratch
into the per-row accumulators:

    <= 1280 B (k <= 11)   lands in unused space          harmless
    1281-2304 B (k 13-19) lands in the SUM accumulator   scale wrong, y still passes
    > 2304 B (k >= 21)    lands in SUM + MAX accumulators scale AND y wrong, max_diff 255

**41% of the declared width range carried this at every M**, and all 20 visible cases missed it
because only one of them reaches that path at all. The fix that adds no UB: after the log-fold
`len` is a multiple of 64 by construction, so view the remainder as `[len/64, 64]` and reduce in
two `OneRepeatProc` levels -- `OneRepeatProc` does not touch the scratch at all.

**And never hand the library a `validCol > 64` on a one-row scratch tile without checking which
`FillTmp` that operation resolves to.**

## C75: A RETURN CODE READ AFTER AN ENQUEUE IS ALWAYS ZERO  🔴 **CRITICAL**

```cpp
int launch_rc = 0;
at_npu::native::OpCommand::RunOpApi("op", [&]{ launch_rc = call_kernel(...); });
TORCH_CHECK(launch_rc == 0, "...");     // <-- ALWAYS PASSES. RunOpApi ENQUEUES the lambda.
```

`RunOpApi` **enqueues** the lambda onto the task queue and returns; the body has not run when
`TORCH_CHECK` evaluates, so `launch_rc` is still the host-stack initialiser. **Every rejection the
kernel can raise is silently discarded**, and the operator returns whatever was in the output
buffer -- uninitialised memory, not an error.

Measured on `add_rms_norm_dynamic_quant`: out-of-contract `D = 16385 / 20000 / 65536` gave
**15/15 silent accepts, no raise**, returning uninitialised buffers. It raised only
*intermittently*, when scheduling happened to let the lambda run first -- which is worse than
never raising, because it looks like a working guard.

**This defeats C51/C53 entirely.** Those rules exist so an out-of-contract shape fails loudly
instead of returning a wrong answer; an rc checked after an enqueue re-opens exactly that hole,
and it is invisible in review because the code *looks* like a correct guard.

**The fix: validate synchronously on the host, BEFORE the enqueue.**

```cpp
TORCH_CHECK(D >= 1 && D <= 16384, "op: D out of contract, got ", D);   // host-side, synchronous
at_npu::native::OpCommand::RunOpApi("op", [&]{ (void)call_kernel(...); });
```

Anything the host can decide -- shape, rank, dtype, declared ranges -- belongs in a synchronous
`TORCH_CHECK`. If a condition can only be known on device, it needs a status word the driver
reads back **after a sync**, not an rc captured by reference across an enqueue boundary.

**How to test it.** Feed a shape you know the kernel rejects and assert the call RAISES. Do not
assert on the returned rc, and do not accept "it raised once". The measured procedure, from an
audit of 14 operators:

1. **Prime the output with a poison sentinel** before the call (e.g. fill with `0xA5`).
2. **Assert it raises over >= 10 attempts.** The race is real and its rate is unstable: measured
   raise rates were `0/12`, `1/12` and `2/12` for different shapes of the same operator, and one
   operator raised `0/12` in the raise probe and then raised on the *first* call of the next
   probe. A single observation in either direction is worthless.
3. **For any attempt that did NOT raise, assert the output is bit-identical across >= 6 repeat
   calls.** A non-deterministic output is the proof the kernel never ran.

**Step 3 is the one that matters**, and it is the one you would skip. In that audit **8 of the 11
exposures left no surviving sentinel at all** -- the caching allocator handed out fresh blocks --
so a poison check alone would have reported "it returns *something*, probably fine". A surviving
sentinel is a bonus, not the primary signal. When it does survive it is damning: one operator came
back with `poison_frac = 1.0` (the entire output was still the sentinel) and another at 0.844.

**And the regression surface of this fix is small**, which makes it cheap to verify: moving a
check to the host can only ADD raises, so screening every published case shape for acceptance is
the whole test. All nine operators fixed in that audit kept 20/20 acceptance, and because
`*_launch.h` is host-only, all nine device binaries stayed **byte-identical**.

---

## C76: EVERY NUMERIC CAP MUST BE JUSTIFIED AGAINST THE **DECLARED** MAXIMUM  🔴 **CRITICAL**

**Rule.** Before packaging, take every numeric constant in your kernel and driver that can cause
a *rejection*, and check it against the maximum the operator's own specification declares. A cap
strictly below a declared maximum is a defect unless you can name the hardware limit that forces
it. "No case I was shown needs more" is not a justification -- it is the definition of the bug.

**Why this rule exists.** C67 says *derive the supported set, never enumerate it*, and C72 gives a
runtime probe for it. Both were in force, and this class still cost four operators, because the
C72 probe varies the **dtype** axis and every one of these failures lives on the **shape and attr**
axes. The measured bill:

| operator | spec declares | our source caps | cost |
|---|---|---|---|
| `softmax` | last axis `1 ~ 2097152` | bucket table ending `(12288, 1)` | 8 cases, **-9.28** |
| `adaptive_avg_pool_3d` | `W 1 ~ 256` | `LAUNCH(SUF,TY,128) LAUNCH(SUF,TY,144)` | 3 cases, **-4.23** |
| `conv_2d` | `K_h 1 ~ 16` | `khkw in {1, 9, 25}` | 3 cases |
| `grouped_matmul` | bias dtypes incl. bf16 | bf16 bias unimplemented | 23 cases |
| `mla_prolog` | an 8-value `n_heads` enum | rejects `n_heads` 1 and 2 | -- |

Five different operators, five different mechanisms -- a template instantiation list, a bucket
table, a lookup table, an unimplemented dtype branch, a hard cap chosen for headroom over the
largest visible case -- and **one shape**: a supported set fitted to the cases that happened to be
visible. The hidden set is 4x the visible set and probes exactly the edges you did not see.

**This is the cheapest rule in this document to obey.** The declared maxima are written in
`desc.md`'s support-range table and the caps are constants in your own source, so the check is
**static**: no device, no execution, no credit. Every failure in that table was findable before
submission by reading two files side by side. `tools/declared_vs_cap_audit.py` in the campaign repo
mechanises it, but the audit is a triage list, not a verdict -- you still have to read each hit.

### THE DECLARED MAXIMUM IS A FLOOR, NOT A CEILING (measured 2026-09-28)

**The hidden set can contain shapes OUTSIDE the operator's own published range.** Matching
`desc.md` exactly is NECESSARY but NOT SUFFICIENT.

`add_rms_norm_dynamic_quant`, hidden case 86:
```
desc.md declares   N (last dim)  1 ~ 16384     annotated "cases.csv 实测 128 ~ 16384"
largest visible                     16384
hidden case 86     D =             28672       1.75x ABOVE the declared maximum
```
Our kernel refused it **citing desc.md** -- a `compile_runtime_error`, the expensive kind, costing
the compile term as well as function and performance. One case, **-4.42**.

So the audit has two halves, and the second one is new:
1. **A cap BELOW the declared maximum is a defect.** Unchanged, and still the class that has cost
   the most -- five operators and counting.
2. **A cap exactly AT the declared maximum is a RISK.** Where headroom is cheap, take it: a
   runtime-tiled path with no fixed ceiling costs nothing extra and cannot be surprised. Where a
   ceiling is genuinely forced by UB, L1 or a workspace bound, write the number and the reason
   down so the next run prices it instead of rediscovering it.

**And do not fit the new constant to the case that bit you.** Raising the cap to exactly 28672 would
repeat the fitted-contract mistake one notch further out -- the same error `conv_2d` made when its
`khkw` set went from `{1,9,25}` to `<= 160` while the declared maximum was 256.

**The discriminator, and it is the whole skill in applying this rule:** a constant is *fine* if it
selects a **tile** and *fatal* if it gates a **rejection**. `wWant = 224` picking a vector width is
correct engineering. `if (R > 12288) return -2;` is a fitted contract. Trace each constant to
whether a value above it produces a *smaller tile and a correct answer*, or an *error return*.
Report both classes; only the second is a defect.

**And prefer a runtime-tiled path to a longer list.** Adding `256` to an instantiation list fixes
the three cases you were just billed for and leaves the next unlisted width just as broken, at the
cost of more compile time and a bigger binary. If you keep a static ladder for speed, put a general
runtime fallback beneath it so that **no declared shape can reach a rejection**. A ladder with a
fallback is an optimization; a ladder without one is C76.

**Two traps when you widen a cap:**
- **A newly reachable shape is newly reachable for the device too.** Lifting `W` to 256 at `C=512`
  may be the first allocation of that size the kernel has ever made -- see **C70**, and stop rather
  than retry if a shape faults the card.
- **A dimension that is prime factors into nothing.** One `softmax` hidden case is `R = 1000003`.
  Any widened path that assumes the extent divides by a tile width fails it. Handle the ragged tail.

Related: **C67** (derive, never enumerate -- C76 is its static, pre-submission check), **C68**,
**C69**, **C70**, **C71**, **C72** (the dtype-axis runtime probe this rule complements).

---

## C77: A CUBE TILE'S PARTIAL EXTENT IS HONOURED ONLY AT THE FRACTAL-ROUNDED DECLARATION  🔴 **CRITICAL**

**Rule.** A Cube tile's runtime valid extent is respected **if and only if** the declared dimension
equals the fractal-rounded valid extent:

```
Rows == ceil(ValidRow / 16) * 16        and        Cols == ceil(ValidCol / 16) * 16
```

Declare a tile wider than that and the result is **silently wrong from output column 16 onwards**,
and a mismatched **transposing (ZN) feed hangs the AI core** (`aicore timeout 507014`).

**Why this is the worst possible failure shape: the first 16 columns are CORRECT.** Any smoke test
that checks a corner, a first row, or a narrow slice passes. The error begins exactly where a
quick check stops looking.

**Probed, with a positive control on a known-bad case:**

| declared `Rows` | `ValidRow` | result |
|---|---|---|
| 128 | 128 | exact, 1.4e-07 |
| 128 | 16, 32, 37, 64 | **WRONG**, rel 8.5-10.7 |
| 48 | 33, 37, 48 | exact |
| 48 | **16** | **WRONG** |
| 16 | 1, 5, 16 | exact |
| 16x16 matched partials | -- | exact, 4.6e-08, on both the ZN and the non-transposing feed |

Note row 4: `Rows=48, ValidRow=16` fails while `Rows=48, ValidRow=33` passes. The rule is not "small
valid extents are fine" -- it is the equality above, and nothing else.

**What to do with a ragged extent.** Two correct options, and the choice is forced by tile width:
1. **Tile is 16 wide** -> use a runtime `DYNAMIC` valid dim; the equality holds automatically.
2. **Tile is wider** (e.g. a 64-wide fast geometry) -> **shift the last chunk back** so the tile
   stays full, and zero the `ov = ceil(N/kT)*kT - N` duplicated columns afterwards on the Vec side
   through an alias at **column offset 0**.

For (2), a companion Vec rule was probed the same way: **an aliased Vec tile at column offset 0 is
exact for every width 1..64**, including non-multiples of 8 floats -- but at a **non-32-byte-aligned
column offset it FAULTS the device**. So the zeroing alias must start at offset 0.

**Shifting back makes two cores write the same output rows.** That they write the same *bytes* is a
claim you must MEASURE, not argue: 5 repeats bit-identical across every shift-back shape, including
a causal `S=2047 / S_kv=2048` and `S=300 / S_kv=1001`. Determinism under a deliberate write overlap
is exactly the kind of thing that is usually fine and occasionally is not.

**And a rejection can be hiding a real bug, not merely narrowing the domain.** The `mha` kernel
refused `D=192` (148 declared combinations) via an explicit `D != 64 && != 128 && != 256` check --
but the guard was not about `D` at all. `kT = (kD <= 128) ? kD : 128` made `kNK = kD / kT` equal
**1** for `kD=192` by integer division, so the contraction silently covered 128 of 192 columns.
`D=192` needs `kT=64`. **Before deleting a rejection, find out what it was protecting** -- widening a
guard that sits on top of an arithmetic bug converts a loud refusal into a wrong answer.

### LIMIT OF THIS RULE: it does NOT extend to the K axis of a transposing `Mat->Right` TEXTRACT

Measured on `conv_2d`, 2026-09-27. Declaring the tiles at the 16-rounded K with a **partial valid K**
(8 or 24) let the Cube consume the surplus rows anyway:

| dtype | gw | result |
|---|---|---|
| fp32 | 1 | **MERE 261 / MARE 77121** |
| fp32 | 3 | **MERE 113 / MARE 108943** |
| every 16-exact K | -- | PASS |
| every fp16 `gw` | -- | PASS |

So on the contraction axis of a transposing feed, **K must be a TRUE multiple of 16** -- the
fractal-rounded declaration is not enough. This is why an odd `Kw` needs a dedicated permute kernel
on fp32: an odd `Kw` has no even divisor, so the parity has to come from the C1 axis, which forces
that axis inner in the filter layout.

**Read C77 as: the equality is NECESSARY everywhere and SUFFICIENT only off the transposing K axis.**

Related: **C76** (the fitted-contract class this guard belonged to), **C52** (barrier ownership),
**C70** (device faults on newly reachable shapes), **C79** (a silently-dropped kernel body).

---

## C78: A `tmp` TILE MUST BE DISTINCT FROM **BOTH** src AND dst  🔴 **CRITICAL**

**Rule.** For any PTO op whose 3-argument form takes a **scratch tile**, that scratch must alias
neither the source nor the destination. `TRSQRT(dst, src, src)` is a **race**; `TRSQRT(dst, src, dst)`
is **deterministically wrong**.

**Mechanism**, from `pto-isa` `109c9f72`, `include/pto/npu/a2a3/TUnaryOp.hpp:281`
(`TRsqrtHighPrecision`):

```
set_mask_count(); set_vector_mask(0, blockSizeElem);
vector_dup(tmp, (T)1.0, ...);              // writes ONE 32-byte block
set_vector_mask(0, validCol);
for r: vsqrt(dst + r*ds, src + r*ss);      // <-- reads src.  NO BARRIER SINCE THE DUP.
pipe_barrier(PIPE_V);
for r: vdiv (dst + r*ds, tmp, dst + r*ds);
```

There is a barrier between the `vsqrt` and the `vdiv`, and **none between the dup and the `vsqrt`**.
So with `tmp == src`, the dup overwrites **8 fp32 lanes** of the source with `1.0` and whichever op
lands first decides the answer.

**WHICH OPS.** Only the families that actually declare a `TmpTile`. In `a2a3` those are
`TUnaryOp` (**TABS TEXP TLOG TMULS TNEG TNOT TRELU TRSQRT TSQRT**), `TCvt` (**TCVT**),
`TGather` (**TGATHER**), `TSel` (**TSEL**), `TSels` (**TSELS**), `TPow` (**TPOW TPOWS**),
`TSort32` (**TSORT32**), `TMrgSort` (**TMRGSORT**).

**A BINARY OP IS NOT THIS BUG, and confusing the two produces a flood of false alarms.**
`TMUL(dst, src0, src1)` lowers to `vmul(dst, src0, src1, ...)` -- its third argument is a **second
source**, so `TMUL(p, p, p)` is squaring in place and is correct. The same holds for `TADD`, `TMAX`,
`TDIV`, `TSUB`, `TOR`. A first sweep of this campaign reported **30 sites; 29 were binary ops and
only 1 was real.** Scan by the op-name list above, and carry a **negative control**
(`TMUL(p,p,p)` must NOT be flagged) alongside the positive one
(`TRSQRT(ptmp, pall, pall)` must be flagged).
`skillyard-cannbench/tools/scan_tile_aliasing.py` does this; a full-fleet sweep of 49 operators found
**zero** further instances.

**Why it survives every gate we have.** The measured case (`rms_norm`, small-D path):

- It corrupts a whole output row -- `1/rms` becomes exactly `1.0`, so the row emerges scaled by
  `rms` -- but only a few hundred elements out of ~10^6, so **MERE stays inside tolerance** and only
  `MARE` fails. See **C69**.
- It is **nondeterministic**: 15 failures over 88 process-runs, and 1 in 6 repeats at a fixed shape.
- The visible cases missed it because the only visible small-path case used a **different dtype**
  (bf16, whose tolerance is 10x looser) **and** a `D` that never lost the race.
- 675 shape points, 5184 dtype-range points, 72 natural draws at 33M elements and **816 planted
  near-zero probes all passed.** Input conditioning was the wrong axis entirely.

**Its fingerprint, which identifies it from a score report alone: `MARE < 1` strictly.** For a
corrupted row `rel = |1 - rms|` at every element, so `MARE = |1 - rms|`, always below 1. A dropped or
stale element instead gives `rel -> 1.0` exactly. Two reported failures at `MARE` 0.609 and 0.852
back-solved to `rms` 0.391 and 0.148.

### The detection technique -- add it to the standard regression set

A pass/fail tolerance gate cannot see this. **Ask a sensitivity question instead:** per row, recover
the implied scale factor `c = ours / (reference without the normalisation)` and flag `|c - 1| > 1e-3`.
That catches a whole-row scale error at 1e-3 where a tensor-wide MARE gate needs 1e-2, and it fired
on the first sweep after 1500+ conventional probes had found nothing.

**And run ONE SHAPE PER PROCESS in a shape sweep.** 15 of the 88 observed failures needed a *cold
launch*; a single-process sweep systematically under-counts a startup-sensitive race.

Related: **C69** (a tiny MERE with a large MARE), **C52** (barrier ownership), the TRECIP
aliased-destination finding (same family: an aliased destination returns 1.0 with no error), and
**C74a** (`TRowSumOp` has a wide `FillTmp`, `TRowMaxOp` does not).


---

## C79: A CAPACITY GUARD CAN DROP THE KERNEL BODY AND STILL RETURN SUCCESS  🔴 **CRITICAL**

**Rule.** An `if constexpr` (or any compile-time) capacity guard around a kernel body can compile the
body **away entirely** while `call_kernel_*` still returns **0**. The operator then reports success,
writes nothing, and scores against whatever happened to be in the workspace.

Measured on `conv_2d`, 2026-09-27: a host slip produced `gw=5 -> KT=80`, past the L0A budget. The
guard dropped the body, the return code stayed 0, and the accuracy gate read **MERE 1.6-3.3** off an
untouched workspace. A second, independent route to the identical signature: a mechanical edit that
left the dtype-dispatch macro **defined but never invoked**.

**This is the same failure family as C73 and C75** -- a kernel that does nothing and reports success --
and it is the hardest of the three to see, because there is no empty output, no rejection, and no
device fault. Nothing in the launch path is wrong; the work simply is not there.

**How it was actually localised, and the technique to reuse: fill the output workspace with a
SENTINEL before the launch and check whether the sentinel SURVIVES.** A surviving sentinel proves the
body never ran. Note the asymmetry with C73's poison check: there, 8 of 11 exposures left no sentinel
because the caching allocator handed out fresh blocks, so survival is a bonus signal for a *missing
launch*. Here the launch DOES happen and the buffer is the one you primed, so the sentinel is
reliable -- it is the primary signal, not a bonus.

**Guard both sides.** A compile-time bound alone is what creates this: pair it with a **host-side
bound** that rejects the configuration before launching, and a **device-side `rc`** for the case that
reaches the kernel anyway. Then an out-of-budget configuration produces a loud refusal instead of a
silent no-op.

**And treat a dispatch macro as code that must be proven to run.** `defined but never invoked`
compiles cleanly and tests green on any case whose path is still wired. Assert at least one device
kernel row per accepted configuration (**C73**'s measurement) so a vanished body cannot pass.

Related: **C73** (a legal no-op must still launch), **C75** (an rc read after an enqueue is always
zero), **C66** (unwired host-side changes), and the `PTO_DBUF` finding (a lever that reserved memory
and did nothing -- the same class of "the change did not take effect").

---

## C80: A VEC OP ON A TILE WITH `validCol < Cols` **AND** `validRow > 1` CAN FAULT THE CORE  🔴 **CRITICAL**

**Rule.** A PTO Vec binary op or `TCVT` on a tile whose **valid column count is below its static
`Cols` while its valid row count is above 1** can fault the AI core:

```
the address for the VEC instruction to read/write UB is out of bounds
```

The **flat** shape is clean -- either `validRow == 1`, or `validCol == Cols`. One `TMUL` on `[112,64]`
tiles with a valid extent of `[3,17]` is enough to trigger it; independently reproduced on an
fp16->fp32 `TCVT` over `[1,2048]` tiles at `validCol` 1536.

**MTE IS NOT AFFECTED.** `TLOAD`/`TSTORE` with a partial `validCol` are fine, and the reason was
measured rather than assumed: they lower to `copy_*_align_b32` with `lenBurst` in bytes and
`ubGap = ((Cols-validCol)*sizeof(T)) >> 5`; the hardware advances the UB side by
`ceil(lenBurst/32)*32 + ubGap*32`, and **those two round-offs cancel exactly for every `validCol`.**
So the hazard is specific to the Vec pipe.

### It is built to survive your test suite

This is the part to internalise, because every cheap check passes:

| what was tried | result |
|---|---|
| a single launch | **passes** |
| 40 repeats over the SAME buffers | **passes** |
| a 3480-case correctness sweep | **never saw it** |
| repeats over **freshly allocated** tensors | **faults at launch 1-7** |

And the obvious explanations are all wrong, each falsified by measurement:
- **not a GM overrun** -- padding by 64 floats did not help, putting all six tensors in one buffer
  with 1 MB canary gaps did not help, and a canary check found **zero** out-of-bounds writes;
- **not a UB height limit** -- shrinking the UB budget from 172032 to 98304 bytes did not help.

It was localised by **compile-guard bisection**: the two `TLOAD`s alone are clean, the arithmetic
block faults, and then down to the single instruction.

### The fix

**Run every vector op and every cast at the tile's FULL STATIC WIDTH.** For a 2-D tile, take a flat
`[1, Rows*Cols]` view over the whole region *including* each row's column padding -- when all buffers
share the pitch, the valid columns get the right answer and the padding computes a value you discard.
**Zero that padding once per launch** so it can never hold a denormal and cost you throughput.

After the fix: 108 configs x 25 fresh-allocation launches, **0 faults**.

### A companion MTE rule from the same probe

**A UB address that is 4-byte aligned but NOT 32-byte aligned faults the MTE.** If you compute a
per-lane base offset at runtime, round it to a multiple of 8 elements (fp32). Also measured there:
index `TGATHER` and elementwise ops with a non-multiple-of-8 runtime length are exact **on the flat
shape**, and a MaskPattern gather concatenates the selected columns **without honouring the
destination's padding**.

Related: **C77** (a Cube tile's partial extent needs the fractal-rounded declaration -- the Cube-side
analogue of this rule), **C74a**, **C79**, and the cookbook entry on a narrow DMA burst rounding its
write up to a 32-byte granule. The family is one sentence: **an operation touches more than the valid
extent you asked for, and which pipe you are on decides whether that is benign.**

## C81: AN IN-PLACE `TDIVS` SILENTLY BECOMES A RECIPROCAL-MULTIPLY  🔴 **CRITICAL**

`DivSOp::BinSInstr` carries a "fix inplace alias" branch: when `dst == src`, `TDIVS(p, p, s)`
lowers to `vmuls(p, p, 1.0f/s)` instead of a true divide. It compiles, it runs, it returns a
plausible answer, and it is **not the same number** -- `1.0f/s` is rounded once and then multiplied,
so you get up to 2 roundings where IEEE division gives one correctly-rounded result.

This matters whenever the divide is the thing you are relying on for accuracy. Measured on
`roi_align` (2026-09-28): an arm built specifically to *remove* a reciprocal-multiply and replace it
with a real division would have silently re-introduced the reciprocal-multiply it existed to
eliminate, because the natural spelling aliased `dst` onto `src`. The arm was built with a raw
`vdiv` into a **distinct destination** instead.

**Rule: when a division's rounding is load-bearing, give `TDIVS` a destination distinct from its
source, or emit `vdiv` yourself.** Same family as **C78** (a `tmp` tile must differ from src and
dst), the `TRECIP` aliased-dst trap (returns `1.0`, no error), and identity-`TCVT` corruption: PTO
has several ops whose collapsed-alias instantiation is a *different operation*, not a no-op.

## C82: `double` IS REJECTED OUTRIGHT INSIDE AN AICORE FUNCTION

Under `--cce-aicore-arch=dav-c220-vec`, bisheng refuses any fp64 in device code:

```
error: cast to/from double precision floating variable is not allowed in aicore function
```

So the standard numerical remedy -- *compute the quotient in fp64 and round once* -- is simply
unavailable on device. This is a hard constraint on how a precision problem can be fixed, not a
tuning choice, and it is worth knowing **before** designing a fix around wider intermediates.

**Consequence:** when an fp32 expression on device is not accurate enough, the wider-intermediate
remedy is unavailable and you must fix the *expression*, not the precision -- see **C83**, where the
culprit was FMA contraction and the fix was a `volatile` launder.

**Moving the arithmetic to the host is the fallback, and on the one case measured it did nothing.**
Supplying every division from the host as IEEE-exact fp32 left the coordinates bit-for-bit unchanged
(proven live by a poison control). So host precompute is a *performance* play on the scalar pipe, not
a correctness fix -- price it separately and do not reach for it before identifying which step is
actually non-IEEE.

## C83: FMA CONTRACTION SILENTLY COSTS 1 ulp IN A SCALAR FLOAT CHAIN, AND `-ffp-contract=off` DOES NOT STOP IT  🔴 **CRITICAL**

A scalar multiply-add chain such as `base + i0*s0 + i1*s1` gets **contracted into FMAs**, which drops
one rounding per fused pair. That is a **1-ulp difference on a minority of values** -- the exact
signature of a "the hardware is not IEEE" red herring.

Measured on `roi_align` (2026-09-28) with an ABI-preserving build that emitted the kernel's own
computed coordinates instead of its result, compared bitwise against torch's fp32 expression:
**5.7-11.8% of coordinates differed, by exactly 1.00 ulp** (case 8 y 56/679, case 2 y 211/1792,
case 12 y 238/2509).

**`-ffp-contract=off` is silently ineffective here.** It **changes the binary** (different md5) while
leaving the arithmetic **bit-identical**, so it reads as a clean negative when it is a no-op. That
false negative was used to "falsify FMA contraction" -- and FMA contraction was the entire defect.
**Never treat that flag as evidence, and never rely on it for numerical reproducibility.**

**What works: launder the products through `volatile` to force a round-to-fp32 at each step.**

```cpp
AICORE inline float coord3(float base, float i0, float s0, float i1, float s1)
{
    volatile float t0 = i0 * s0;      // volatile forces the fp32 rounding
    volatile float t1 = i1 * s1;      // that the FMA was skipping
    const float a = base + t0;
    return a + t1;
}
```

Result on `roi_align`: coordinates **0/679, 0/1792, 0/2509, 0/2100** bitwise-exact, `max|dev| = 0.0`;
the gate went **6/20 -> 12/20** with **no case broken**, every fp32 case improving **13x-71x** in
`max_diff`, for ~10 lines at 3 sites -- no ABI change, no host buffer, no redesign. Cost: all 20 cases
inside the +-6.29% noise band, but 18 of 20 slower with a consistent sign, so score it *at or below
the noise floor, directionally slightly negative*, not free.

**Why 1 ulp is not a footnote.** `frac = c - floor(c)` cancels the large integer part and leaves the
error at full size against a now-tiny value, so a 1-ulp *relative* coordinate error becomes an
*absolute* weight error of `ulp(coord)`, reaching the output scaled by `max|x|`. Predicting
`ulp(coord)*max|x|` tracked observed error within 0.4-2.8x on all 9 fp32 cases; the
accumulation-order prediction `ulp(x)*sqrt(count)` was **20-90x too small on every one**.

**RETRACTED in the same run, do not repeat:** the claim that AICore `vdiv` is not correctly rounded.
A standalone probe over 65536 fp32 inputs spanning 2^-20..2^20 found `vdiv(a,16)`, `vmuls(a,0.0625)`,
`vdiv(a,7)` and `vdiv/16` vs `vmuls*0.0625` **all 0/65536 bitwise different, max 0.000 ulp**. `vdiv`
is exactly correctly rounded for power-of-two and non-power-of-two divisors alike. The original claim
was an inference from a 1e-11 wobble in a 50M-element fp16 case -- a wobble read as an ISA property.
**Scalar `/` is IEEE too:** supplying every division from the host, IEEE-exact, changed the
coordinates by *nothing* (`56/679` -> `56/679`), proven live by a poison control that drove the same
probe to `679/679`. The divisions were never the non-IEEE step.

**How to find this class at all.** A result-level diff cannot separate a bad coordinate from a bad
accumulation, and the two have opposite fixes. Build a diagnostic that emits the **intermediate**, and
when you supply a value from outside, **poison it** to prove the kernel actually reads it -- a silent
fallback and a genuine no-effect look identical.

## C84: THE >255-BLOCK ROW STRIDE OVERFLOWS -- BUT ONLY WHERE A ROW STRIDE IS EMITTED  🔴 **CRITICAL**

A multi-row Vec tile's row stride in 32-byte blocks is `Cols*sizeof(T)/32`. For fp32 that is
`Cols/8`, so **`Cols = 2048` gives 256 and overflows the 8-bit repeat-stride field.**

This rule has been wrong twice in opposite directions, so the *scope* is the whole content:

**A tile is exposed only when the row stride is actually emitted.** Two shapes qualify:
1. a **`ValidCol < Cols` sub-view** -- the rows are not contiguous, so each row advance is a stride;
2. a **multi-row broadcast/expand** that re-reads one operand per row (`TCOLEXPANDMUL`,
   `TROWEXPANDMUL`, and the same family).

`ValidCol == Cols` elementwise work is **safe at any width**, because the tile flattens into one
contiguous run and no row stride is ever emitted. `rblk == 1` is safe for the same reason -- there is
no row advance.

Measured on `arnq` (2026-09-28), `W in {512,1024,2048,4096} x rblk 1..16`, 64 configs:
**512 and 1024 clean at every rblk; 2048 and 4096 fail at every `rblk >= 2`, pass at `rblk == 1`.**
Signature: within each row-block **row 0 is exact, rows 1..rblk-1 are garbage** -- good rows exactly
`{0,8}` at rblk=8 and `{0,3,6,9,12,15}` at rblk=3. It is **not the reduction**: swapping in a 2-level
`OneRepeatProc` reduce moved the failure by 2 elements out of 3.09M.

**How to probe it, because the obvious probe returns a false negative.** An earlier 137-point sweep
concluded there was *no* cliff and got `Cols=4096` exact -- every probe tile had `ValidCol == Cols`,
so it flattened and the stride was never emitted. **A stride-limit probe must vary `ValidCol` away
from `Cols` AND force `rblk >= 2`.** Otherwise it is structurally incapable of seeing the fault it is
looking for.

**And do not lift such a cap without pricing it.** Lifting arnq's `W <= 1024` was measured at
**+0.34 (W<=2048) / +0.42 (W<=4096)** -- inside the ~0.4 cross-session drift -- for **four**
correctness fixes on a path that currently has none. HAP compresses hard near the hardware limit: a
1.14x kernel speedup moved one case's HAP only 0.488 -> 0.532. Correctly declined.

Related: **C49a** (scratch width), and the row-reduce scratch trap, which is a *different* fault with
the same symptom -- a wrong reduction output beside a bit-exact elementwise output points at the
scratch, not at the source tile's width.

## C85: A QUERIED-CAP FIX IS PROVABLY NULL ON A PART WHERE THE QUERY RETURNS THE OLD CONSTANT

When a fix replaces a hardcoded constant with a runtime query (`get_device_limit` /
`GetDeviceResLimit`), **an A/B on a part whose queried value equals that constant is a no-op by
construction** -- and a null result there says nothing about the fix's value on a part where the values
differ.

Measured: an A2 910B2 reports `vector_core_num = 48`, **exactly** the constant the fix replaced. The
A2 A/B came back `-0.028` against a `0.019` null band, and that number was then used to argue the fix
carried no value on A3. That inference is invalid. **A null on the part whose value matches is what a
correct fix looks like.**

**Prove liveness instead of arguing about it.** Force the *fallback* to an obviously wrong value and
show the geometry changes: with the fallback forced 48 -> 7, the plan read `rblk 6 / items 43` before
any op call and snapped to `rblk 5 / items 52` -- exactly the shipped 48-core plan -- after one call,
proving the runtime push executes on the live registered path and its value wins. Three shapes, same
result, source restored byte-for-byte afterwards.

This is the same failure mode as an **unwired** lever measuring "neutral": if you cannot show the
lever changing an observable, you have not measured the lever. Related: the L2-alias trap where
passing a literal device id makes `rtGetL2CacheOffset` return 0 -- success, and useless -- so the
alias never executes and the technique gets wrongly retired.

## C86: NO DECLARED VALUE MAY REACH A REJECTION, AND EVERY REJECTION MUST NAME ITSELF  🔴 **CRITICAL**

This is the **largest and most repeated defect class** in the cann-bench campaign: a kernel-side
contract fitted to the cases that happened to be visible. Measured cost so far, hidden cases lost:

| operator | our gate | declared | cases |
|---|---|---|---:|
| `gqa` | `D != 128 && D != 256 -> -3`; `Skv % 128 != 0 -> -4`; `Nq*D > 65535 -> -8` | `D 64~512 64-aligned`; `S_kv 1~8192`; `Nq<=256` | **26** |
| `grouped_matmul` | bf16 bias unimplemented | bias dtypes incl. bf16 | 23 |
| `softmax` | bucket table ending `(12288,1)` | last axis `1~2097152` | 8 |
| `moe_gating_top_k_softmax` | `E > 512 -> ep = 1024`, no instantiation above | `E 1~2048` | 7 |
| `adaptive_avg_pool_3d` | `LAUNCH(...,128) LAUNCH(...,144)` | `W 1~256` | 3 |
| `conv_2d` | `khkw in {1,9,25}` | `K_h 1~16` | 3 |
| `scatter` | index dims must equal data outside `dim` | PyTorch allows `index.size(d) <= src.size(d)` | 3 |

**THE DETECTOR, and it is mechanical: if your accepted set EQUALS the spec's `cases.csv 实测`
column, you have fitted the contract.** `desc.md` annotates declared-vs-exercised per axis and hands
you the gap for free. `gqa` accepts exactly `D in {128,256}` where 实测 is "128 / 256", and exactly
`Skv % 128 == 0` where 实测 is "128 ~ 2048". That is not convergent engineering, it is the visible
cases written into the source. `mha` is the same shape -- a declared-surface probe returned **476 of
673 combinations REJECTED**, led by `D=192`, which 实测 notes never appears.

**Rule 1 -- a static ladder needs a general path beneath it.** Adding `192` to an instantiation list
fixes the cases you were just billed for and leaves the next unlisted value just as broken. Keep the
fast ladder for the common widths, then fall through to a runtime-tiled path so **no declared value
can reach a rejection**. The discriminator stays: a constant that selects a **tile** is correct
engineering; a constant that gates a **rejection** is a fitted contract.

**Rule 2 -- a rejection must be self-describing.** `gqa`'s 26 failures arrived as one opaque line:

```
AI算子执行失败: gqa: shape outside the kernel contract (gqa_plan rejected)
```

The kernel distinguishes `-1` through `-8` internally and **none of it is surfaced**, so an 80-case
remote run that already knew which axis failed told us nothing, and the defect had to be re-derived
locally from `desc.md`. Emit the **axis name, the offending value, and the bound it violated** --
`"D=192 not in {128,256}"` -- and one hidden run becomes a complete, ordered defect list.

**Why this pays more than it looks.** These rejections are `compile_runtime_error`, so they scale the
**compile** term as well as function: `gqa` lost 6.5 compile marks and 10.1 function marks on top of
the performance those 26 cases would have carried. Its full-pass projection is **78.04 against a
68.82 posted entry, +9.22** -- the largest single gain on the board, from one class of constant.

## C87: BACK-TO-BACK ACCUMULATING MMADs INTO ONE L0C NEED `pipe_barrier(PIPE_M)`  🔴 **CRITICAL**

`TMATMUL_ACC` reads L0C as its C-matrix source **and** writes it. In a K loop with an L0 ping-pong
path, if nothing drains the **M pipe** between iterations the next MMAD's L0C read can overlap the
previous MMAD's L0C write. The hardware reports:

```
aicore error, error code = 0x40000
errorStr: The address for VEC to read L0C conflicts with that for CUBE to write L0C
```

on **every** Cube block, and the host sees ACL **507015** plus an unrecoverable device fault that
cascades every later case.

**It needs `nk >= 2`** (two K sub-tiles in one L1 chunk), and **whether a part tolerates it depends on
the accumulator size.** Measured: one A2 tolerates it from `MT*NT >= 1024` up; an A3 does not. That is
the whole trap -- see the portability note below.

**Fix: `pipe_barrier(PIPE_M)` between consecutive accumulating MMADs.** Verified to clear the fault
*with the ping-pong forced on for every tile*, i.e. it neutralises the mechanism rather than the
trigger. Cost on `conv_3d_backprop_filter`: **0.10 points**, all of it lost MMAD-to-MMAD pipelining.

**Two plausible fixes that were built, measured, and FAILED -- do not retry them:**
1. **Ping-ponging the accumulator between the two halves of L0C.** The obvious candidate, because the
   error names the *address*. It was written, built and measured neutral before a reproducer killed it.
2. **Declaring the `Acc` tile with the issued MMAD row count (`mrow`) instead of `mvalid`.**

**Do not gate the barrier on accumulator size.** The safe size differs per part and is unknown on
anything you have not tested. A fitted threshold is what caused this fault: the shipped kernel's
`MT*NT >= 1024` ping-pong condition was tuned by trial on an A2 and happened to sit exactly at that
part's tolerance boundary, so every scored case avoided the hazard and the A3 hit it on the first small
shape. **A threshold fitted by trial that guards a HAZARD is a portability landmine, not a tuning
choice** -- the shape-fitting rule of **C86** applied to hazards rather than to declared ranges.

**Build a reproducer before you believe a fix.** Forcing the hazardous path on for a small tile
(`MT = NT = 16`) reproduced the identical error code, error string and ACL code on local hardware --
and that reproducer is the only reason the two failed candidates above were caught instead of shipped.

**A TRANSITIVE `M -> MTE1 -> M` FLAG CHAIN ALSO DISCHARGES C87, and it is stronger than the
barrier.** A kernel that keeps a **single** L0A/L0B slot must already protect that slot's WAR, so it
carries:

```
set_flag(PIPE_M, PIPE_MTE1, id); wait_flag(PIPE_M, PIPE_MTE1, id);   // prev MMAD -> this TEXTRACT
  ... TEXTRACT into the single L0A/L0B slot ...
set_flag(PIPE_MTE1, PIPE_M, id); wait_flag(PIPE_MTE1, PIPE_M, id);   // that TEXTRACT -> next MMAD
TMATMUL_ACC(...)
```

That fully **serialises** M rather than merely draining it, so L0C protection falls out as a
by-product. Measured on two operators: adding the explicit barrier to a kernel that already had the
chain cost **+0.18 / null** (sign test p=0.50) -- the pipe was already empty, as predicted before
measuring.

> **CORRECTION, and it is the important part. "The `M -> MTE1` half is the one that matters" is
> WRONG -- it cleared 3 of 4 kernels that are actually exposed, because all four HAVE that half.
> The missing clause is DISTANCE, not presence:**
>
> ```
> coverage requires   chain_ordering_distance  >=  accumulation_edge_distance
> ```
>
> A kernel with **S** L0A/L0B slots places its `M -> MTE1` wait **S iterations back**
> (`if (t >= S) wait_flag(PIPE_M, PIPE_MTE1, ev[t % S])`), which transitively orders `A(t-S)`
> before `A(t)`. If the accumulation edge into one L0C is **D** apart and **`D < S`**, that edge is
> **undischarged however complete the chain looks.** The two safe kernels were safe because they keep
> a **single** slot, so `S = 1 = D` -- **a property of the slot count, not of the flag's presence.**

**So compute S and D per edge.** Worked verdicts from one trace over four kernels:

| kernel | ACC sites | edge | S | D | verdict |
|---|---|---|---|---|---|
| lstm | 225, 267 | cross-iteration `kb->kb+1` | 1 | 1 | **covered** |
| conv_2d `kernel_conv.cpp` | 331 | per m-block `kk->kk+1` | 2 | `nmb` | **UNCOVERED at `nmb==1`** |
| conv_2d `kernel_conv_gen.cpp` | 387 | consecutive `t->t+1`, one acc | 2 | 1 | **UNCOVERED always** |
| grouped_matmul | 297, 511 | 511: `ks->ks+1` | `kL0D?2:1` | 1 | 297 covered; **511 UNCOVERED whenever `kL0D`** |
| gmsq | 388 `cube_mm`, 433 `cube_mul` | consecutive sub-steps | 1 / `kNL0=2` | 1 | 388 covered; **433 UNCOVERED** |

**Two more forms the presence rule gets wrong, both measured:**

- **A same-iteration `set_flag(PIPE_MTE1, PIPE_M, id); wait_flag(...)` ring is NOT a discharge.** It
  makes M wait for MTE1; it does not drain M and never orders M before M. It sits immediately before
  the MMAD and looks exactly like the chain's second half. Three of the four kernels above have one.
- **A TRAILING `set_flag(PIPE_M, PIPE_MTE1, id); wait_flag(...)` pair AFTER the MMAD, in the same
  function, DOES discharge** -- it closes the chain at distance 1 with no barrier. That is why
  `gmsq`'s `cube_mm` and `grouped_matmul`'s line-297 helper are safe.

**A kernel's own comments may document the exposure as a performance argument.** One header states
outright *"Two L0 buffers move that flag to distance 2."* That sentence **is** the C87 defect, written
down by whoever introduced it. Another file's comments record a device fault blunted by disabling
double-buffering, diagnosed as "an exactly-full L0B", with the caveats *"the fault's footprint is not a
clean function of the shape"*, *"at fixed K=272 it fires at nvalid 20/24/32 and not 16/34/44"*, and
*"N=160 faulted through the packaged driver while the same shape was clean through a standalone
harness"*. **Address- and timing-sensitive, shape-incoherent: that is this defect's signature**, and
the blunt workaround was costing 0.46 points and three cases' speedup that the barrier may recover.

**Three consequences for how you check it.**

1. **Grepping for `pipe_barrier(PIPE_M)` is not a C87 audit.** It reports safe kernels as FAIL, and --
   worse -- would *clear* a genuinely racy ping-pong kernel that happens to contain one unrelated
   barrier. A drain hidden behind a macro (`#define MHA_MBAR() pipe_barrier(PIPE_M)`) also counts once
   as a literal while the call sites are what actually drain. Resolve macros, then **count logical
   accumulation edges against discharges**, per file -- and compute **S and D**, per the distance rule
   above. **Audit only the sources the wheel actually builds:** one tree carries 14 files in
   `variants/` that its CMakeLists never registers, one of them an older revision of the shipped
   file, and a scan of all of them buried the single live file so completely that it never appeared in
   the report at all.
2. **Count the edges; do not eye them.** Three easy misses, all real: an accumulator declared
   *outside* the K loop makes `kb -> kb+1` a **cross-iteration** edge; a **bare `TMATMUL_ACC`** outside
   the usual `if/else` slot-select form is a third edge the pattern skips; and a `first` flag never
   reset inside an inner loop makes **both** `j -> j+1` and `kc -> kc+1` accumulate. An ordered token
   trace of every MMAD and pipe flag, comment- and string-suppressed, is the only reliable method.
3. **The chain can be present and still not cover every edge.** The ping-pong form skips the
   `M -> MTE1` wait for the first iterations (`j < 2`). Present-but-conditional is not discharged.

**Land the drain anyway even when the chain already covers it.** An invariant that holds by
coincidence vanishes the moment someone double-buffers L0 -- which is exactly the schedule one of
these kernels' own headers records as racy. It was free on one operator and cost **0.22** (~1.5%,
sign test p=0.041, concentrated entirely in the short cases) on the other, against a triage of
**-18.07** for serving an artifact that can take down a shared runner. **No hoist exists**: the edge
is between consecutive *innermost* iterations, so any outer placement fails to cover it. Cost tracks
**placement depth, not the operator** -- once per K step is free, twice per innermost iteration is 6-7x
the `conv_3d` reference.

**And check the reachable accumulator size through the DECLARED surface, not the scored cases.** Both
operators above reach `MT*NT = 256` on declared-but-never-scored paths -- one because
`gen_kT(S_kv) = (S_kv >= 64) ? 64 : 16` while its checker accepts `S_kv` from 1, the other because its
reject function requires only `1 <= Dv <= Dk` and never that `Dv` be 64-aligned. The visible cases use
neither. **A tolerance that depends on accumulator size plus a cap fitted to the visible cases is the
C86 failure wearing a hazard's clothes.**

## C88: THREE MORE SHAPES OF LATENT DEVICE FAULT, ALL CORE-COUNT-INDEPENDENT

Found in one operator alongside **C87**, each reproduced on local hardware and each able to fault
*any* part. Check these whenever a kernel has a fast path fitted to the sizes it was tested on:

1. **A tail-overlap expression that goes NEGATIVE below the planner's chunk floor.** `m0 = DHW - CH`
   with `CH = 128` and `D*H*W < 128` makes the overlap negative, so TLOAD/TSTORE address **below the
   base** of the tensor -- vector core exception, ACL **507035**. Any `x - CONST` used as an offset
   needs a floor at 0 and a separate small-problem path.
2. **`scatter_vnchwconv_b16` with `repeat == 1` and non-zero repeat strides is an ILLEGAL encoding** --
   it returns **silent garbage**, not an error. Measured wrong for `D*H*W <= 16` and correct from 17.
   When a repeat count collapses to 1, zero the repeat strides.
3. **Fixed-width epilogue UB tiles with no bound on the reduction extent.** `EPI_TMAX`-wide tiles with
   no check on `Kd*Kh*Kw` or `Cin/groups` faulted (507035) at `K = 9x9x9` (extent 729) and at channel
   counts 600 and 1024. Chunk the epilogue over **every** axis that can exceed the tile, not just the
   one the visible cases stress.

**Harness note that cost a false positive:** torch_npu's task queue can submit a torch `fill_`
**after** a direct `ctypes` kernel launch on the same stream, so a poisoned-guard check read a torch
write as a kernel bug. Put `torch.npu.synchronize()` after the poison fills before launching.

## C89: A SAME-PRECISION `native_output` MUST BE COMPUTED ON THE **CPU**  🔴 **CRITICAL**

`compare_tensors` uses the same-precision reference's **per-element error COUNTS** as the CPU
denominator of its small-value and cancellation fallbacks:

```python
cancel_passed = cancel_error_count / max(cancel_cpu_error_count, 1) <= 2
```

An `native_output` computed with `device="npu"` therefore carries **NPU-sized errors**, inflating that
denominator until the ratio test passes **unconditionally**. The harness reports a clean sweep and the
real evaluator does not.

Measured on `mla` (2026-09-29): `mla_common.native_ref` used `device="npu"`, and that is **why its
record claimed 20/20** when cann-bench scored it **11/20 / 48.30**. The evaluator computes the
reference on CPU (`evaluator.py:401-405`, `native_inputs = tensors_to_cpu(...)`, `to_device=False`).
After the harness was corrected it reproduced the evaluator **exactly** — same 11/20, the same nine
cases, MERE agreeing digit-for-digit.

**Rule: the same-precision reference is a CPU computation at the case's own dtype. Never on device.**
This is the cheapest possible instance of the self-harness-pass class: one keyword, and it manufactures
a false full pass on an operator whose real score is 20 points lower.

**Related trap in the same family — MERE is not the gate.** On that operator case 1 *passed* at MERE
`3.361e-4` while case 13 *failed* at `2.186e-4`. `compare_tensors` runs a fast path
(`mere < thr and mare < 10*thr`) and then a three-domain fallback; what fails is usually
`normal_passed`. Never reason about pass/fail from MERE alone.

**And dtype slack is not symmetric.** bf16 gets 8x more room than fp16 on **two** axes — MERE threshold
`2^-7` vs `2^-10`, and `small_value_threshold` `2^-8` vs `2^-11`. The second matters more: a wide
small-value band can swallow **every** element into the small-value/cancellation domains, leaving
`normal_total_count == 0` and a reported `MERE = MARE = 0.0`. So "the bf16 case passes" is **not**
evidence that the arithmetic is sound — it may mean no element was left in the normal domain to fail.
Check `normal_total_count` before drawing any conclusion from a bf16 pass.

## C90: A UB ARENA OVER-RUN THAT IS BENIGN AT `db==1` IS A HARD FAULT AT `db==2`  🔴 **CRITICAL**

An over-run that "cannot matter because of ordering" stops being ordered the moment you double-buffer.
With `db == 2` the **next** work item's tile is prefetched into slot 1 while the current item is still
consuming slot 0. If slot 1's staging top crosses into the chunk the current item is **reading**, the
consumer is corrupted — and unlike `db == 1` nothing heals it, because MTE2 issues the prefetch *after*
the chunk's own DMA while the consumer waits only on that chunk's flag, which was set **before** the
prefetch.

Measured on `scatter` (2026-09-29), 36 declared-shape cases:

| | shipped | fixed |
|---|---|---|
| hard device fault | **12 / 36** | 0 |
| silently WRONG | **4 / 36** | 0 |
| PASS | 20 / 36 | **36 / 36** |

```
507035 ... aivec error, core id 8 ... errcode:(0,0x4000,0)
errorStr: The GM address accessed by scalar exceeds 48 bits
```

Ordinary shapes: `[4096, 43, 256] dim=1`, `[4096, 86, 128]`, `[4096, 172, 64]`. Neighbouring `D = 42` and
`44` pass, so nothing in the visible set could have found it.

**Rule: reserve the inter-arena gap in the capacity bound AND in the `db` decision, not just in the
tile size.** If an over-run is argued benign by ordering, state which flag orders it and check that the
prefetch is on the same flag. It usually is not.

## C91: AN ARENA BOUND MUST COUNT UNMASKED-REPEAT OVER-PROCESSING, NOT DMA EXTENTS  🔴 **CRITICAL**

A capacity bound computed from **DMA extents** is wrong whenever a vector helper processes whole
repeats. `widen`/`narrow`-style helpers write `ceil(n/64) * {256,128}` bytes regardless of `n`, so the
real high-water mark exceeds `align32(len * ESZ)` by up to **126 B**.

Measured: counting DMA extents put a threshold at `L = 21969`; the true threshold is **`L = 21953`**, and
a chunk sized to the DMA figure still over-ran by 64 B. A first fix reserving 288 B **still** over-ran by
64 B on 860,807 of 66,701,000 audited plans; only 384 B cleared it.

**Rule: size the arena from what the widest helper WRITES, add a named reserve, and pin it with a
`static_assert` that states why the number is what it is.** The root cause on `scatter` was that the
arena arithmetic was entirely runtime and host-side with **not one `static_assert` in the kernel**.

## C92: torch's `amin`/`amax` PROPAGATE NaN — `(n<o) ? n : o` DROPS IT

`scatter_reduce_` with `amin`/`amax` propagates a NaN from `src`. The natural device form
`(n < o) ? n : o` is **false** for NaN and therefore keeps the old value, silently dropping it. The
comparator then fails at the **NaN-position gate** with `MERE = MARE = 0.000000` — the signature that
looks like bit-exactness and is not.

Fix: `(n < o || n != n) ? n : o` for floating accumulators. Also check the narrowing path —
`f32_to_bf16` turned some NaNs into `-0.0`, which needs its own `v != v` guard.

This class is **invisible to the C86 declared-surface detector**, because the value is *accepted and
mis-computed* rather than rejected. A declared-unconstrained value range means NaN is a legal input.

## C93: A PLANNER SWEEP CAN BE BLIND TO THE ONLY BRANCH THAT REJECTS

A sweep of 213,850 planner shapes reported **0 rejections** and was wrong twice over:

1. the query entry point **never passed `vecPath`**, so the vector path was unreachable by construction;
2. its `outer` grid was `(1, 2, 13)` and **never reached the core count**, so the cost model never chose
   the widest tile — *the only branch that can reject*.

A corrected sweep over 10,160,600 shapes found **17,940 rejections** in that same planner, 3,588 of them
reachable at runtime, every one with an **empty** message.

**Rule: a sweep must be shown to REACH each branch it claims to clear.** Instrument the branches and
assert coverage, or the zero means nothing. This is the seventh uncontrolled scan in this campaign to
return zero and be wrong.

## C94: A TASSIGN OFFSET CONTAINING A RUNTIME COLUMN INDEX FAULTS THE CORE  🔴 **CRITICAL**

A Vec tile `TASSIGN`ed to a **non-32-byte-aligned** UB address raises `aicore exception` / ACL **507015**
— a hard fault, not a wrong answer. Any offset built from a runtime *column* index will eventually be
unaligned.

Isolated on `mla` with a paired control:

| offset | result |
|---|---|
| exact column offset (`UB_TMPB + (r*kCW + off)*4`, off=249 → 996 B) | **aicore exception** |
| same rounded to 8 fp32 (32 B) | no fault, but MERE 7.6e-02 — wrong by the ≤7 columns the rounding masks |

So the obvious repair (round the offset) trades a fault for a wrong answer. **The safe idiom is to fill
the whole chunk, then write each row's valid prefix at column 0 — never index into a row.**

And note the diagnostic trap: **the first fault poisons the process**, so 13 of 14 cases in that sweep
reported failure from one bug. One fault means re-run from a clean process before believing any later row.

## C95: TWO A2/A3 LIBRARY FACTS THAT CLOSE OBVIOUS DESIGNS

Both checked against the pinned headers rather than assumed, and each killed a plan that looked free:

1. **`PadValue::Zero` does NOT zero the inactive rows of an L1 Mat tile.** It zero-fills only the final
   partial C0 block. Every "load fewer rows and let the pad be zero" scheme is therefore wrong, and a
   partial reduction over such a tile reads **stale L1**. Workable alternative, used on `mla`: size the
   chunks so **every tile is statically full** (e.g. 64-wide + 16-wide), handle the M and reduction tails
   by *idempotent overlap* where the result is stored rather than accumulated, and pre-fill the operand
   from an in-bounds window so `0 x finite == 0` exactly.
2. **`TSTORE`'s `preQuantScalar` cannot scale an fp32 Acc into an fp16 GM store.**
   `CheckAcc2gm<..., isQuant=true>` at `pto/npu/a2a3/TStore.hpp:594` `static_assert`s an int8/uint8
   output. Any "scale it for free in the fixpipe" plan for a 16-bit float output is closed by the
   library, not by judgement.

## C96: THE COST OF A GENERAL PATH IS ITS LAUNCH SYMBOLS, NOT ITS CODE SIZE

Widening a fitted contract usually means adding a general path, and the reflex objection is that the
binary grows and the fast path slows. Measured on `mla` with a **stub ablation** — keep the added launch
symbols, delete the 45.7 KB general-path body:

| arm | added `__global__` symbols | per-case median ratio |
|---|---:|---:|
| control | 0 | 1.0000 |
| general path, 1 symbol | 1 | 1.0143 |
| general path, 2 symbols | 2 | 1.0183 |
| **stub: 2 symbols, bodies REMOVED** | 2 | **1.0218** |

Removing the body costs **the same or more** than keeping it. So the cost tracks the **number of added
entry points**, not their size — and folding two general launches into one recovered 0.4pp at zero cost
in generality. `scatter` measured the same thing from the other side: PATH G's +383 KB and 46 extra
instantiations were **neutral**.

**Consequence: "the general path would cost performance" is not a reason to keep a fitted contract.**
Add the generality, then minimise the number of launch entry points.

**Method note that earned this:** the score-level delta read as a null (−0.37%, ranges overlapping)
while the per-case median said 1.5–2% slower with **17 of 19 cases slower** (sign test p ≈ 4e-4). One
aggregate delta would have been wrong in **both** directions on the same data — report per-case
distribution and a sign test alongside the score.

## C97: A WIDE LAUNCH PARAMETER BLOCK IS A FIXED PER-LAUNCH TAX  🔴 **measured twice**

**C96** says the cost of a general path is its launch *symbols*, not its code size. This is the other
half: the **width of the launch argument block** is a real, measurable cost, and it lands entirely on
the small cases.

Measured on `quant_matmul`: the same correctness, implemented with a wide ABI, cost **−0.63** against a
0.272 control range — outside the band, so real. Per-case ratios localised it exactly:

| cases | ratio |
|---|---|
| small (16, 10, 11, 14) | **0.900 / 0.912 / 0.913 / 0.912** |
| large (19, 4, 6) | 0.984 / 0.988 / 0.993 |

That is a **fixed per-launch** cost from **13 extra kernel arguments**, not a per-work cost. Two
ablations inside the wide design were both null (removing host `std::vector` churn; replacing
by-reference out-params with a value return) — only **narrowing the ABI to two strides** recovered it,
and the narrow version measured **+0.016**, a null.

**Rule: when you add generality, pass the minimum the device needs.** Derive what you can on device
from what is already there rather than adding arguments. And measure the small cases separately — an
aggregate score hides this completely, because the large cases barely move.

## C98: A PROBE'S OWN dtype CHOICE CHANGES WHICH THRESHOLD `compare_tensors` USES

`compare_tensors` selects its threshold from the **dtype of the tensors you hand it**. Upcasting
`ai_output` to fp64 before comparing makes it pick the **float64** threshold `2^-13` instead of bf16's
`2^-7` — **48x stricter than the real gate**.

Measured on `quant_matmul`: that mis-bucketing manufactured one **false FAIL** (a 1-ULP bf16 tie), and
— worse — once corrected it **exposed a real failure the strict call had been hiding**. A wrongly
bucketed comparison is not merely conservative; it moves which cases land in which fallback band, so it
can conceal as well as invent.

**Rule: hand the comparator the operator's own dtypes, never a promoted copy.** And two companion
traps found in the same pass:

- **A single-element positive control is too weak for this comparator.** A `+1000` spike on 1 of 16,384
  elements returned `passed=True` (MERE 1.0e-2, MARE 1.6e2) — the cancellation clause legitimately
  forgives one outlier when the reference is noisy. Scale a **tenth of the output** by 1.5x instead.
- **Add a NEGATIVE control that aborts the scan if the probe rejects something the evaluator passes.**
  A probe that transcribed the fp64 oracle for *both* the oracle and the same-precision reference made
  the reference far too accurate, tightening the cancellation clause, and reported **122 of 124 non-PASS
  including shapes the real evaluator passes 20/20**. Import the task's own `golden.py` rather than
  re-deriving either reference.

## C99: THREE MORE WAYS A FITTED CONTRACT HIDES, ALL FOUND IN ONE PASS

From `weight_quant_batch_matmul`, whose accepted set equalled the `实测` column on **all three** shape
axes — `M` 1–128 of a declared 1–512, `K` only multiples of 256 of a declared 1–65535 (**255 of every
256 values refused, including the declared maximum**), `N` only multiples of 128:

1. **A cap can be architectural rather than a tile knob — check before widening.** `M > 128` was not a
   tuning choice: `MR = 2M`, so M=512 overflows L0A (256 KB vs 64 KB), L0C (1 MB vs 128 KB) *and* the
   double-buffered L1. The fix was **host-side M blocking**, exact because the operator is
   row-independent, and M<=128 still takes exactly one launch with the old geometry. Widening the tile
   would have been wrong; widening the *contract* was right.
2. **A padded store needs its conversion tile at the VALID width, not the padded one.** Widening the
   int8 conversion tiles made `TCVT` write `sw` lanes from a `kw`-wide source. Detectable with
   `x=1, w=1, scale=1` so `y` must equal `K`: K=1000 gave **974**, K=1023 gave **1011**. Keep conversion
   at the valid width and widen only the store.
3. **The GM row-stride field is 16 bits, and a declared maximum can sit exactly on it.** A flat
   `[MR, Kp]` workspace put `Kp` in that field. With `x=1, w=1, scale=1/K` so `y` must be 1.0:
   `Kp=65024` and `65280` gave 1.000000; `Kp=65536` gave **1.007812**, *identical* for K=65281, 65300
   and 65535 — a stride artifact, not precision loss (the surplus 1/128 is 512 lanes, one chunk
   double-counted). Cured by a chunk-major workspace with a constant stride.

**And note what was NOT the hole:** the quantisation axes looked like the obvious suspect (per-channel
vs per-group, group sizes, scale/offset dtypes) and were already correct, because `numel()`-based checks
accepted the declared 2-D `[1,N]` form. The shape axes were the defect. Audit every axis; do not stop at
the one that looks most likely.

## C100: NEVER EXPRESS A MASK, GUARD OR SKIPPED TERM AS ARITHMETIC ON THE VALUE  🔴 **CRITICAL**

**The rule: `keep * x` must be a SELECT. A skipped tap must be SKIPPED, not weighted by zero. A
staged reference expression must not be re-associated. An accumulator must be wide enough that it
cannot reach `Inf` where the reference stays finite.**

This is one defect generator that produced **9 failing hidden cases across 6 operators** through five
different mechanisms. It is worth a rule rather than nine patches.

**Why it happens.** The reference does something *structural* -- a branch, a `masked_fill`, a select,
an exact-precision evaluation. We do the *arithmetically equivalent* thing: multiply by a 0/1 mask,
add a finite penalty, fold a coefficient, accumulate narrower. **Every one of those substitutions is
exact over finite floats and unsound over the IEEE extended reals.** It introduces `0*Inf`, `0*NaN`,
`Inf-Inf` or `0/0` at a position whose correct *finite* answer is unaffected. Reduced to primitives
there are exactly **two** offending operations:

```
0 * non-finite          non-finite - non-finite
```

**The signature, and why it is invisible to every metric you would normally use.** These cases report
`MERE = MARE = 0.000000` with *every finite element bit-clean*. That is not bit-exactness: the
comparator returns **dataclass defaults** because it exited at the NaN-position gate before measuring
anything. **A MERE/MARE harness structurally cannot see this class.**

Two consequences read directly from `compare.py`, both of which kill the obvious hypotheses:

- The NaN gate (L390-399) is a **hard early return**. Nothing downstream of it runs -- not the stage-1
  fast path, not the band analysis.
- **Inf handling (L407-450) is NOT a gate.** An `Inf` on one side only is replaced by `max_finite` and
  the comparison **continues**; only `both_inf` with *opposite signs* fails, with a different message.
  So **overflow-to-`Inf` on a narrowing store cannot produce this failure**, and neither can a finite
  `-1e30` sentinel where the reference emits `-inf`. A finite sentinel reaches this gate only if the
  `-inf` it replaces would have produced a **NaN downstream**: the sentinel is an upstream cause, the
  gate is always a NaN.

**The five mechanisms, so you can recognise them while writing rather than after a hidden run:**

1. **Fold / re-association.** A staged reference expression algebraically folded into fewer ops.
   Exact on finite inputs; different NaN/Inf pattern on non-finite ones.
2. **Zero-weight tap or zero-mask multiply.** Measured on an interpolation: our blend computed
   `x[i0]*(1-lam) + x[i1]*lam` with `i1 = min(i0+1, S-1)`, **both taps always multiplied**. A one-hot
   extraction showed the reference agreed on *every weight* but **did not evaluate a zero-weight tap at
   a different index**. At a position whose inputs were **all finite**, ours NaN'd from
   `0.0 * x[far] = 0*inf`. Exhaustive `S,T in [1,12]^2`: **45/530 fail**, trigger set `S==T`, and
   **dtype-independent** -- which is how you tell it from mechanism 4.
3. **Additive finite penalty instead of an overwrite.** The reference does `masked_fill(-inf)`; we add
   a large negative bias. Same finite result, different non-finite algebra.
4. **An fp32 intermediate overflowing into subtract-the-max.** Raw scores overflow fp32 to `Inf`, then
   `Inf - Inf` in the row-max subtraction NaNs the **whole row**. **Structurally bf16-only**: the
   analytic threshold at `D=128` is `sqrt(3.4e38/128) = 1.63e18`, measured clean at `1e18` and
   1920/2048 NaN at `1e19`. **fp16 can never reach it** (`65504^2 * D = 5.5e11`); bf16 can
   (`max 3.39e38`). That prediction is the discriminator -- if a failure is bf16-only, suspect this;
   if it fires at every dtype, suspect mechanism 2.
5. **A guard the reference has and we do not.** Check the reference's guard; **do not assume it leaves
   the case undefined.** One loss kernel computed `loss_i = keep_i * (max_i + log(sum exp) - x_t)`
   while the reference **never evaluates an ignored row at all** and writes 0 -- so an ignored row with
   any non-finite logit gives `0*NaN = NaN` on our side and `0` in the reference, and `sum`/`mean`
   inherit it. **The fix is to apply `keep` as a select, not a multiply** (or zero the bracket first).

**A corollary worth its own line: size every magnitude constant to the ACTUAL dtype, never to fp16.**
One attention kernel carried `MASK_RAW_PER_D = 2*65504^2` with the comment *"Taking M = 65504 (the
fp16 maximum)"*. On bf16 the true bound is `2*(3.39e38)^2`, so that one constant produced **two**
independent bf16 defects -- a mask leak above `|x| ~ 65504` and the mechanism-4 overflow above
`|x| ~ 1.63e18`. This is **C86's shape-fitting failure applied to the value axis**: a constant fitted
to one dtype's range, used on a dtype with a vastly wider one.

**The test that finds the whole class, and it costs no device time.** Run the reference on **CPU in
fp64**, run the kernel, and **diff the NaN masks only** -- over the four distributions the benchmark's
own generator can emit: `[-inf, inf]` (about 5% `+Inf`, 5% `-Inf`), `[nan, nan]` (about 50% NaN),
dtype-boundary magnitudes, and zeros. Note the generator **does** have a NaN path even where the spec
disclaims one, so non-finite input is never hypothetical. Two warnings from building this probe:

- **Both sides NaN everywhere is a PASS**, so a 50%-NaN generator alone does not discriminate. It was
  the `[-inf, inf]` and `x1e20` distributions that separated mechanisms 3 and 4.
- The probe needs a **positive control** plus negative controls for the two hypotheses that the gate's
  structure already rules out (one-sided `Inf`, finite sentinel vs `-inf`), or it will report a clean
  sweep it did not earn.

## C101: TWO MMAD TILE-VALIDITY FACTS THAT SILENTLY RETURN WRONG DATA  🔴 **CRITICAL**

Both were measured on real hardware, and **both were guessed wrong on the first attempt.** Neither
faults, neither warns: the kernel runs and the numbers are wrong.

**1. An MMAD whose RIGHT tile has `ValidCol < Cols` returns wrong data.**

Isolated on an *in-contract* control shape with a forced-general build, so nothing else differed: the
query branch (`ncols = 512 = 2x256`) was correct at `mere 1.4e-6`, while the kv branch (`ncols = 192`)
came back **garbage at `mere 3.3`** -- same code path, only the right-tile `vn` differing.

**The fix is not to widen the tile. Keep the Right tile fully valid and let the N tail ride on the
store**, which is exact: `TStoreAccNz2nd` takes its L0C stride from `validRow` alone and its width from
`validCol`, so a narrower store is a correct projection of a full-width accumulator.

**2. A partial K panel is correct only when `ceil(vk/16)*16` equals the tile's PHYSICAL K width.**

Measured: `D=127` in a 64-wide panel **passes** (64+63, and `ceil(63/16)*16 = 64` matches). `D=16` in a
64-wide tile **fails at `mere 6.3e-2`**, as do `D=17`, `He=257`, `Hcq=257`. `He=320` is the clean
control. So the rule is not "K must be a multiple of 16" and not "any K works" -- it is that the
rounded-up valid K must **fill** the panel it is declared in.

**Fix: template the K panel width** -- 64 when `K % 64 == 0`, else 16 -- rather than padding the data or
moving the base pointer.

Both of these are why a general path needs a **forced-general** correctness run: a build where every
phase is pushed onto its general path even for shapes the fast path would take. On one operator that
run (4/4 shapes x 2 seeds) is what found both bugs, and the ordinary declared-surface sweep did not,
because the fast path shadowed them.

## C102: INSIDE ONE LONG-RUNNING KERNEL, CODE SIZE COSTS EVEN WHERE NEVER EXECUTED

**This is the inverse of C96, and both are true of different situations.** C96 says the cost of adding
a general path is the number of added **launch symbols**, not its code size -- measured by a stub
ablation that deleted 45.7 KB of body while keeping the symbols and cost the same. That holds for the
**per-launch** case.

But inside a **single long-running MIX kernel**, the opposite was measured. Adding general paths as
device functions in the one existing launch -- **zero new launch symbols, zero new kernel arguments** --
still cost about 1%:

| arm | median | spread | sign test vs control |
|---|---:|---:|---|
| control (pre) | 69.6293 | 0.2333 | -- |
| `gen` (general paths, +61.8 KB device image) | 69.5359 | 0.5193 | **16/20 slower, p = 0.0118** |
| `stub` (branches kept, **bodies deleted**, +1.3 KB) | 69.6773 | 0.2751 | 14/20 *faster*, p = 0.115 |
| `control_repeat` (**same pre binary again**) | 69.6471 | 0.2973 | 12/20 faster, p = 0.5034 |

**Read the attribution carefully, because the score cannot see it.** The null band from the same binary
twice is **0.3283**, and `gen - control = -0.1023`, i.e. **0.31 of the null band -- not separable by
score at all.** Only the sign test resolves it, and the **stub ablation attributes it**: branches cost
nothing (`stub` is if anything faster), while `gen` vs `stub` is 15/20 slower at `p = 0.041`. So the
cost is **body bytes**, in code that never runs for the measured shapes -- an instruction-cache or
image-locality effect, not a branch.

**Consequences for how you add generality:**
- Prefer **device functions inside an existing launch** over new launches (C96/C97 still apply, and a
  wide launch argument block is its own per-launch tax).
- But then **watch instantiation count**, because that is what multiplies body bytes. On the measured
  operator the lever was six instantiations (`NPL x CT x KGT`) reducible to four by making `NPL` a
  runtime loop bound.
- **Always run the stub arm.** Without it, a ~1% regression is indistinguishable from "branches are
  expensive", which would have sent the next pass optimising the wrong thing.

## C103: `TAXPY`'s SCALAR IS TYPED BY THE **SOURCE** TILE, NOT THE DESTINATION  🔴 **CRITICAL**

`pto`'s signature is

```cpp
TAXPY_IMPL(dst, src, typename TileDataSrc::DType scalar)     // npu/a2a3/TAxpy.hpp:132
```

so the scalar is **converted to the SOURCE tile's dtype before the multiply**. An fp32 destination
does not protect it. With `dst` fp32 and `src` `half`, an fp32 weight is silently rounded to fp16 --
**4.9e-4 relative** -- and on data with `|x| ~ 65000` that became `max_diff = 32.0`.

**It accounted for 7 of 8 remaining failures on one operator**, and the tell was an **asymmetry**: the
row combine used the `TAXPY` wrapper and lost the weight, while the column combine called the raw
`vaxpy` with an fp32 scalar against an fp32 source and did not. Same kernel, same weight, two
precisions.

**Fix, when the source must stay 16-bit** (it often must -- a 16-bit operand is what a Cube MMAD or a
narrow store requires): split the fp32 scalar into two 16-bit terms and issue two `TAXPY`s.

```cpp
if constexpr (sizeof(T) == 2) {                 // leave the fp32 path alone
    const T wh = static_cast<T>(w);
    const T wl = static_cast<T>(w - static_cast<float>(wh));   // exact, by Sterbenz
    TAXPY(dst, src, wh);
    pipe_barrier(PIPE_V);                       // same-pipe RAW on dst, see C48
    TAXPY(dst, src, wl);
}
```

That is ~22 mantissa bits: weight error `4.9e-4 -> 2.4e-7`, a **2000x** improvement where ~16x was
needed. `w - (float)wh` is exact by Sterbenz's lemma, so the pair loses nothing. **Cost measured
null** -- and only because a third arm caught it: the nine touched cases showed geomean **+2.4%** with
a sign test at **p = 0.0195**, and a **byte-identical rebuild of the control read +2.2% on the same
nine**, leaving an attributable **1.002**. Two arms would have banked a phantom 2.4% regression.

**Generalise the audit, not the fix:** any `pto` call taking a scalar may type it from a tile rather
than from the literal. Check the signature in the header before assuming an fp32 scalar survives, and
suspect this class whenever **two code paths computing the same quantity disagree only in precision**.

## C104: ALIGN-MODE DMA DOES **NOT** ZERO-FILL ITS PADDING  🔴 **CRITICAL**

`copy_gm_to_ubuf_align_b16` / `_b32` take `leftPadding` / `rightPadding`. **They do not zero those
columns -- they leave whatever was in UB.** Any kernel that relies on the pad arguments to supply zeros
for out-of-image columns reads residue.

Measured consequence on one operator: output columns `wo = 0` and `wo = Wo-1` were **NaN on every
16-bit case that had padding** -- 12 of 20 cases failing `NaN位置不匹配` with `MERE = MARE = 0`, i.e. the
NaN-position gate of **C100**. The only 16-bit survivors were a 1x1-pad-0 case and an all-NaN case; the
fp32 case showed the same residue as an inf/Naha placement difference at the same two columns.

**Fix: scrub `[0, lp)` and `[lp + len, Wc)` explicitly after every align-mode DMA.** Do not try to
clean it up later -- see C105: `vmin`/`vmax` cannot remove a NaN.

**And this is the defect that device 0 hides.** The operator's recorded score of `67.57 at 19/20` was
taken on **physical device 0**, whose UB happened to be zero, so the residue read as zeros and the
kernel looked correct. Re-measured on a good card the same commit scores **38.894 at 7/20**, failing
identically. The excluded card does not only compute wrong answers -- **it can make a broken kernel
look correct**, which is the more dangerous direction. Any recorded number whose script did not set
`ASCEND_RT_VISIBLE_DEVICES` is suspect, because the evaluator always runs logical 0.

## C105: `vmin`/`vmax`/`vmins`/`vmaxs` PROPAGATE NaN BUT CLAMP Inf

Probed on a2a3: `inf -> +-1e30` (clamped), `NaN -> NaN` (propagated).

**So a NaN cannot be scrubbed after the fact -- it must be prevented.** A saturating clamp is a valid
way to keep an *infinity* out of an accumulator, and no way at all to remove a NaN that already exists.
Plan the order accordingly: prevent the `0 * Inf` / `Inf - Inf` (C100), then clamp.

**Corollary, measured:** an operator's **bias** is inside its declared value range too. One case
generates `+-inf` biases, and an infinite seed made the first compensated residual `(bias - s) + pC`
an `Inf - Inf = NaN`, poisoning all 2528 positions where the golden is `+-inf`. **Saturating the bias
seed** took that arm from 19/20 to 20/20. Audit the value range of every *parameter*, not only of the
data tensor.

## C106: `TFusedMulAdd` / `TMulAddDst` ARE FUSED IN NAME ONLY -- NEITHER `vaxpy` NOR `vmla` SINGLE-ROUNDS

Probed with the standard discriminator `a = b = 1 + 2^-23`, `c = -(1 + 2^-22)`: a true FMA leaves
`2^-46`, two roundings leave `0.0`. **All three arms -- `vmla` with a full `src1`, `vmla` with a
stride-0 broadcast `src1`, and `vaxpy` with a scalar -- returned `0.0`.** So on the Vec pipe there is
**no single-rounding multiply-add available**, whatever the intrinsic is called. (A stride-0 `src1`
broadcast does work, so that part of the idiom is fine.)

**This does not contradict C83** -- keep the two straight, they are different units:

| | behaviour | consequence |
|---|---|---|
| **scalar** unit (C83) | **contracts** into an FMA, and `-ffp-contract=off` does **not** stop it | costs 1 ulp against a reference that did not fuse; needs a `volatile` launder |
| **Vec** pipe (C106) | does **not** fuse, ever | you cannot *gain* a rounding; a 10-term sum costs 10 roundings |

**What follows for accuracy work:** when a CPU model shows a case needs single-rounding, you cannot get
it from the Vec pipe. The routes that do work, measured on a 10-addition accumulation that failed
`6/2 of 881` with two roundings: **Kahan** (`2/2`, PASS) or a **compensated two-float** accumulator.
Model it on the host first -- a CPU model of the exact tap order reproduced both the shipped numbers
and a previously reverted attempt **bit-for-bit**, which is how "no ordering of ten fp32 additions can
pass" was shown to be true of *orderings* and false of the operator.

## C107: A CAPACITY GUARD THAT **GIVES UP** IS A SIXTH FITTED-CONTRACT MECHANISM

C86 lists five ways a supported set gets fitted to the visible cases. Here is a sixth, and it is the
most deceptive because it **reads like a tile ladder and behaves like an unchecked buffer**.

One path staged `C*VP` rows into a buffer bounded by `NROW = DSTC = 1024` with **no channel blocking**,
unlike its two sibling paths which had it. Its only protection was an **R-shrink loop that floors at
32** -- so the bound holds while `C <= 32` and silently stops holding above it. Declared `C` goes to
**512**, i.e. 16x past where the guard works. At `C = 257` fp32 nearest the device took an aivec
exception: `retCode = 0x31`, `errorStr = "MTE accesses an invalid GM address"`.

**The guard is not a rejection and not a clamp. It is a best-effort shrink that runs out of room and
then proceeds anyway.** Grep sees a loop that adjusts a tile, which looks like correct engineering.

**And the reason it survived every earlier sweep is the part to internalise: the fault region is
NON-MONOTONE, because a cost model chooses the path.**

```
C =  33 FAULT    64 PASS   128 PASS   256 PASS
C = 257 FAULT   300 FAULT  384 PASS   512 PASS
```

**A powers-of-two sweep returns CLEAN. A sweep that checks only the declared maximum returns CLEAN.
Both are wrong.** The tell was `C = 33` -- one past a boundary, the same probe shape that finds
alignment defects. **Whenever a cost model or a heuristic selects between paths, the reachable-failure
set is not an interval, so sweep off-by-one values around every threshold in the chooser, not just the
extremes of the declared range.**

**Audit rule:** when one path among siblings lacks a blocking loop the others have, that asymmetry is
the defect -- the same signal as C103's row-vs-column precision asymmetry. Ask of every capacity guard:
*what does it do when it cannot shrink far enough?* If the answer is "continues", it is this class.

## C108: A UB TILE BASE BUILT FROM A RUNTIME PRODUCT IS ONLY SOMETIMES 32-BYTE ALIGNED  🔴 **CRITICAL**

```cpp
TASSIGN(ge, UB_GE + static_cast<int32_t>(k * srun) * 4);   // aligned only when srun % 8 == 0
```

A Vec instruction requires a **32-byte-aligned** UB address. A base formed as
`UB_BASE + <runtime expr> * sizeof(T)` is aligned only when the expression happens to make the byte
offset a multiple of 32 -- and a **declared axis usually does not**. Here `srun` is the *product of the
spatial dims*, declared `1~512` over 2D-5D input, so **seven of every eight values are misaligned.**

It raises `507035`, *"The vector core execution is abnormal"* / *"The UB address accessed by the VEC
instruction is not aligned"*. One such case **took down a shared runner and cascaded 21 more**.

**The predicate was established exactly -- 81/81 points, zero mispredictions:**

```
FAULT  <=>  mode == 1  AND  cg >= 2  AND  srun >= 2  AND  (srun % 8) != 0
cg == 1   -> clean (k is only ever 0, so the offset is 0)
srun == 1 -> clean (a separate branch avoids the k*srun offset)
```

**Measured, `cg=2`, fp16:** `1 ok | 2..7 FAULT | 8 ok | 9 FAULT | 16 ok | 17 FAULT | 32 ok | 33 FAULT |
64 ok | 65 FAULT | 129 FAULT`. **Smallest reproducer: `x = [2,2,2]` fp16, `num_groups=1` -- eight
elements.** Value range is irrelevant; unit-scale data faults.

**This is C107's sibling, and it sharpens the sweep rule. A POWERS-OF-TWO SWEEP OF THAT AXIS IS CLEAN
AND WRONG** -- 8/16/32/64/128 all pass. **The tells are 9, 17, 33, 65, 129: sweep the `% 8` residues of
any axis that multiplies into a UB offset**, not its extremes and not its round numbers.

**The fix that costs nothing:** walk `k` **descending** and start each channel's tile at the largest
8-float boundary `<= k*srun`, lengthening it to `(k+1)*srun - beg`. The extra head elements spill
backwards into channel `k-1`'s span, and `k-1` is expanded *afterwards*, so it rewrites exactly those
elements with its own value. **When `srun % 8 == 0`, `beg` IS `k*srun` and `len` IS `srun`**, so every
shape that already worked is written **byte-identically** -- only the loop order changes, and those
writes are disjoint. Measured: per-case `t_hw_us` **exactly equal on 20 of 20** visible cases, and the
raw output bytes SHA-256 identical, with a live positive control proving the comparison can see a
difference.

**Two process points this cost:**
- **Why no gate saw it.** `mode 1` needs `rowlen <= 2048`, i.e. a tiny row, and the only two mode-1
  visible cases dodge it -- one has `srun=128` (aligned), the other `srun=1` (the special-cased
  branch). A declared axis was fully exercised in *range* and never in *residue*.
- **It was NOT a regression.** The pre-existing base build faults identically, 10/10 matching points
  on both sides of the boundary. Suspecting the newest code path is a reasonable prior and it was
  wrong here -- **test the base build before attributing a fault to the latest change.**

## C109: A TSTORE OUT OF AN L0C `Acc` TILE WRITES THE FULL 16-ROW FRACTAL  🔴 **CRITICAL**

Not the declared valid rows -- **the whole fractal**. The overrun lands on whatever the layout puts
next, so the symptom is wherever that happens to be.

**Measured instance.** An operator accepted `projSize` 1..15 and was **silently wrong** for them
whenever `numLayers >= 2` -- relative error **~1.0** at `P = {1,2,5,8,15}` for `L = 2,3`, and ~1e-6 at
`P = 16`. The store overran `xt[l+1]` and overwrote the **K-augmentation ones row** that the folded bias
depends on (and, bidirectionally, the next direction's slot).

**Every observation follows from that one cause**, which is how you confirm it rather than guess:
`P % 16 == 0` works; `P == 0` works (a Vec/UB store honours the row count); and **`L == 1` works because
`xt[1]`'s ones row is only read by iproj for layer 1** -- which is precisely why **every single-layer
`projSize` probe passes and hides it**.

**Fix:** align the destination slot to the fractal (`Eslot = alignUp(E,16)`) and place the dependent row
past it (`D*Eslot`). `P == 0` or `P % 16 == 0` then changes nothing.

**Audit rule:** wherever an `Acc` tile is stored with a valid-row count below 16, ask **what occupies
the next 16-row-aligned bytes**. If anything downstream reads that region, it is already corrupted.

## C110: DUPLICATED GEOMETRY BETWEEN DRIVER AND KERNEL IS A SILENT OOB WRITE

A host constant and a kernel constant describing the same geometry **will** drift. A stale host
`Kpmax` of 64 against the kernel's 128 produced an out-of-bounds GM write -- **no error, no fault.**

**The symptom is the diagnostic:** results that vary **run to run** and *converge* as the workspace
fills with its own residue -- measured `4.267 -> 5.018 -> 5.212 -> 5.212`. A stable wrong answer is a
logic bug; a **drifting** wrong answer that settles is uninitialised or overrun memory.

**Rule:** a geometry constant lives in exactly one place. If the ABI forces it into two, assert the
relationship at the boundary and write the rule at **both** sites. Dump the driver's cached workspaces
when a wrong answer will not reproduce.

### C125: A SATURATING TRANSCENDENTAL REBUILT AS `cheap_form + correction` LEAKS THE CORRECTION'S FLOOR WHERE THE CHEAP FORM WAS ALREADY EXACT  🔴 **CRITICAL**

PTO has **no `TTANH` and no `TEXPM1`** (verified: a search for `tanh` returns nothing; Elementwise
Tile-Tile carries `TEXP`/`TLOG`/`TDIV`/`TRECIP` only), so `tanh` must be built. The two-step trap below
cost a real operator half of a 22-case failure class, and **step 2 is the part nobody checks**.

**Step 1 -- the half-angle form cancels near zero.** `tanh(z) = 2*sigmoid(2z) - 1` with
`sigmoid(2z) ~= 0.5 + z/2` carries an **ABSOLUTE** error of ~1 ulp at 1.0 (**measured 7e-08, FLAT in
z**) while the true value is ~`z`, so the **RELATIVE** error is unbounded. With decaying activations
the signal falls ~10x per layer while that error does not: measured `rel_fro` **3.2e-05 at
numLayers=3 rising to 1.18 at numLayers=8**. No algebraic rearrangement helps -- `(1-u)/(1+u)` with
`u = exp(-2z)` has the identical cancellation, because `TEXP` destroys the information (`u` is
accurate to a *relative* 1e-7, which near `u == 1` is an *absolute* 1e-7 in `1-u ~= 2z`), and it
additionally returns **NaN for `z <= -20`** without a guard. A small-argument polynomial is mandatory.

**Step 2 -- the mask-free composite then breaks saturation.** The fix for step 1 is
`tanh(z) = E(z) + [poly(zs) - E(zs)]`, `zs = clamp(z, +/-0.25)`, chosen mask-free so it needs no mask
tile and no `TSELS` per site. But **for every `|z| > 0.25`, `zs` is pinned at +/-0.25, so the bracket is
a CONSTANT ~+/-6e-08** -- and `E(z)` alone **is exact at saturation** (`exp` underflows, `2*1-1 == 1`).
So the correction **de-saturates a `tanh` that should be exactly +/-1**. Measured absolute error,
shipped composite vs gated:

| \|z\| | composite | gated | torch |
|---|---|---|---|
| [0.5, 9) | 2.37e-07 | 1.78e-07 | 2.99e-08 |
| [9, 20) | 5.96e-08 | **3.05e-08** | 3.05e-08 |
| **[20, 1000)** | **5.96e-08** | **0** | 0 |

**It cannot be tuned away.** `|corr|` is set by `E`'s own ~6e-08 absolute floor (the `2*sigmoid-1`
cancellation) and that floor is **flat in the clamp value** -- identical at clamp 0.05, 0.25 and 1.0 --
so no `kTanhClamp` makes `1.0 + corr` round back to `1.0`. Retuning the clamp is also
hardware-fragile (it would depend on `TEXP` at one point).

**Rule:** when you rebuild a saturating function as `cheap + correction`, **gate the correction off
wherever the cheap form is already exact.** Arithmetic gate, 6 Vec ops, no mask tile, no extra UB, and
bit-identical on the small-argument path by construction (`d == 0` exactly there):

```cpp
TSUB(w, z, zs);          // d = z - zs : EXACTLY 0 where zs == z
TMUL(w, w, w);
TMULS(w, w, -1.0e6f);
TADDS(w, w, 1.0f);
TMAXS(w, w, 0.0f);       // 1.0 at |d|==0, exactly 0.0 for |d| >= 1e-3
TMUL(dst, dst, w);       // d^2 overflowing to inf still yields gate 0
```

**And the limit of the whole exercise, measured -- read this before budgeting accuracy work.** At a
**catastrophic-cancellation position** (`c = f*c_prev + i*g` where two O(1) terms cancel to a few
ulps) the residue is an exact integer multiple of one operand ulp, `2^-24 = 5.96e-08`. Against a
comparator denominator of `|golden| + 1e-7`:

```
golden == 0        : MARE = 5.96e-08 / 1.0e-7  = 0.596   FAIL (limit 0.5)
golden == 1e-8     : MARE = 5.96e-08 / 1.1e-7  = 0.544   FAIL
golden == 5.96e-08 : MARE = 5.96e-08 / 1.6e-7  = 0.3735  pass
```

**ONE ulp of disagreement is already 1.09x-1.19x over the limit.** Passing such a position needs
**zero** units of difference, i.e. bit-exactness with the reference. Three measurements say that is
out of reach: the **exact fp64 answer rounded to the case dtype FAILS** (MARE 1.37 and 23.49 on two
configs); **PyTorch's own non-oneDNN fp32 path FAILS** (1.553, 2.652); and **two equally valid
references disagree by 75% of the entire budget** (oneDNN on vs off: MARE **0.3735** between the two
goldens, leaving an implementation 0.1265 of 0.5). Swapping a single transcendental moves MARE
0.81/1.03/1.07/1.27/3.6 in **no consistent direction**.

So split the class before you build: where the failure is **your own contamination** it is
convertible (measured 24x-92x MERE improvement, 12 fail->pass / 1 pass->fail over 334 configs); where
it is **amplification of an unavoidable ulp** it is unwinnable at any accuracy. A **perfect** tanh
converted only **7 of 16** failures -- so "make it more accurate" has a measured ceiling well below
"all of them". See **C115** (judge by failure-set SUBSET, not count -- this fix is 12:1 and still not
a strict subset: one config went 0.3177 -> 0.5012, 0.24% over, on a pure rounding lottery).
