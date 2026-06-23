# Reviewer Mode Protocol

This document governs the second-pass reviewer/fixer role. Read this only
when invoked as the reviewer over a draft kernel.

---

## Core Principle

Preserve semantically correct portions of the draft. Apply the minimum
rewrite needed to remove forbidden patterns and compile/validator failures.
Do not introduce a brand-new lowering family unless the draft family
cannot be repaired safely.

---

## Repair Workflow

```
1. □ Read the draft kernel completely
2. □ Classify the draft's archetype (vec_only, cube_only, etc.)
3. □ Run the CRITICAL rule scan (C1-C11) — fix all violations
4. □ Run the STANDARD rule scan (S1-S10) — fix all violations
5. □ Verify banner accuracy (A1)
6. □ Verify UB budget still holds after any tile changes
7. □ Verify single launch_* symbol
8. □ Return the repaired kernel as raw C++ (no JSON, no markdown)
```

---

## CRITICAL Repair Rules

These take priority over all other considerations.

**R-C1. GM access repair.** If any `__gm__` pointer is scalar-indexed,
wrap it in `GlobalTensor` and convert to TLOAD/TSTORE. → SKILL C1

**R-C2. Include/namespace repair.** Remove any indirection guard macros.
Ensure `#include <pto/pto-inst.hpp>` and `using namespace pto;` are both
under `#if defined(__CCE_AICORE__)`. Ensure the `AICORE` fallback is defined. → SKILL C2

**R-C3. Host/device split repair.** Ensure exactly one `extern "C" __global__
AICORE void launch_*` definition. Ensure `call_kernel` is wired to it.
Do not duplicate or rename launch symbols inconsistently. → SKILL C3

**R-C4. Type surface repair.** Replace any invented aliases with the closest
approved family (COOK-§0.5). Do not mix partially qualified and partially
invented APIs. → SKILL C4

**R-C5. Cube layout repair.** Fix BLayout/SLayout combinations per COOK-§8.5.
Add `TRESHAPE` for transposed operands per COOK-§8.7. → SKILL C5

**R-C6. FFTS bootstrap repair.** Add `set_cross_flag` bootstrap signals
before the first `wait_flag_dev`. Check that A2/A3 V→C flags account for
both Vec subblocks. → SKILL C6

**R-C7. UB budget repair.** Recompute UB footprint after any tile changes.
Add/update `static_assert`. → SKILL C7

**R-C8. Vec subblock repair.** If both vids use the same UB addresses,
add `vid != 0` early return or partition addresses. → SKILL C8

**R-C9. Barrier repair.** Add `pipe_barrier(PIPE_ALL)` after any TLOAD or
TSTORE that lacks one. Remove blanket barriers that fire after every op. → SKILL C9

**R-C10. AICORE qualifier.** Do not emit an empty-host `AICORE` fallback
(`#define AICORE`); keep `AICORE` mapped to `__aicore__` so launch
entrypoints remain correctly qualified.

---

## STANDARD Repair Rules

**R-S1. Scalar loop replacement.** If the dominant math is implemented as
scalar `for` loops over GM pointers, replace with the nearest PTO tile
pattern from the cookbook. → SKILL S1

**R-S2. Scalar approximation removal.** Replace any `exp_scalar`, polynomial
transcendental helpers, or raw `exp()`/`expf()` calls with `TEXP` on tiles. → SKILL S2

**R-S3. Contraction site enforcement.** If `@`/matmul/einsum sites are
implemented as scalar loops or left as `Lowering gap` comments, lower them
on the nearest Cube contraction path (COOK-§8.7 or §8.8). → SKILL S3

**R-S4. Dimension symbol restoration.** If runtime dimensions have been
frozen to guessed constants, restore the symbolic parameter and add a
runtime guard if needed. → SKILL S4

---

## Anti-Regression Rules

These must hold on every repair turn. Never introduce these patterns
as a fix for other issues.

| Anti-pattern | Why forbidden |
|-------------|---------------|
| `WF_HAS_PTO_STAGE_IMPL` or `PTO_STAGE_HAS_IMPL` guard macros | Indirection violates C2 |
| `!defined(MEMORY_BASE)` guards disabling PTO exposure | Use `__CCE_AICORE__` directly |
| Downgrading correction/recurrent to vec-only elementwise | Violates S3 (contractions required) |
| Ad-hoc host-fallback macros for prep gather/scan | Changes numerical behavior |
| Empty-host `#define AICORE` (no `__aicore__`) | Launch entrypoints lose qualification |
| NaN-producing placeholder arithmetic | Violates C10 |
| Uninitialized accumulation | Violates C10 |
| `#include "common.h"` helper paths for recurrent contractions | Use direct PTO surfaces |
| Dynamic `TileType::Mat` + `TEXTRACT` for recurrent unless cookbook-proven | Unproven surface |

---

## Semantic Preservation Rules

When repairing, preserve these invariants by stage trait class:

### Prep / Preprocess / Cumsum

- Query scaling applied exactly once — do not add or remove a scaling site
- `g_prefix` remains inclusive cumsum over BT
- Beta gather keeps lane-correct `[rows, heads] → [rows]` semantics
- Keep validated ND block-load + lane extraction gather shape
- Keep direct row-tile scan shape (TLOAD/TADD/TMOV/TSTORE)

### Correction / Transfer and Projection

- Seed uses anchored-difference semantics — do not switch to factorized seed paths
- Beta row vs beta column remain distinct applications (not collapsed)
- Closure is strict-lower before final diagonal/identity handling
- `w` and `u` remain true A-projection contractions
- When subfamily is `transfer_and_projection` or `correction_projection`,
  preserve both projection contractions:
  `w = A @ (exp(g_prefix) * k_chunks)` and `u = A @ v_chunks`
- Keep `BT`/`K`/`V` runtime-symbolic; do not freeze to guessed constants
- Do not downgrade to structural-only placeholders

### Recurrent / Chunk Scan

- Recurrent state update semantics preserved
- No output copy-through replacement (`o = u`-style)
- Required contractions remain tile/Cube based
- Preserve both output terms: `(q_i * exp(g_i)) @ S`, `Aqk @ v_i`
- Preserve decay + accumulate update of `S`
- Recurrent outputs must not degrade to copy loops

---

## Validator-Driven Repair

When the validator reports a specific failure:

1. Identify the failing port/rule from the validator output
2. Patch ONLY the failing surface — do not rewrite the entire kernel
3. Keep all other stage invariants intact
4. Verify the patch does not introduce any anti-regression pattern
5. If the fix requires a layout or type change, re-derive the UB budget

---

## Repair Output Format

Return only the repaired C++ translation unit:
- no JSON envelope
- no markdown fences
- no analysis prose
- no "I am inspecting local kernels..."
- the response is the file body

Keep the required banner sections (Stage role, Architecture / dataflow,
Key PTO ops used) on every repair output.
