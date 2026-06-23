---
name: pto-kernel-optimizer
description: "Optimize a CORRECT staged/fused compute kernel toward a performance target or a reference baseline, without breaking correctness or determinism. A general measure->decide->attack->re-measure method: per-work-unit cost (slope) decomposition, bottleneck classification, the right lever per bottleneck (remove redundant memory traffic, swap a wasteful algorithm, keep operands resident, compose lean parts, overlap independent work), and honest stop-criteria. Hardware/algorithm-agnostic; grounded for PTO/Ascend. Triggers: make the kernel faster, optimize kernel/pipeline, close the gap to a hand-tuned/reference baseline, reduce per-iteration cost, fusion optimization, kernel is correct but slow."
---

# Kernel Optimizer (general method, grounded for PTO/Ascend)

Take a kernel/pipeline that is ALREADY correct and make it FAST toward a target,
keeping it correct + deterministic at every step. This is the optimization phase,
distinct from generation: generation produces a correct kernel from a spec; this
skill takes a working kernel + a target and drives it down via a measure->decide->
attack LOOP, not a one-shot rewrite.

**The method below is general** — it applies to any staged or fused kernel on any
accelerator that has (a) a notion of repeated work units (tiles / chunks / blocks /
token-segments), and (b) distinct execution resources whose work can overlap (e.g. a
matrix/compute engine and a vector/elementwise engine, or compute vs memory-movement).
The last section, **Instantiation (PTO/Ascend)**, maps each step to the concrete
code-level rules for this codebase (`pto-stage-kernel-generator-v2/references/
cookbook.md`, `COOK-§8.6P #1-23`). Read the general method first; reach for
the instantiation when emitting code.

## When to use / when NOT
USE when a correct kernel/pipeline is slower than a target or a reference and you want
to close the gap. Do NOT use to fix correctness — that is generation/repair. Optimize
correct code only, and keep it correct at every step.

## Inputs
- The kernel/pipeline (and, if staged, the per-stage pieces) — correct + validated.
- A correctness oracle (a high-precision reference) and a determinism check.
- A way to measure latency at >= 2 problem sizes, on the real target.
- A baseline to chase: a reference implementation, or a target cost-per-work-unit.

## 1. The core principle: optimize the SLOPE, not the intercept
Total latency ~= `slope * (work units) + intercept`. The **intercept** is fixed
overhead (launch, fill); the **slope** is the per-work-unit steady-state cost. At
production scale (many work units) the slope dominates and the intercept is noise — so
a kernel can "win" a small-size benchmark purely on a lower intercept while losing by
multiples at scale. **Always optimize and report the slope**, fit across >= 2 sizes:
`slope = (lat@big - lat@small) / (units@big - units@small)`. A reference's real
advantage is almost always a leaner slope.

## 2. Measurement discipline (non-negotiable on noisy hardware)
- Measure with a tight, **within-process PAIRED A/B** (alternate A,B per repeat in one
  process); common-mode device drift cancels. A "win" that only appears unpaired /
  across sessions is drift, not a real speedup — re-confirm every win paired.
- Reproduce the **baseline + a known anchor** on the same device/run before trusting any
  number. Flush caches; use a device-side timer; serialize timed runs.

## 3. The campaign loop
1. **Decompose the slope.** Measure each stage/section standalone at 2 sizes -> per-part
   slope. The whole slope ~= sum of part slopes. This says WHERE the time is.
2. **De-risk before any expensive build (the highest-leverage check).** Lower-bound the
   achievable cost cheaply BEFORE building: noop one resource to measure the other's
   floor (see Diagnostics). The sum of the irreducible per-part floors is a hard bound
   no fusion/overlap can beat. **If that bound already exceeds the target, the planned
   approach is futile** — stop and attack the parts instead. One measurement can save a
   multi-day build.
3. **Attack the DOMINANT part.** Classify it (taxonomy below), apply the matching lever,
   re-measure paired. Repeat until no part dominates.
4. **Compose lean parts, don't merge-then-tune.** Make parts faster STANDALONE first;
   then compose them with the *lowest-coupling* mechanism that still removes the
   per-launch / inter-part overhead (e.g. a shared data layout + ordered chaining so the
   composition penalty is ~zero). Reserve a tightly-coupled in-place merge only for the
   one part where it is the sole remaining lever AND its cost is hideable.
5. **Overlap + residency.** Keep reused operands resident instead of re-fetching; overlap
   independent work of one resource behind another; run-ahead the work that does NOT
   depend on the previous step's result.
6. **Stop** at the irreducible floor or a wholesale-clone boundary (see Stop-criteria).

## 4. Bottleneck taxonomy -> lever (the decision tree)
Classify the dominant part, then apply the matching lever:
- **Redundant-traffic-inflated** — re-reads/re-writes/recomputes the same value, or
  inserts unnecessary commits/syncs between dependent same-resource ops -> REMOVE them
  (a lightweight local barrier suffices); hoist loop-invariant work out of the loop.
  *Usually the biggest, most common win.*
- **Algorithm-suboptimal** — doing far more operations than the problem needs (a naive
  O(n) where a blocked/recursive O(log n)-ish form exists) -> SWAP the algorithm. *The
  single biggest lever when one part dominates.*
- **Memory-transfer-bound** — genuine, non-redundant operand movement between producer
  and consumer -> keep operands RESIDENT and feed the consumer in place; restructure so
  a raw input reaches the consumer un-modified (fold scalars into the other operand).
- **Fully-hideable** — its work can run entirely behind another resource's work ->
  it is ~free under overlap; leave it, ensure it overlaps.
- **Serialization / recurrence-bound** — a loop-carried dependency forces ops to run in
  order -> keep the carried state resident, run-ahead only the NON-dependent operands,
  split the independent work across parallel sub-units. This is the irreducible long
  tail; partial only.

## 5. Diagnostics (the toolkit)
- **Noop-floor / lower-bound probe.** Stub out one resource's real work (keep its
  handshakes) to measure the other resource's intrinsic floor. If the bottleneck floor
  already exceeds target, the overlap-based plan cannot win — redirect.
- **Flat-floor-but-slope-drops = overlap (not op-count).** If a change drops the real
  slope while the noop floor stays flat, the win was latency-hiding/overlap; if both
  drop, it was op-count reduction. Tells you what you actually changed.
- **Paired A/B** — the only trustworthy measurement on a drift-prone device.

## 6. Stop-criteria (stop honestly)
- The sum of irreducible per-part floors already exceeds the target -> the gap is
  intrinsic per-part work; closing it means re-deriving the baseline's algorithms (a
  clone). Stop, document the path.
- The residual is a serial loop-carried recurrence -> that is the floor for ANY
  implementation, the reference included.
- A lever needs a wholesale architecture port for marginal/parity gain -> document it as
  the path, do not sink budget. Best-case parity rarely justifies a clone.

## 7. Hard discipline
- **Correctness + determinism gate EVERY step.** Re-validate vs the reference at small
  AND large sizes and re-run the determinism check after every change. A speedup that
  breaks determinism is not a speedup.
- **Never re-add a flush/barrier to mask a race.** A nondeterministic failure exposed by
  removing a sync is a real ordering/aliasing bug — fix by isolation/ordering, not by
  restoring the heavy sync. (The heavy sync was hiding a latent hazard.)
- **Honest-negative is a valid result.** "This lever provably cannot reach target"
  (with the measured floor + the bounding reason) is valuable. Keep the last-good
  version as the deployable fallback; never ship nondeterministic/regressed code.
- **Provenance.** You may study a reference's STRUCTURE (op sequence, data layout, sync
  protocol); the kernel must be GENERATED/derived, not copied verbatim.
- **Capture new levers.** Any new general lever -> codify it back into the shared
  pattern reference for the next run.

## 8. Instantiation (PTO / Ascend Cube+Vec)
The two overlappable resources are the **Cube** (matrix/GEMM) and **Vec** (elementwise)
engines; the work unit is a **chunk/tile**; cost is **us/chunk**. The general levers map
to concrete, validated rules in `pto-stage-kernel-generator-v2/references/
cookbook.md` (`COOK-§8.6P`):
- Redundant-traffic-inflated -> **#16** (drop cargo-cult GM commits / `TMULS` no-ops;
  `pipe_barrier(PIPE_V)` for Vec->Vec), **#17** (per-row GM round-trip -> block-resident
  scan; hoist masks).
- Algorithm-suboptimal -> **#13** (block-recursive fractal triangular inverse vs full
  Neumann), scan-as-matmul (#17).
- Memory-transfer-bound -> **#19** (lean named-UB prep->GEMM; fold a contraction-axis
  scalar into the matmul operand so the raw tensor loads Cube-direct).
- Compose lean parts -> **#21** (shared BSND layout + chain `launch_*` in one host
  `call_kernel`; stream ordering is the free seam — fused slope = sum of lean slopes).
- Residency + overlap -> **#20** (UB-resident recurrent state), **#22** (recurrence
  run-ahead of non-recurrent operands), **#23** (2-vid HalfC split; cross-vid coherence
  via cheap `dsb`, never bulk `dcci`; per-core workspace, not per-head).
- Cross-core correctness baseline -> **COOK-§8.6 3-rule hand-off** (same-pipe FFTS
  signal, no bulk `dcci`, distinct GM regions).
- De-risk -> **#18** (`Σ` per-stage cube-noop Vec floor before fusing). Diagnostics ->
  **#10** (noop-floor probe), **#14** (paired A/B), **#22** (cube-noop-flat=overlap).
Platform gotchas live alongside those patterns (e.g. `TTRI` fp32-only, `TROWEXPAND`
RowMajor, `TMOV Acc->Mat` half-dest, width-changing `TCVT` needs disjoint src/dst).

## 9. Worked example (the method's provenance)
The KDA fused kernel went from racy + 3.5-6.6x slower than its per-stage chain to
PRODUCTION PARITY with a hand-tuned reference (slope ~104 vs ~78 us/chunk; T=4096
1.017x; faster at small T): per-part de-inflation (4-8.4x on individual stages),
an algorithm swap (5.9x on the inverse), lean-then-compose fusion, resident state +
recurrence run-ahead. The one lever that would have beaten the reference (a 2-sub-unit
split of the recurrence) hit an irreducible cross-core coherence race on a per-head
workspace — the documented wholesale-clone boundary. That whole campaign IS this method.
