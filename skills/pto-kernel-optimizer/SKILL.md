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
- **Get a TUNED baseline at the MATCHED shape before drawing ANY bottleneck conclusion.**
  Generated-vs-generated comparisons only rank your own kernels — they cannot locate the
  hardware ceiling, and a generated kernel is routinely 3-15x off it. A vendor / hand-tuned
  reference (e.g. `torch_npu.npu_fused_infer_attention_score`, a vendor GEMM) run at YOUR
  exact shape — not the reference's most favorable config — is the only honest ceiling. If
  the reference is only available at a denser shape (e.g. it wants head_dim=128 but yours is
  32), run it AT your shape too and split the gap: `intrinsic` shape penalty (un-fixable
  in-kernel; e.g. a narrow K=32 contraction underfills the cube fractal ~4x) vs `fixable`
  codegen gap. Optimize only the fixable part, toward the matched-shape ceiling. When no
  single vendor op exists, COMPOSE a reference from vendor primitives (GEMM + native
  elementwise/softmax) — still an achievable ceiling. Fall back to the analytic roofline
  (`max(FLOPs/peak, essential_bytes/peak_BW)`) ONLY for a genuinely novel primitive, and
  then treat it as a THEORETICAL peak: it proves "far -> inefficient" but NOT "near ->
  optimal", so keep any "at the limit" call tentative and lean on the noop-floor /
  per-stage-sum diagnostics (§5) instead.

- **If a change touches a STRUCTURAL parameter as well as the optimization, build the
  variant that changes ONLY the structural parameter, and time it too.** Otherwise the
  attribution is guesswork, and it is routinely guesswork in the wrong direction.
  Worked instance: adding an MTE2 prefetch needed a second UB input slot, which only
  fitted after halving rows-per-item. Measured, canonical protocol, null control valid:

  | variant | ratio to vendor |
  |---|---|
  | A rows=4, no prefetch (starting point) | 1.120 |
  | B rows=2, no prefetch (**the control**) | 1.327 |
  | C rows=2 + prefetch | 1.084 |

  Reported as A -> C alone, the prefetch "bought 3%". Against its own control it is
  worth **1.224x**, and the tiling change is a 1.19x regression that had to be paid
  for. Those lead to opposite decisions about whether to keep hunting for UB room.
  A control is cheap: same source file, one compile-time switch.

- **If reducing `block_dim` makes the kernel FASTER, you are footprint-bound, not
  compute-bound.** This is a one-command diagnostic and it is decisive, because using
  fewer cores should always cost time unless the cores were never the constraint. Sweep
  `block_dim` down and watch both the wall time and the per-core-normalized cost
  (`ms * bd / bd_max`). Measured instance, a flash-attention kernel whose GM workspace
  was `block_dim x O(S)`:

  | S | bd=24 | bd=16 | bd=12 | bd=8 |
  |---|---|---|---|---|
  | 8192 | **7.70 ms** | 10.70 (1.39x) | 14.13 (1.83x) | 19.98 (2.59x) |
  | 16384 | 59.03 ms | **44.62 (0.76x)** | 55.40 | 77.83 |

  At 8192 the reduction costs what you expect. At 16384 it *gains* 1.4x, and the
  per-core cost falls 59.0 -> 25.9. Confirm with the implied bandwidth: if it exceeds
  the part's measured streaming ceiling at small sizes and drops below it at large ones,
  a cache was absorbing the traffic and has stopped. (Here: 1673 GB/s at S=8192 against
  an 811 GB/s HBM ceiling, then 695 GB/s at S=32768.)

- **A GM workspace that scales with a SWEPT dimension is a cliff waiting to happen.**
  Write the workspace formula down and check which terms carry the sweep axis. In the
  case above, `per_core = 2*(BQ*S*2 + BQ*S*2 + BQ*D*4)` had two of three terms O(S), so
  the footprint went 51 MB -> 771 MB across the sweep and fell out of L2 partway. The fix
  is structural (stream the axis, keep an O(1) tile) -- tuning `block_dim` only buys back
  part of it.

- **State the RANGE over which an optimization was validated, and expect it to invert
  outside that range.** The 2-slot GM workspace in that kernel was measured worth 1.24x
  at S=2048 and was costing 2.5x at S=32768 -- the same construct, a good decision inside
  the contract sweep and a bad one outside it. Nothing had asked, because the contract
  stopped at 2048. If your stage has a sweep axis, probe at least one point BEYOND the
  contract's top size before calling a structural choice settled.

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
6. **Stop** at the irreducible floor or a wholesale-clone boundary (see Stop-criteria),
   but not before the mandatory attempt budget below.

## 3.5 MANDATORY attempt budget: 10 attempts, and they go in the report

Every generated kernel gets an optimization campaign. It is not optional, and it is not
finished when the kernel merely validates.

**The budget is 10 measured attempts.** An "attempt" is a *change with a paired
re-measurement* — a hypothesis, a build, a number. Reverted regressions COUNT, and are
often the most informative entries; do not quietly drop them.

| stage archetype | minimum attempts | early stop allowed? |
|---|---|---|
| `mixed` (Cube + Vec, cross-core, composed/fused) | **10, always** | **No.** Run all 10 even when it is expensive. |
| `vec_only` | 10 | Yes — see the gate below |
| `cube_only` | 10 | Yes — see the gate below |

**Early-stop gate (single-engine stages only).** You may stop before 10 *only* if you can
show the kernel is at a **hardware limit**, with a measurement, not an argument:
* achieved bandwidth is within ~10% of the measured streaming ceiling (A2/A3: a PTO
  `TLOAD` extracts **~920 GB/s** from an out-of-L2 working set — see
  `PLAT-§ReadCeiling`; measure it for your shape, do not quote it), **or**
* achieved FLOP/s is within ~10% of the engine's measured roofline at this shape, **or**
* a noop-one-resource probe shows the remaining time IS the irreducible floor of the
  other resource.

State which gate fired and the number that fired it. **"It looks memory-bound" is not a
gate. A roofline percentage on its own is not a gate** — a marginal-cost probe once
disproved exactly that reasoning here (doubling every matmul cost 4.8%, while doubling
the Vec loads cost 32.2%, in a kernel diagnosed as Cube-underfilled).

**Two ways this gate has been got wrong. Both cost a whole campaign.**

* **Do not compare against the vendor's rate.** A vendor fused operator streams at
  ~1493 GB/s where PTO reaches ~920; the vendor's number is not a ceiling you can reach,
  so measuring yourself against it guarantees the gate never fires. `grouped_matmul_swiglu_quant`
  burned 13 attempts concluding "754 GB/s against a 1244 GB/s vendor rate, no gate fired"
  when it was already at the PTO ceiling by attempt 1. Compare against the **PTO** ceiling
  (`PLAT-§ReadCeiling`), and report the vendor gap separately as a platform fact.
* **Divide by the bytes you actually issue.** GB/s on *essential* bytes understates the
  rate whenever the schedule re-reads anything (that same stage issued **1.50x** its
  essential weight). Compute both; the gate uses issued bytes.

When this gate fires on an out-of-L2 stream, **try the uncached address alias FIRST**
(`PLAT-§L2Bypass`): a streamed operand read once and never reused should be loaded through
`ptr + rtGetL2CacheOffset()`, which measured **1.67x** (915 -> 1527 GB/s) on an identical
kernel binary and is bit-exact. Wider bursts, deeper rings, more cores and an NZ ABI are
all measurably flat and are not worth an attempt. After that, the remaining lever is
**shrinking the footprint** so the hot set fits L2.

Why `mixed` gets no early stop: its cost is a *composition* — cross-core handshakes,
seam sync, and overlap between two engines that a single-engine roofline does not model.
An engine can sit at its roofline while the composition wastes most of the wall clock.

**And the inverse, which is just as expensive to learn late: a stage can be far off its own
roofline and still be FREE.** Before optimizing any stage inside a composition, check
whether it is on the critical path — compare the composed time against the dominant stage
alone. On `grouped_matmul_swiglu_quant` the stage-2 Vec chain ran at 205 GB/s against a
~790 GB/s roofline and looked like the obvious target; it was optimized 1.15x (47.75 ->
41.54 us, validated), folded into the chain, and the chain moved **571.8 -> 572.5 us —
nothing**, because that stage runs on AIV behind stage 1's AIC weight stream and was
already fully hidden. The tell was available before the work: the chain (648.8 us) barely
exceeded stage 1 alone (622.8 us), so there was ~26 us of exposed stage-2 cost to win, not
46. **A roofline gap on a hidden stage is not an opportunity.**

**Why this rule exists.** Two regenerations of the same case differed by **1.16x vs
1.52x against the vendor** with *identical* correctness, purely because one run spent an
optimization pass on tile geometry and the other declared it out of scope. Run-to-run
variance in optimization effort was larger than every rule change between the two plugin
versions. An unoptimized kernel is not a result.

**RE-VALIDATE EVERY ATTEMPT ON THE DEGENERATE CASES, NOT THE PRODUCTION SHAPE.**
This is the most dangerous hole in a measure-decide-attack loop and it has now been
demonstrated. In one campaign, two attempts passed **15 consecutive paired
measurements** while being non-deterministically wrong, and a third measured as **the
fastest point of the entire campaign** while being wrong on 110,066 elements. A paired
A/B re-measurement checks SPEED; it does not check correctness, and the production
shape is usually the *least* discriminating one -- it is the shape with no ragged
tail, no empty group, no partial tile.

So after every attempt, before recording a ratio:
* re-run the FULL validation sweep, including the degenerate cases (empty group, zero
  rows, unaligned boundary, single-element tail, `items_per_lane >= 3`), not just the
  contract point;
* re-run the determinism check -- a race can pass one validation and fail the next,
  so a single clean run is not evidence;
* if it fails, mark the attempt `"correct": false` and keep it in the JSON. Do NOT
  drop it: a fast wrong attempt is exactly what the trajectory graph must show, and
  the plotter draws it as a red cross excluded from the best-kept line.

An attempt whose correctness was not re-checked has no ratio. Record it as
`"ratio": null` rather than reporting a number you cannot stand behind.

**Record every attempt as you go, in `reports/optimization_<stage>.json`:**

```json
{"stage": "...", "archetype": "mixed|vec_only|cube_only", "baseline_ratio": 2.31,
 "attempts": [{"n": 1, "hypothesis": "...", "changed": "...", "ratio": 1.94,
               "ci": [1.93, 1.95], "kept": true,
               "correct": true,          // FALSE if it failed validation
               "kind": "candidate",      // or "diagnostic" for a probe
               "why": "...",
               "kernel": "src/variants/kernel_<stage>_a01.cpp"}],
 "stop_reason": "budget_exhausted|hardware_limit",
 "gate": "...", "gate_value": "..."}
```

`correct` and `kind` are not bookkeeping -- they change what the graph asserts. A real
campaign produced an attempt that measured as **the fastest point on the whole chart**
and was numerically wrong (0/14 cases, 2.65M elements off by more than 1); a reader's
eye goes straight to the lowest point. Mark a failed-validation attempt `"correct":
false` and it is drawn as a red cross, excluded from the best-kept line, and banner-ed.
Mark a noop-floor or strided-vs-contiguous probe `"kind": "diagnostic"` so it is not
read as a kernel you could have shipped. **Speed for a wrong kernel is not a result.**

**Archive EVERY attempt's kernel** under `src/variants/kernel_<stage>_a<NN>.cpp` —
including the one you keep. Overwriting the main kernel in place with the winner and
archiving only the losers loses the winning kernel's identity the moment a later attempt
supersedes it; one run here did exactly that, and its best intermediate is now only
recoverable because it happened to be the last one.

**Then plot it with the plugin's script — do not hand-roll a chart:**

```bash
<py> ${CLAUDE_PLUGIN_ROOT}/scripts/plot_optimization.py reports/optimization_<stage>.json
```

It draws all 10 budget slots regardless of how many were used, so an early stop is
*visible* as shaded unused budget, marks kept vs reverted attempts distinctly, traces
best-kept-so-far, and prints a red PROCESS FAILURE banner on a `mixed` stage that ran
short. That last part is deliberate: a campaign that stopped early should not be able to
look complete.

**Required in the report (Phase 8):**
1. **The trajectory table** — one row per attempt: `#`, hypothesis, what changed, measured
   ratio (+95% CI), kept or reverted, and *why*.
2. **The trajectory graph** from the script above, embedded.
3. **The stop reason**, explicitly: budget exhausted, or which hardware-limit gate fired
   with its number.
4. If fewer than 10 attempts were made on a single-engine stage, the gate evidence.
   If fewer than 10 on a `mixed` stage, that is a **process failure** — say so plainly
   in the report rather than presenting the result as complete.

Attempts must be measured under the same protocol throughout the campaign (see §2), and
correctness must be re-verified on the kept kernel — a faster wrong kernel scores zero.

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
- **Do NOT declare a "bandwidth-bound / hardware / not-achievable" floor without PROVING it
  against a tuned reference.** This is the most common false stop. Two hard gates before you
  write "bandwidth-bound" or "hardware limit": (a) compute achieved GB/s vs HBM peak AND
  achieved TFLOP/s vs compute peak — if you are far from BOTH (e.g. ~24% of HBM and ~21% of
  compute), you are neither bound, you are just inefficient (bulk-synchronous barriers,
  single-buffering, low occupancy); (b) if a vendor/reference does the SAME workload on the
  SAME silicon faster, the wall is your kernel, not the chip. In practice "impossible on this
  arch" (e.g. "a correct single-MIX Cube->Vec hand-off is A5-only", "attention is
  bandwidth-bound on a2a3") was disproven repeatedly by a working reference — each was a
  missing technique (a FIFO-pipelined hand-off, deeper run-ahead), not silicon. Distrust your
  own hardware-wall conclusion until a reference confirms the wall.
  **One measured exception to (b): the wall can be below everything you control.** A PTO
  `TLOAD` extracts ~920 GB/s from an out-of-L2 stream where a vendor fused operator reaches
  ~1493 GB/s on the same bytes. Every parameter a generated kernel can vary (conversion,
  contiguity, burst length, ring depth, descriptor size, `block_dim`, address partition,
  both engine classes at once) is measurably **flat**, and so is the raw CCE DMA intrinsic
  called directly with `TLOAD` bypassed (`PLAT-§ReadCeiling`). So (b) still holds — a vendor
  being faster does mean *someone* can go faster — but it does **not** follow that a
  technique exists at the level you are writing. Claiming this exception requires the sweep,
  not an assertion: if you have not swept those knobs and shown them flat, (b) applies and
  the wall is your kernel.
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
