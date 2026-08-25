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

### SHIP A block_dim SCHEDULE, NOT A block_dim CONSTANT

**`block_dim` is a RUNTIME ARGUMENT, not a property of the compiled kernel.** Tuning it once at the
contract's production point and shipping that constant is the single largest avoidable defect this
campaign found in the generated artifact, and it is invisible at the shape you tuned on.

Measured cost: **0.10-0.17 us of device time per launched block.** At `block_dim=24` that is ~2.4 us
of pure launch overhead -- 1.1% of one kernel's 212 us production runtime, and approximately the
ENTIRE runtime at the smallest shape in its own contract sweep. Re-tuning the single integer at the
small shape, same `.so`, **output bitwise-identical** (33 configs gated, max relative difference
exactly 0):

| case | smallest shape | shipped `block_dim` | best `block_dim` | effect |
|---|---|---|---|---|
| `gelu` | N=40,000 | 24 | **8** | 1.49x slower -> 1.04x FASTER |
| `reshape_and_cache` | T=14 | 24 | **2** | 1.43x slower -> 1.77x FASTER |
| `rope` | S=128 | 48 | **10** | 1.09x slower -> 1.43x FASTER |
| `rotary_mul` | S=128 | 24 | **4** | 1.31x -> **2.53x** FASTER |
| `dynamic_quant` | N=64 | 24 | **1** | 1.16x -> **1.66x** FASTER |

**No single constant works.** The optimum moves monotonically with problem size (measured for
`gelu`: 4 -> 8 -> 20 -> 24 -> 24 -> 24 as N grows), and the best compromise value still costs ~1.4x
somewhere in the contract's own sweep. A useful model for the large-shape side is
`t(bd) = t(bd_max) * (bd_max/bd) * ceil(bd/bd_max)`, which predicted the measured curve to 3.1%.

**Required:** emit a **per-shape `block_dim` selection** in the host wrapper, not a constant.
Cheapest correct form is a host-side autotune over one integer at first call per shape, cached, and
**gated on bitwise-identical output** against the shipped configuration -- the gate is what makes it
free of correctness risk, since `block_dim` must not change results. A static table keyed on the
contract's sweep points, with interpolation between them, is an acceptable substitute where a
first-call probe is unacceptable.

**Why this was missed, and the general rule:** the pipeline tuned `block_dim` at the same shape it
reported, so the configuration was optimal exactly where it was scored and nowhere else. **Never
tune a free runtime parameter only at the point you report.** Sweep it across the contract's whole
benchmark range and ship a schedule; if you ship a constant, state the range over which it was
validated.

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

## 3.5 MANDATORY attempt budget: 25 attempts, and they go in the report

Every generated kernel gets an optimization campaign. It is not optional, and it is not
finished when the kernel merely validates.

**The budget is 25 measured attempts.** An "attempt" is a *change with a paired
re-measurement* — a hypothesis, a build, a number. Reverted regressions COUNT, and are
often the most informative entries; do not quietly drop them.

> **Raised from 15 to 25 in v0.94.0, on measured evidence.** In the v0.93 campaign every
> stage that ran to exhaustion overshot the nominal 15 anyway — 21, 23, 28, 29 and 35
> candidates — and stages that stopped *below* 15 were still far from any floor
> (one stopped at 10/15 while sitting **6.9x above its own streaming roofline**, and that
> stage was the campaign's only regression against the previous generation). 15 was
> neither a real ceiling nor a sufficient floor.

**THE BUDGET IS A HARD CAP IN BOTH DIRECTIONS.**

* **You may not exceed it.** At 25 measured attempts you STOP and report
  `budget_exhausted`. Diagnostics (probes that change no shipped code — ablations, noop
  launches, roofline measurements) do NOT count against it and MUST be reported under a
  separate `diagnostics` count. If you find yourself at attempt 26, you have been
  miscounting diagnostics as attempts or attempts as diagnostics; say which.
* **You may not stop below it** except through the early-stop gate below. Running out of
  ideas is NOT a licence to stop. If you cannot think of another hypothesis, that is a
  reportable state — see "when you run out of hypotheses" below — but it is
  `budget_exhausted` with the unspent count stated, never a new stop reason.

| stage archetype | attempts | early stop allowed? |
|---|---|---|
| `mixed` (Cube + Vec, cross-core, composed/fused) | **25, always** | **No.** Run all 25 even when it is expensive. |
| `vec_only` | 25 | Yes — see the gate below |
| `cube_only` | 25 | Yes — see the gate below |

**When you run out of hypotheses before the budget.** This is the commonest way a
campaign quietly under-runs, and the escape below is DELIBERATELY expensive to reach --
the first version of this rule offered it unconditionally and stages promptly stopped at
**6 of 25**, worse utilisation than the 15-budget it replaced.

"I have no more ideas" is a statement about the search, not about the kernel, and it is
**not** sufficient on its own. Before you may report `budget_exhausted` below the cap you
must have DONE and RECORDED all four of these:

1. **Re-measured the binding resource** per `3.6`, in the CURRENT configuration -- not the
   one you measured at attempt 1. The bottleneck moves as you optimize; a stale diagnosis
   is the usual reason the idea list looks empty.
2. **Walked the bottleneck-to-lever tree in `4` for the resource you just named**, and
   recorded each lever in that class as attempted, measured-and-reverted, or
   **inapplicable with a one-line reason**. "Not tried" is not one of the three.
3. **Retried at least one lever you rejected on REASONING rather than measurement.**
   Reasoning has been wrong here repeatedly -- a named headline lever measured 0.999x, an
   L2 alias sign-flipped with the protocol, and a "faster" attempt scored 1/38 on
   validation. If every rejection in your log is argued rather than measured, you have not
   run out of hypotheses; you have run out of willingness to measure.
4. **Named the structural hypothesis** that would close the remaining gap, with your floor,
   the structure-independent bound and the ratio between them.

If all four are in the record, report `budget_exhausted` with `attempts_spent`,
`attempts_unspent`, and one sentence on what more attempts would buy. **If any of the four
is missing, you have not earned the stop -- keep going.** A campaign that reports
`budget_exhausted` at low single-digit attempts with an empty lever table is a process
failure, and the report must say so rather than presenting it as a result.

Note the asymmetry, which is intended: stopping at the cap needs no justification;
stopping below it needs four pieces of evidence.

**Early-stop gate (single-engine stages only).** You may stop before the budget *only* if
you can show the kernel is at a **hardware limit**, with a measurement, not an argument:
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

**Check the barrier scope before anything else on a Vec-heavy stage.** `pipe_barrier(PIPE_ALL)`
drains every pipe and destroys the overlap you are trying to create. A census of the shipped
vendor kernels finds it in **0.3%** of files (19 of ~5900) against 1198 using scoped
`PipeBarrier<pipe>` — while **95% of our generated kernels use it** (105 of 111, 417
occurrences). `pipe_barrier` accepts any `pipe_t`, so the scoped form was always available.
See `references/vendor_idiom_census.md`.

This is an unmeasured hypothesis, and a delicate one: barrier *removal* has already failed
validation once (`dequant_swiglu_requant` attempt 1) and a per-item `PIPE_ALL` was silently
protecting output tiles against a WAR hazard (`deep_norm_backward` D6). The work is to
**replace each barrier with the correctly scoped flag class and re-validate**, never to
delete barriers and hope.

When the bandwidth gate fires on an out-of-L2 stream, **try the uncached address alias FIRST**
(`PLAT-§L2Bypass`): a streamed operand read once and never reused should be loaded through
`ptr + rtGetL2CacheOffset()`, which measured **1.67x** (915 -> 1527 GB/s) on an identical
kernel binary and is bit-exact. Wider bursts, deeper rings, more cores and an NZ ABI are
all measurably flat and are not worth an attempt. After that, the remaining lever is
**shrinking the footprint** so the hot set fits L2.

Why `mixed` gets no early stop: its cost is a *composition* — cross-core handshakes,
seam sync, and overlap between two engines that a single-engine roofline does not model.
An engine can sit at its roofline while the composition wastes most of the wall clock.

**And the inverse, which is just as expensive to learn late: a stage can be far off its own
roofline and still be FREE.** Before optimizing any stage inside a composition, **bisect the
composition** — compile the chain to stop at the seam (a `-DSTOP_AFTER_<stage>` flag is
worth carrying in every generated chain kernel for exactly this) and time it. The
difference bounds everything downstream of that seam, and it costs two commands.

On `grouped_matmul_swiglu_quant` that bisection returned 568.4 us against a full chain of
572.5 us: **the entire second stage — its GM intermediate read-back, its whole Vec chain and
its stores — was worth 4.1 us of 572.** That single measurement retired both a stage-2
optimization campaign and a proposed compute-fusion, each of which would otherwise have
been days of work for <1%. Bound the prize before building. On `grouped_matmul_swiglu_quant` the stage-2 Vec chain ran at 205 GB/s against a
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
 "stop_reason": "budget_exhausted|structure_limit|hardware_limit",
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

It draws all 15 budget slots regardless of how many were used, so an early stop is
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
4. If fewer than 25 attempts were made on a single-engine stage, the gate evidence.
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

## A large win against a vendor op is a claim about the VENDOR, until proven otherwise

When a generated kernel beats a vendor operator by more than ~3x, the default explanation is
**not** that the kernel is exceptional. It is that the vendor operator is off its own
capability at that shape. Establish which before reporting:

1. **Convert both arms to GB/s** (or FLOP/s) on the essential bytes.
2. **Measure the vendor's OWN ceiling** by sweeping it to a shape it is built for, and
   report our rate as a percentage of that.
3. If our rate is unremarkable (say 50-70% of the vendor's own best) while the vendor's rate
   at the tested shape is far below it, **the finding is "the vendor does not cover this
   shape"** -- report it that way, with the GB/s pair, not as a bare speedup.

Worked case: `nsa_select_attention` measured **18.6x**. Ours was 534 GB/s -- 64% of the
830 GB/s the vendor itself reaches at BS1=64 -- while the vendor at BS1=1 ran at 28 GB/s,
30x below its own capability, from a fixed ~2.7 ms floor flat across BS1=1..16. The same
operator **overtakes us at BS1=16**. A bare "18.6x" would have been indefensible; the GB/s
pair is both defensible and more informative.


## A measured inefficiency need not be RECOVERABLE -- price the fix, not just the loss

`group_norm_swish` measured a real, specific inefficiency at its production shape: 64 groups
quantising onto 48 lanes cost **1.41x** (0.716 vs 0.507 us/group at N=6). The number was
solid and the diagnosis was correct.

**Four attempts to recover it all regressed.** A `-DSTOP_AFTER_<phase>` bisect then priced
the actual barrier at only **~1.9 us** -- so the barrier was never the problem. The loss came
from *rescheduling*: the fused per-group loop already runs both passes over the same group
back-to-back, and every split that "fixed" the lane quantisation destroyed that locality for
more than it recovered.

**The lesson: locating a loss does not mean you can collect it.** A quantisation, imbalance
or occupancy gap you can *measure* is an upper bound on what a fix could win, and the fix's
own cost can exceed it. Before spending attempts:

1. **Price the fix separately from the loss.** Bisect with `-DSTOP_AFTER_<stage>` to bound
   what the restructuring itself costs. If the mechanism you would remove is 1.9 us and the
   loss is 40%, the loss is not where you think it is.
2. **Ask what the current structure is buying.** A shape that looks wasteful in one dimension
   is often paying for locality, reuse or pipelining in another. Splitting to fix the visible
   axis silently gives that up.
3. **Cap the attempts.** Two consecutive regressions on the same hypothesis means the model
   is wrong, not that the implementation needs another pass. Record the inefficiency as
   *measured but unrecovered* and move on -- that is a legitimate, reportable outcome, and
   far more useful than four regressions and no explanation.

Report such a finding explicitly: "X costs 1.41x, N attempts to recover it regressed, the
mechanism is Y" is a stronger result than silence about a known gap.


## BEFORE SPENDING ANY ATTEMPTS: check whether you are optimizing the wrong thing

The optimizer tunes a kernel. It cannot fix a **decomposition**. Two cases in the campaign spent
their full 15-attempt budgets, reached **99.5% of their measured bandwidth ceilings**, and still
lost to the vendor by 1.09x-3.35x -- because the gap was the dataflow, not the code.

**Run this check first, from the stage plan's seam analysis:**

1. **Compute the traffic amplification** -- our total GM bytes divided by the fused lower bound
   (the bytes a single fused kernel would have to move: inputs + outputs, nothing else).
2. **Compute how it grows across the sweep.** If amplification is roughly constant, tuning can
   win. If it **grows with the sweep dimension**, the intermediate is asymptotically larger than
   the inputs and **no tuning will close it** -- measured 3.43x -> 17.00x across S=128->1024 on
   `flash_attention_grad`.
3. **Compute the local scaling exponent per sweep step**: `log2(t[i]/t[i-1])` over
   `log2(S[i]/S[i-1])`. If it **exceeds the traffic model's order**, a working set has crossed a
   capacity boundary -- look for L2 (192 MB on A2) before optimizing anything.

**If the amplification grows: stop and report it.** Say the kernel is at N% of its achievable
ceiling and that the remaining gap is `<amplification>x` of traffic inherent to the
decomposition. That is a complete, useful result. Burning the budget to confirm it is not.

**Report the ceiling you are at AND the ceiling you cannot reach.** "730 of a measured 734 GB/s
ceiling (99.5%), moving 17x the vendor's bytes" tells a reader exactly where the work is. "1.09x
slower" does not.

### Corollary: an ablation that removes all the work must change the time

If a variant with the arithmetic removed measures the same as the full kernel, you have not
found a hardware limit -- **you have found a harness fault**. One run concluded "hardware limit"
from nine variants agreeing to 0.5%, one of which did **no work at all**; its driver was
draining the stream per window and inflating everything 2-5x. A real 1.127x was still available.


## DATA LAYOUT IS AN OPTIMIZATION AXIS -- and the current search misses it entirely

Measured failure. Two independent runs of the same int8 grouped-matmul case, at a plugin version
~50 releases newer than the best-known result, both landed **~2.6x short of it** -- reliably, with
only ~12% spread between them. Auditing all **31 optimization attempts across the two runs: ZERO
changed the data layout of the streamed operand.** Every attempt tuned *how to traverse a fixed
layout* -- tile width (`kNT` 128/192/256/512/1024), macro-blocking (MB=2, MB=4), row-tile count,
block_dim, core distribution, barrier scoping, L2 alias, double-buffering.

The earlier, faster run had shipped a **pre-packed weight** (`[E, N/NT, K, NT]`, contiguous,
1128 GB/s). The newer runs shipped the **native strided ND** weight (`[E, K, N]`, 424-735 GB/s)
and then optimized *within* it -- correctly, thoroughly, and to a genuine hardware gate. They
were solving the wrong problem well.

**The exploration is insufficient, not the rules.** Both newer runs validated bit-exactly,
seam-analysed correctly, and reported honestly. They simply never asked whether the operand
should be laid out differently.

### The rule

> **If an operand is READ-ONLY and PERSISTS ACROSS INVOCATIONS, its memory layout is a FREE
> VARIABLE. A one-time host-side repack amortizes to zero, so the layout belongs in the search
> space alongside the tiling.**

Model weights are the canonical case: loaded once, used for millions of inferences. This is
exactly why the vendor ships weights in a fractal/NZ format rather than plain ND -- and why a
vendor rate measured on that format is **not** a ceiling a plain-ND kernel can reach (see
`PLAT-§ReadCeiling`: quote the ~920 GB/s ND `TLOAD` ceiling for an ND kernel, not the vendor's
1236 GB/s NZ rate).

### Where to spend the attempt

**Before tuning the traversal, ask this once and record the answer:**

1. **Which operand dominates the traffic?** (Here: the weight stream at 470 MB, vs an 8.4 MB
   intermediate -- 98% of the bytes.)
2. **Is it read-only and reused across calls?** If yes -> **layout is in scope.**
3. **What does its access pattern actually look like under the chosen tiling?** Strided reads of
   `kNT` bytes at stride `N` are the tell: a "widen `kNT` for longer bursts" attempt is a
   *workaround for a layout problem*. If you find yourself widening a tile purely to lengthen a
   burst, the burst is short because the layout is wrong.
4. **Price the repack honestly.** State it as a one-time host cost and say what it amortizes
   over. If the operand is *not* reused across calls, the repack must be paid per call and
   usually does not pay -- say so and move on.

**Budget at least one attempt to a layout variant whenever (1) and (2) hold.** An attempt that
tests a packed/fractal layout and loses is a real result; never testing it is a blind spot, and
in the measured case that blind spot cost 2.6x.

### Diagnostic that would have caught it

Achieved bandwidth as a fraction of the ceiling *for the layout you chose*, reported next to the
ceiling *for the best available layout*. The newer runs reported "46-80% of the 920 GB/s ND
ceiling" and gated as load-bound -- correct, and it hid the fact that a different layout has a
higher ceiling. **Report both ceilings, and if the gap between them is large, that gap is your
next attempt.**


### A layout's DEPLOYABILITY is part of its score, not just its bandwidth

Two layouts that deliver the same bandwidth are **not** equally valuable. What a layout is
*parameterized by* decides whether the caller's prepared tensors survive your next tuning pass:

| layout parameterized by | stability | example |
|---|---|---|
| **hardware constants** | **stable forever** | a fractal/NZ layout keyed to the 16-element fractal and the 32 B burst granule |
| **contract dims** | **stable** while the contract holds | blocking by `E`, `K`, `N` |
| **a TUNING parameter** | **fragile** -- retuning invalidates every prepared tensor | `[E, N/nt, K, nt]` keyed to *your chosen tile width* `nt` |

This is exactly why the vendor's `FRACTAL_NZ` is a better *deliverable* than a tile-width-keyed
repack of identical cost: `npu_format_cast` is keyed to the hardware fractal, so it survives any
retuning the vendor does internally. A layout keyed to `nt` means **a caller who stored weights
must re-prepare them whenever the optimizer moves `nt`**.

**Rules, in preference order:**

1. **Prefer a layout parameterized by hardware constants or contract dims.** If a
   hardware-keyed layout reaches within a few percent of a tuning-keyed one, **ship the
   hardware-keyed one** -- the small bandwidth loss buys a stable ABI.
2. **If you ship a tuning-keyed layout, FREEZE the parameter into the contract.** Promote it
   from a free tuning knob to an ABI constant: record it in `tensor_layouts.coupled_to`, state
   that later attempts may not retune it without a version bump, and `static_assert` the kernel
   against it so a mismatch is a compile error rather than silent corruption.
3. **Never leave a tuning-keyed layout implicit.** A caller who cannot see what the layout is
   keyed to cannot know when their stored tensors went stale. That is a silent-wrong-answer
   generator, and it is worse than a slower kernel.

**Report it in the ladder.** When a layout attempt wins, state its parameterization alongside
its bandwidth -- "1.41x, keyed to `nt=256` (tuning-coupled, frozen as ABI)" is a materially
different result from "1.38x, keyed to the hardware fractal (stable)", and the second may be the
better ship.


### Refinement: a tuning-keyed layout is FREE when the tensor is an INTERNAL intermediate

The deployability rule penalises layouts keyed to a tuning parameter because retuning invalidates
tensors the **caller** prepared. That reasoning does not apply when the caller never sees the
tensor.

**An internal intermediate -- allocated, written and consumed entirely inside your own kernel or
chain -- has ZERO deployability cost, whatever it is keyed to.** Retuning simply regenerates it.
So the preference order applies to **caller-visible operands only**; for internal buffers, pick
the fastest layout and freeze it with a `static_assert` for correctness, not for ABI stability.

Measured example: a chain shipped its internal `[N/128, M, 128]` intermediate keyed to a tuning
knob (`NTB=128`), correctly recorded as `coupled_to` and `static_assert`ed -- while both
caller-visible weights stayed in the reference layout with `drop_in: true`. That is the right
outcome, and a rule that penalised the intermediate's coupling would have been wrong.


### WHEN the layout axis pays -- a measured discriminator

Layout is not always worth an attempt. Two cases from the same campaign, both matmul-family, both
optimized under the same rules, with opposite outcomes:

| | grouped int8 (layout worth **2.2x**) | fp16 FFN (layout worth **nothing**) |
|---|---|---|
| weight share of chain traffic | **470 MB = 98%** | **3.28 MB = 1.8%** |
| working set vs L2 (192 MB) | **470 MB -- EXCEEDS** | **97.6 MB -- fits** |
| weight reuse | ~21 tokens/expert | **M = 8192** |
| contraction axis | the slow (strided) axis | already the fast axis |

**The layout axis pays when the operand's stream (a) DOMINATES the traffic and (b) EXCEEDS L2.**
If the working set is L2-resident, repacking raises no ceiling -- the operand is already being
served from cache, and a better layout only reorders hits. If the operand is a small fraction of
traffic, even a large relative improvement moves nothing end-to-end.

**Ruled out as explanations by the same pair:** dtype and operand count. The FFN's two weights
behaved **differently from each other** (one packing was null, the other an 18% regression), which
neither dtype nor count can account for -- so do not reach for those.

**Confirmed a second time WITHIN a single kernel, which is the stronger form.** A later fp16
grouped matmul measured its own layout win at two operating points with everything else held
fixed -- same binary, same dtype, same operand count, same generator version:

| weight share of essential traffic | layout win |
|---|---|
| 3.6% | **1.098x** |
| 87.3% | **1.404x** |

Same monotone relationship, with every cross-case confound eliminated by construction. Prefer
this design when testing any future axis: **vary the suspected variable inside one kernel** rather
than comparing two kernels that differ in a dozen ways. The cross-case pair told us *that* the
axis was conditional; only the within-case sweep told us *what* the condition is.

> **CORRECTION -- the vendor-facing half of this claim was a measurement artifact.** This rule
> previously ended: "the axis moved it from 1.131x slower to 1.614x faster than the vendor." A
> corrected re-measurement (K>=16 calls per event window) **flips that**: the same kernel reads
> 1.04x faster at K=16 and 1.22x SLOWER at K=256, never converging, so *no* vendor-facing claim is
> established for this case. The old 1.614x is reproduced almost exactly by K=1 (1.621x), which
> identifies it as the enqueue artifact.
>
> **What survives, and why it survives, is the part this rule actually needs.** The layout win is
> measured **ours-vs-ours** -- packed-NZ against our own ND, same kernel, same launch path -- at
> **1.08x on the primary shape and 1.22x on MoE**. An ours-vs-ours comparison is *structurally
> immune* to the enqueue bias, because both arms pay an identical host cost that cancels in the
> ratio. So the optimizer's internal decisions were sound even while the vendor comparison was not.
>
> **General lesson for every rule in this file:** an optimization axis should be justified by an
> ours-vs-ours A/B, never by a vendor delta. Vendor comparisons belong in the final report, not in
> the search loop -- they carry a bias that an internal A/B does not.

**Practical gate, before spending an attempt on layout:** compute the candidate operand's share of
total traffic and the live working set against L2. Under a few percent of traffic, or comfortably
inside L2, spend the attempt elsewhere and record why. This is the same shape as the seam
analysis: a cheap up-front calculation that tells you whether an axis can possibly pay.

---

## 3.6 NAME THE BINDING RESOURCE, WITH ITS NUMBER, BEFORE CHOOSING WHAT TO ATTACK

The single highest-value first attempt available to you is a **one-resource / noop-floor
probe**: rebuild the same source with the arithmetic deleted (traffic and sync identical),
and again with the loads deleted. It costs one or two builds and it has repeatedly
overturned the obvious story.

Measured consequences of skipping it:

* A campaign stopped at **"85% of HBM peak"**. The probe showed compute-only 26.30 us,
  DMA-only 19.64 us, full kernel 27.29 us -- the DMA was ~100% hidden and the kernel was
  **vector-bound, 39% above its real ceiling.** The 85% was a roofline for the NON-BINDING
  resource and licensed a stop that had not been earned.
* Another kernel's headroom was entirely in one cache state: an arithmetic-deleted probe
  showed the production HBM-bound case was **already ON its floor** (81.04 vs 80.22 us), so
  every attempt was correctly aimed at the L2-resident state instead, and the HBM arm came
  out neutral by design rather than by failure.
* A third declined to build two plausible structural rewrites because the probe showed the
  memory path had **45% headroom** and neither change removed a vector op.

**A roofline percentage is a valid stop gate ONLY for the resource measured to be binding.**
Which resource binds must come from a probe -- never from the kernel's shape, its dtype, or
how it "looks". Report the binding resource, its probe value, and your ratio against THAT
ceiling.

### CRITICAL: an engine-nulled floor bounds YOUR STRUCTURE, not the problem

A noop-floor probe measures the floor of **the kernel you wrote**. It says nothing about
what a *different* decomposition could reach, and it will happily license a stop far from
the achievable time.

Measured, same algorithm, same contract, same hardware, two independent generations:

| | shipped | its own measured "floor" | stop reason it claimed |
|---|---|---|---|
| generation A | **74.92 us** | -- | budget |
| generation B | 122.29 us | 115.78 us (Vec-only ablation) | **"hardware limit", 105.6% of floor** |

Generation B stopped at a *correctly measured* 105.6% of its own Vec-only floor -- while a
different structure for the same problem ran **1.63x faster than that floor**. The probe was
not wrong; it was answering a narrower question than the one that mattered.

**So before claiming a hardware-limit stop, sanity-check the floor against something
structure-independent**: bytes that MUST move against the measured streaming ceiling, or the
essential FLOP count against the engine's issue rate. If your "floor" is far above that
bound, you are at the floor of a structure, not of the problem, and the remaining move is a
**redesign, not a schedule change**.

### THREE STOP REASONS. Report exactly one, and never upgrade a weaker one.

| stop reason | condition | what it licenses |
|---|---|---|
| `budget_exhausted` | attempts spent, gates still open | more attempts would help; SAY SO |
| `structure_limit` | at your own ablation floor, but that floor is far above the structure-independent bound | **a REDESIGN, not more attempts** |
| `hardware_limit` | at the structure-independent bound (within ~10%) | genuinely done |

`hardware_limit` is the ONLY one that means "done", and it requires the
structure-independent bound -- not your own ablation floor. An engine-nulled ablation can
only ever produce `structure_limit`.

**THESE THREE NAMES ARE CLOSED. Inventing a fourth is a process failure.** In the v0.93
campaign five stages across two cases emitted `budget_partially_spent` and
`no_further_hypotheses` -- both are `budget_exhausted` with an unspent count, and naming
them otherwise concealed that the stage stopped with headroom it never used. If your
situation does not fit one of the three, it is `budget_exhausted`; say what is unspent and
why you stopped. The stop reason MUST also be written to `pipeline_results.json` under
`optimization.stop_reason` (and per-stage under `stages[].optimization.stop_reason` when
stages are optimized separately) -- a stop reason that exists only in prose is not
reportable and did not happen.

**When `structure_limit` fires, the report MUST carry:** your floor, the
structure-independent bound, the ratio between them, and a named structural hypothesis for
what would close it (a different decomposition, tiling, residency plan or traversal count).
That hypothesis is the deliverable -- it is what lets a caller decide whether to spend a
regeneration.

**A schedule search cannot cross a structural floor.** Spending more attempts against
`structure_limit` is the single commonest way to burn budget for nothing. Escalate instead:
regenerate the stage with the structural hypothesis, then optimise the new structure. The
attempt budget is per-structure, and a caller may always ask for more rounds -- but more
rounds on the wrong structure buy nothing.

**A traffic ratio is a hypothesis, not a diagnosis.** "We move 1.5x the bytes the vendor
does" says nothing until a probe shows those bytes are not already hidden. On one case that
exact argument was refuted: deleting ALL of the load path saved **2.7%** and **0.8%** on the
two stages, because the redundant read was >97% overlapped.

## 3.7 EVERY ATTEMPT IS A PAIRED (PERFORMANCE, CORRECTNESS) OBSERVATION

Run the numerical gate on **every measured attempt**, not only on the final binary. The
optimiser's objective function actively rewards two specific bugs:

* removing a `pipe_barrier(PIPE_V)` between dependent vector arithmetic -- **3-4% faster,
  relative error up to 1.3e+20** (COOK-§6.26);
* replacing a two-pass reduction with a single-traversal one -- **6.6% faster, 31x over
  tolerance under a DC offset** (COOK-§22).

Both are **faster AND wrong**, and both are shape- or distribution-dependent: they pass at
some points of the sweep and fail at others. An attempt that improves time while failing
tolerance is a **REVERT plus a reportable finding**, never a candidate.

## 3.8 A PAIRED INTERLEAVED A/B IS INVALID WHEN THE ARMS SHARE A RESOURCE THE TREATMENT MODIFIES

Interleaving defeats drift and you should keep using it. It has its own failure mode, and it
produces a confidently wrong answer with every safeguard green.

Measured: an interleaved A/B priced an L2-bypass alias at **1.0000, CI [1.0000, 1.0003]**,
three shapes, valid null control. Measured **single-arm, one arm per process**, the same
alias is a **1.15x REGRESSION**. The alias suppresses cache *allocation* but not *lookup*, so
the alias-off arm was **populating the cache for the alias-on arm**. On a second case the
contaminated reading had already SHIPPED and been defended as load-bearing before a
single-arm re-measure inverted its sign.

**Before trusting a paired A/B, name the resource the two arms share and ask whether the
treatment changes it.** Cache residency, queue depth and DVFS all qualify. When they do,
measure single-arm in separate processes, or from a cold state where there is nothing to
leak. **If a paired and a single-arm reading disagree, the single-arm one wins.**

**Free contamination detector:** treatments that are independent must COMPOSE. If A alone
and B alone do not multiply to AB, the pairing is leaking. One case read
`alias_x` 0.916, `alias_y` 1.001, `alias_xy` **0.636** -- physically impossible, since
`alias_y` alone is a no-op. Single-arm gave 0.791 / 1.000 / 0.791. The non-composition was
visible for free, with no re-measurement.

## 3.9 CACHE-BYPASS ALIASES: measure LAST, in the FINAL configuration

Across eleven measured cases the alias trigger mispredicted on every one where it was
applied early. What survives is a discriminator and one number, not a recipe:

* **The alias wins iff the live working set EXCEEDS the last-level cache**, and loses when
  the set is resident. Measured crossover on schedule redundancy: **~1.9x**.
* A capacity sweep on one kernel: flat **1.000x** from 100 to 176 MiB, **1.089x** at 192 MiB,
  **2.55x** at 240 MiB. It is a **cliff at capacity, not a slope**.
* **Any bottleneck-moving change invalidates a prior alias measurement, in either
  direction.** The same operand at the same working set flipped 0.867x -> 1.026x purely from
  software-pipelining the loop. So "alias first, then tune" is exactly backwards.
* "Write-only operands should always bypass" is **falsified** (0.915x, replicated 3x).
* Per-operand aliases do not compose: two that each helped (1.10x, 1.07x) were a **2.03x
  regression** together.

**Rule:** ship the alias as a runtime knob defaulted OFF, measure it as your LAST attempt in
the final configuration, single-arm (§3.8), and record the cache state with every number. An
alias figure quoted without its cache state is not reproducible.

## 3.10 DECIDE KEEP/REVERT ON THE FULL CONTRACT SWEEP

An attempt measured **1.008x at the production point and 0.952-0.965x** where the kernel's
actual headroom was. A production-only view would have shipped it. Keep/revert is a decision
about the contract, not about one shape.

Corollary: effects under ~3% require interleaved replication (subject to §3.8) before they
are believed. One campaign retracted a 1.006-1.008x "win" with a within-process CI clear of
1.0 after replication put it at 1.000-1.004x against a 0.6-1.3% spread.
