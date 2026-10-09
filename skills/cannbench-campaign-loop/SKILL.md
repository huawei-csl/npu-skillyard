---
name: cannbench-campaign-loop
description: "Run one operator through the full cann-bench campaign loop: establish a control, pass the free pre-submission gates, submit, retrieve and decompose the result, decide whether a fix is worth anything, fix it, and record it. Encodes the scoring model, the credit economics, the evaluator's traps, the failure taxonomy, and the defect classes that have actually cost points. Triggers: submit a kernel to cannbench, a hidden evaluation failed, why did this operator score X, is this fix worth a credit, package an operator for the benchmark, analyse a job result, update the scoreboard."
---

# cann-bench campaign loop

One operator, end to end: **control -> gates -> submit -> retrieve -> decompose -> decide -> fix -> record.**

Follow the phases in order. Each exists because skipping it has cost real points or real credits;
where that is true the cost is named, so you can weigh it rather than take it on faith.

**The one rule that subsumes the rest: never spend a credit to learn something a free check would
have told you.** Most of this document is free checks.

---

## Phase 0 -- The scoring model

Get this right first, because every "is it worth it" decision downstream is arithmetic on it.

```
compile     = 20 * (N - compile_runtime_fail_cases) / N
function    = 30 * accuracy_passed_cases / N
performance = 50 * sum(HAP_i) / N            over the cases that RAN
total       = compile + function + performance

HAP_i = (T_base - T_hw) / ((T_cand - T_hw) + (T_base - T_hw))
```

`N` is 20 for the standard set and **80 for hidden**.

**The compile term scales only with `compile_runtime_error` failures, not with all failures.**
A `precision_mismatch` leaves compile at a full 20. This is the single most important asymmetry in
the model and getting it backwards has caused two wrong priority calls in this campaign:

- An operator failing 8 cases on **shape rejections** loses compile AND function AND the
  performance those cases would have contributed. `softmax` at 71/80 scored 70.70 and a full pass
  projects to ~79.4: **+9.3**.
- An operator failing 3 cases on **numerics alone** already has all 20 compile marks, so fixing
  them recovers far less. `apply_adam_w` at 77/80 recovers ~+6.8 despite failing fewer cases than
  it looks like it should.

**A leaderboard entry is your best FULLY-PASSING run, standard or hidden -- not your best score.**
A partial run posts *nothing*, however high it scored. This is why an 82.10 hidden result can sit
behind a 78.13 posted standard entry and change nothing.

### The projection that decides everything

Before spending a credit on a hidden run, or an agent on a fix, compute what a **full pass** would
actually score:

```python
comp = 20*(N-fails)/N ; func = 30*passed/N ; perf = score - comp - func
avg_hap = perf / (50/N) / passed            # HAP averaged over the cases that ran
full_pass = 20 + 30 + 50*avg_hap            # assumes the fixed cases match that average
```

## THE PASS-COUNT GATE: `full_pass` IS A CONDITIONAL, NOT A SCORE

**Read this before the arithmetic below, because the arithmetic is the trap.**

`full_pass` is what the operator scores **if and only if every case passes**. It is not a
prediction and it is not a partial credit. A run that fails even one case is a `partial_pass` and
**cannot displace a passing run** -- so a submission whose ceiling you have already calculated as 78/80
is a **guaranteed zero**, and the `+8` attached to it is imaginary.

> **REFINEMENT, measured 2026-09-30.** "A partial posts nothing" holds **when you already hold a
> passing run**: an 82.10 hidden partial did not displace a 78.13 posted entry. But for an operator
> with **no** passing run, the failed job *does* occupy the slot -- one operator's `partial_pass` at
> **5.68** became its `best_job_id` with `correctness_passed: false`. So on an empty slot a partial is
> **actively harmful**, not merely worthless: it posts a *low* score where you had none, and only a
> passing run clears it. The gate is unchanged (a partial never gains anything) but the downside is
> real, which is one more reason the pass count must be established before the credit is spent.

**So the rule is: send only if the hidden can PASS.** Not "is better than the entry", not "+8 over
the posted score" -- a projected **80/80**.

**Before proposing any submission, state the projected PASS COUNT and name every still-open case.**
If any case is open, the answer is *finish it first*. The work queue is "close the last case", never
"bank the improvement".

This was got wrong at scale on 2026-09-30, and the failure is worth naming because the rule was
already written in this file. Four operators were repaired and each was proposed for a credit on its
delta, while every one of their ceilings had already been computed here:

| operator | delta pitched | actual ceiling | why |
|---|---:|---|---|
| `mla_prolog` | ~+8 | **78/80** | 2 cases OOM the benchmark's **own** fp64 golden at 16 GiB |
| `weight_quant_batch_matmul` | +8…+10 | **78/80** | Group B (2 cancellation cases) still open |
| `gqa` | +6.87 | **78/80** | case 68 blocked by the `aclrtMemcpy` anti-cheat |
| `roi_align` | +0.83 | **19/20 standard** | case 12 is the separable identity's own floor |

**A permanently-unreachable case makes the operator's hidden permanently unspendable.** When the
benchmark's own golden cannot run a shape -- it OOMs, or the reference cannot express it -- no kernel
work will ever produce 80/80. **Record that verdict once and stop revisiting the operator**, rather
than re-deriving a delta that can never be collected. `mla_prolog` is the worked example: its kernel
computes those two shapes correctly and the *reference* is what fails.

**Fixing is not gated; only submitting is.** Repair and package freely -- a silent wrong answer is
worth closing whatever the board pays, and this campaign's most valuable finding of that day
(`depthwise_conv_2d` was 7/20, not the recorded 19/20) came out of work with no board value at all.
What needs the pass-count check is the moment a credit is spent.

---

### There are TWO boards, and the hidden one is the campaign's objective

**This section exists because an earlier version of this file taught the wrong board, and the error
survived three separate re-readings.** It priced every fix against the **per-operator** entry and
printed `fix earns zero` for two operators, and that verdict was then repeated out loud at the user
three times for `unique`, `resize_bilinear` and `nms` -- each time against a user who had already
said, in their own words, that the hidden set is what we are losing on.

| board | what it posts | does a hidden PARTIAL count? |
|---|---|---|
| **per-operator** | your best **fully-passing** run, standard or hidden | **No.** A partial posts nothing. |
| **solution** | per-operator columns incl. a separate **hidden** column | **Yes.** The partial's score sits in that column. |

**The consequence, and it inverts the old rule:** a hidden score increase pays on the solution
hidden column **even when the operator entry cannot move, and even when the run stays a partial**.
`full_pass < posted operator entry` therefore does **not** mean "earns zero" -- it means *"earns
nothing on the operator board"*, which is a statement about one of the two boards.

Worked example, 2026-10-08, `nms`. Operator entry **77.3861 and already #1**; hidden 62.0414 at
78/80 with a full-pass projection of **63.38**. The old rule prints `fix earns zero`. The truth is
**+1.33 on the solution hidden column** for the two cases, and the hidden *performance* term --
`case_score_mean` 0.2675 against 0.5477 on the same submission's standard run -- is worth up to
**~+18 more** on that same column. The operator board being unmovable concealed the single largest
prize on the operator.

**So price every fix against the column it actually lands in:**

```
operator-board gain = (full_pass >= 80/80 reachable) ? max(0, full_pass - posted_operator_entry) : 0
solution-hidden gain = new_hidden_score - current_hidden_score      # partials included
```

and report **both**, never just the first. If the user has named the hidden set as the objective,
the second number is the one that answers them, and the first is a footnote. **Their stated
objective is not yours to re-rank** -- if they say the hidden column is where the campaign is
losing, do not answer with an operator-board argument for dropping the work.

One carry-over from the old rule still holds, because it is about a different thing: on an **empty**
operator slot a partial is actively harmful (it posts a low score with `correctness_passed: false`
where you had none). That is an argument about *which* run to send, not about whether hidden work
is worth doing.

And when comparing against a solution entry, compare against **the best entry under our own
submission tag** -- teammates post under the same tag, so another member's entry is the bar, not
a rival's.

State both projections explicitly before recommending any fix. "It fails 6 cases" is not a reason;
"fixing 6 cases takes the pass count to 80/80, moves the operator entry from X to Y, and moves the
solution hidden column from P to Q" is -- **all three**, pass count first.

**Two calibrations, both measured -- do not substitute intuition for either.**

**1. Hidden HAP is NOT systematically lower. That claim was in this file and it is false.** It was
tested by pairing each partial hidden run with the standard run on the *same submission* (hidden case
ids are 21..100, disjoint from the visible 1..20), over 16 operators: **median hidden-visible HAP
+0.020, mean +0.008, negative in only 5 of 16.** So do not discount a hidden projection by default.
What *is* true is that where it bites, it bites hard and produces the entire negative tail --
`sparse_flash_attention` -0.329, `nms` -0.280, `resize_bilinear` -0.178, `gmsq` -0.065. The mechanism
is **not harsher scoring but shapes the schedule was never tuned for**: `nms`'s hidden `avg_speedup`
is 0.394 (2.5x slower than baseline) and `sfa`'s is 0.091 (**11x slower**). So assume parity, then
look specifically for a schedule that does not generalise -- a fitted tile ladder, a bucket table, a
core count, anything chosen against the visible shapes.

**2. `|delta| < ~1` is indistinguishable from zero.** The model was calibrated against the six
operators whose projections can be checked against an actual later hidden full pass: errors
`strided_slice` +2.62, `softmax` +0.81, `exp` +0.07, `rms_norm` +0.03, `add_rms_norm_dynamic_quant`
-0.03, `adaptive_avg_pool_3d` -4.87, **median absolute error 0.44**. It **overshoots when the failing
cases are the largest shapes** and **undershoots when the fix also changes the schedule**. Treat a
sub-1-point delta as a coin flip, not a gain.

**And a partial that beats your posted entry is not by itself evidence the operator is actionable.**
`moe_finalize_routing` scored 80.72 on hidden against a 77.99 entry -- a +9.28 projection, the second
largest on a 16-operator table -- and all six of its failures are prefixed `Golden执行失败` at
`op_runner.py:304`, the *golden* path, not `AI算子执行失败` at `:395`, which is ours. **Our kernel is
never invoked.** Check which side of the harness threw before pricing anything.

**CORRECTED 2026-10-06: a `Golden执行失败` DOES bill the compile term.** `evaluator.py:332` sets
`failure_type = FAILURE_TYPE_COMPILE_RUNTIME_ERROR` for it, so such a case costs compile **and**
function **and** performance exactly as one of ours would -- it is simply **unwinnable**, because the
golden runs **first** (`evaluator.py:318`, returning at `:321`) and our kernel is never reached. So
price it as a permanent `cf`, not as a free pass. Worked example: a hidden `projSize >= hiddenSize`
case can never be won, because `torch.nn.LSTM` itself **raises** at `proj_size >= hidden_size`
(verified: `P=31,H=32` constructs, `P=32,H=32` raises) -- and our own rejection code for it is
**unreachable**.

### HAP **SATURATES**. Price a performance lever in POINTS, never in microseconds.

A gap-to-leader is the **prize**, never the difficulty, and the two differ by an **order of
magnitude**. HAP's denominator carries **our own** excess over the roofline:

```
HAP_i = (T_base - T_hw) / ((T_cand - T_hw) + (T_base - T_hw))
```

Once `T_cand - T_hw >> T_base - T_hw` the ratio is already near zero, so **shrinking `T_cand` barely
moves it**. Measured on `weight_quant_batch_matmul` case 15: `T_base - T_hw = 95.8 us` against
`T_cand - T_hw = 588 us`, so a real, cleanly-separated **1.526x** speedup on the operator's largest
case was worth **+0.16 of 100**. Its +8.75 gap needed roughly **2x on every case** -- per-case
requirement 1.47x to 6.4x, **median 2.1x**.

**So run this free gate BEFORE assigning an agent to a performance gap:**

1. Pull per-case `T_base`, `T_cand`, `T_hw` from the evaluator JSON.
2. **Invert HAP for the target score** to get the **required per-case speedup**; report its median
   and its max.
3. If the median required speedup is above roughly **1.5x**, the gap is a **structure** problem, not
   a schedule problem. Say `stop_reason: structure_limit` **with that arithmetic** instead of
   spending a device-day on a tiling sweep.

**Use the inversion to TARGET, not only to stop.** The required-speedup table is per case, so it
also says *where* a gain is cheap. On `adaptive_avg_pool_3d` the gate fired on the original +10.76
target (median 1.59x, and cases 5 and 17 **mathematically unable to contribute** -- their HAP was
already 0.942/0.915, so the needed increment exceeded 1.0), **but the cases whose required speedup
was small were exactly the ones already classified as overhead-bound**, and spending the attempts
there returned **+6.25** (78.62 -> 84.87, 20/20 on every rep, ours-vs-ours geomean 1.63x). So read
the whole column before stopping: a high median can hide a cheap subset.

**And prefer a structure-independent FLOOR to a required-speedup ratio when you can measure one.**
The same campaign measured `~81 ns per MTE2 descriptor` fixed plus `~26 GB/s per core` above ~2 KB,
summed the per-case floor, and got a **score ceiling of 89.97**: we sit at **87.1%** of it and the
leader at **97.0%**. That reframes the gap as "land within ~3% of a measured descriptor-rate floor on
nearly every case", which is a far better decision input than "median 1.39x". Report
`stop_reason: structure_limit` only when you are **at** such a bound; if you are at 87% of it, the
honest stop reason is `budget_exhausted`.

The same campaign shows the bound is computable before a kernel is written. Both available levers at
their mathematical limit on **measured pipe times** -- traffic-optimal tiling **+0.86**, plus
**perfect zero-cost** AIC/AIV overlap **+6.57** -- reached 74.52 against the **76.49** needed, short
by **1.97**. That is a complete answer, and it cost no device time to derive.

**Two measurement corollaries from the same run.** Code-layout noise is **+/-10% on an UNCHANGED
kernel and code path** (two cases moved 0.94x and 0.90x with no change reaching them), so no
per-case claim below ~10% on an unchanged path is reliable -- use the aggregate. And **`npu-smi`'s
process table reads EMPTY under concurrency**: test for the literal `No process in device` sentinel,
never a process-line count, or a contended card reads as free.

### `t_hw` IS A CLOSED-FORM FORMULA. RECOVER IT AND THE HAP CEILING FALLS OUT FOR FREE.

`t_hw` is not a measurement. On a memory-bound elementwise operator it is
**`touches * N * itemsize / assumed_bandwidth`**, with both constants fixed per platform -- and
fitting it takes one pass over `tasks/metadata/<platform>.json` and `cases.csv`. Measured on
`apply_adam_w`, all 20 cases, two platform files:

| platform file | recovered formula | ratio to the fit | stdev |
|---|---|---:|---:|
| `910b2.json` | `4 * N * itemsize / 1920 GB/s` | **1.0000** | 0.00038 |
| `950pr.json` | `4 * N * itemsize / 1600 GB/s` | **1.0000** | 0.00024 |

Four significant figures on 20 of 20 cases is not a coincidence, it is the generator.

**Why this is worth doing first: the touch count can be WRONG, and then `HAP = 1.0` is
unreachable by construction.** `apply_adam_w` reads var/grad/m/v and writes one output --
`proto.yaml` declares a single `Tensor y` and `golden.py` returns one tensor without mutating the
moments -- so physics needs **5** touches while `t_hw` charges **4**. The floor is therefore

```
T_min / t_hw = (touches_real / BW_achievable) / (touches_charged / BW_assumed)
```

which at the **measured** 1730 GB/s (fp32) gives **1.387**, not the 1.25 you get by assuming peak.
Substituting into HAP with `r = T_base / t_hw`:

```
HAP_max = (r - 1) / (r - 1 + 0.387)
```

On the large cases `r ~ 2.3`, so **`HAP_max ~ 0.77`** -- and the operator measures **0.719-0.771**.
It is **at** its ceiling on exactly the cases that carry the score. On the small cases `r ~ 8` gives
0.95, which is why a hidden set skewed small shows a *higher* `case_score_mean` (0.8469) without
any of it being winnable.

**So before briefing any performance work on a bandwidth-bound operator:** recover the formula,
count the real touches against `proto.yaml` plus `golden.py`, divide by the **measured** achievable
bandwidth rather than the datasheet, and compare `HAP_max` to the live `case_score_mean`. If they
are equal you are done, and it cost no device time. This supersedes nothing in
`roofline-baseline-can-exceed-the-hardware` -- it gives that observation an exact formula and a
per-case ceiling.

### AN ABLATION OF CODE BEHIND A DISABLED GUARD IS A NULL CHANGE. PROVE LIVENESS FIRST.

Deleting a barrier is a rigorous upper bound on any correct replacement -- **but only if the
barrier executes.** On `apply_adam_w` a census found 4 `pipe_barrier(PIPE_ALL)`, an agent reported
2 of them as compiled out, and the truth was that **all four sit inside `#if PTO_NS == 1` while
`CMakeLists.txt` passes `-DPTO_NS=2`**. The shipped binary executes **zero**. The "delete all four"
arm therefore measured **1.002x** -- which is the instrument's noise floor, not a bound on barrier
cost, because the two builds were the same program.

**The arm looked like a clean negative and had no power at all.** Same family as a lever that is
only reserving memory and a local A/B over a branch no visible case reaches: a null result is only
informative once the treatment is known to reach the executing code.

**The check, before pricing any deletion ablation:** resolve the enclosing `#if` chain for every
site, grep the build files for the macro's actual value, and -- decisive -- confirm the two builds'
**device `.text` inside `.aicore_binary`** differ. Identical device code means you measured noise. A
source-level grep is a census of *text*, and text is not a count of executed instructions.

**Do NOT hash `.aicore_binary` itself** -- its `.strtab` carries the source filename, so a pure
**rename** moved the hash by 8 bytes and would be reported as a live code change. Hash the device
`.text` within it, after confirming the compiler is deterministic for your source.

**And do NOT fall back to the host `.text` for a host-side knob.** CORRECTED 2026-10-09 by two
independent agents: the host section is **not deterministic** on this toolchain -- one measured
**3 distinct hashes over 3 builds of identical source at an identical 35,596 bytes**, and another
saw host hashes move for arms that changed only device code, because that section embeds
build-path-dependent material. Prove a host-side knob by **exporting the decision it makes and
reading it back** (e.g. dump the planner's per-case geometry through `ctypes`), and confirm it is
host-only by showing the **device** `.text` is byte-identical to the control.

---

## Phase 1 -- Control

**A stored baseline is not a control.** Cross-session score drift is ~0.4, which is larger than
most effects being chased. Re-measure the OLD artifact in the SAME session, on the SAME card, beside
the new one. Otherwise a null change reads as a regression and a regression reads as a win.

**Devices.** Exclude **0** (computes silently wrong answers while `npu-smi` reports Health OK) and
**3 and 5** (another tenant). Usable: **1, 2, 4, 6, 7**.

Before claiming a card: `npu-smi info -t proc-mem` must show zero processes, **twice, 60s apart**.
Re-verify exclusivity immediately **before and after every timing run** -- a contended card silently
corrupts timing and there is no error to tell you. A sibling agent's stray evaluation contended
device 1 for 23 minutes in this campaign and the affected timings were only caught because the
window was known.

**Do not exceed one agent per usable card.** Seven agents against five cards means two are racing.

**Effects under ~3% need replication.** A null control and a tight CI are not sufficient on their own.

---

## Phase 2 -- The free gates

### RUN THE AUDIT TOOL FIRST. IT IS THE CHECKLIST, AND IT IS NOT OPTIONAL.

```bash
cd "$CANNBENCH_REPO"          # the campaign repo holding submissions/ and tools/
python3 tools/preflight_kernel_audit.py submissions/<op>          # one tree
python3 tools/preflight_kernel_audit.py submissions/*/ --json out.json   # the fleet
```

It runs **every** class that has cost this campaign hidden cases or credits, self-tests its own greps
against known-bad and known-good strings before reporting (six uncontrolled scans in this campaign
returned zero and were wrong), and exits non-zero on any CRITICAL failure. It also prints the eight
classes it *cannot* decide, so they get a device probe instead of an assumption.

**Why this exists:** every one of these classes was already written down, and nothing enforced them.
Operators were fixed one class at a time, whichever the last hidden run happened to report, and
declared "ready" with three other known classes still open. A fleet run on 2026-09-29 found
**31 of 53 trees with at least one mechanical failure**, including seven already posted:

| class | trees | what it costs |
|---|---:|---|
| `RC` (`rc != 0` instead of `rc < 0`) | **19** | a kernel returning a positive value on success is read as failed -> operator ZEROED |
| `C86b` rejections that do not name axis/value/bound | **19** | an 80-case hidden run that knew the answer teaches you nothing |
| **`C87`** L0C hazard, no `pipe_barrier(PIPE_M)` | **7** | unrecoverable A3 device fault, cascades every later case |
| `C66` hardcoded core count | 2 | ~2-5 points, and a wrong count can fault |
| `C73` early return may skip the launch | 1 | `no_npu_kernel_detected` zeroes the operator |

`strided_slice` is the cautionary row: it already scored **0** on hidden once for exactly the C73
pattern, recovered to 86.16, and the pattern is still in the tree.

**An operator is not "ready" until the audit is run AND read, and every FAIL is either fixed or
recorded with an explicit reason.** A MANUAL verdict on a CRITICAL class means a device probe is owed,
not that the class is clear.



All of these cost nothing: no device, no credit. Run them all before submitting anything.

### C76 -- static cap audit (the highest-yield gate)

Put `desc.md`'s `支持范围` table beside every numeric constant in your own source. **A cap below a
declared maximum is a defect unless you can name the hardware limit forcing it.**

This one check would have caught five operators before they cost anything:

| operator | spec declares | our source | cost |
|---|---|---|---|
| `softmax` | last axis `1 ~ 2097152` | bucket table ending `(12288,1)` | 8 cases |
| `adaptive_avg_pool_3d` | `W 1 ~ 256` | `LAUNCH(...,128) LAUNCH(...,144)` | 3 cases |
| `resize_bilinear` | -- | `W <= 4096` + "exactly representable rational" scale | 4 cases |
| `conv_2d` | `K_h 1 ~ 16` | `khkw in {1,9,25}` | 3 cases |
| `grouped_matmul` | bias dtypes incl. bf16 | bf16 bias unimplemented | 23 cases |

Five mechanisms -- template instantiation list, bucket table, lookup table, unimplemented dtype
branch, hard cap with headroom over the largest visible case -- and one shape: **a supported set
fitted to the cases that happened to be visible.**

### THE HIDDEN SET DOES NOT RESPECT `desc.md`. CLEARING THE DECLARED SURFACE IS NOT ENOUGH.

**Read this before the detector below, because it bounds what the detector can buy you.**

Worked failure, `mla_prolog`, 2026-09-29. An agent enumerated **all 49,152 combinations** of the
declared surface and took ACCEPT from 36,864 to 49,152 — genuinely exhaustive over every axis
`desc.md` gives a range for (`B`, `S`, `He`, `N`), holding the rest fixed because the spec says
**固定**. Standard run: 68.35, a clean new entry. Hidden run: **50.97, 28 of 80 failing, every one a
rejection across NINE different rc codes** — on the axes the spec calls fixed:

| axis | `desc.md` | what the hidden set used |
|---|---|---|
| `Hcq` | **固定 1536** | 2048, 1536, 768, **64** |
| `D` | **固定 128** | **64**, **127**, **256** |
| `Hckv` | **固定 512** | 1024, **511**, **257**, 256, 64 |
| `Dr` | **固定 64** | **63** |
| `He` | **{6144, 7168, 7680}** | **4096, 4097, 2048, 128** |

Two of those cases (51, 59) fail with `DefaultCPUAllocator: can't allocate memory: you tried to
allocate 17179869184 bytes` — **16 GiB on the host, inside the benchmark's own golden**, at
`He=4097 Hcq=1537 n_heads=127`. The hidden set is reaching past the contract far enough to break its
own reference, so this is not a case of the spec being merely incomplete.

**Consequences, and they change the strategy:**

1. **A kernel must DEGRADE, not reject, beyond the declared surface.** Accepting everything declared is
   the floor, not the goal. Any axis the spec pins to a single value is exactly where a general runtime
   path pays, because a pinned value is the strongest possible hint that the implementation was built
   around it — and the hidden set probes it anyway. `D` declared 固定 128 and probed at 64/127/256 is
   the clearest example.
2. **"Exhaustive" is only exhaustive over the axes you chose to vary.** 49,152 combinations sounds
   complete and covered four axes of eight. When reporting a surface sweep, state **which axes were
   varied and which were held**, and treat every held axis as an untested exposure rather than a
   cleared one.
3. **Non-multiples and off-by-one values are the probe.** `He=4097`, `Hckv=257`, `Dr=63`, `D=127` — the
   hidden set walks one past each alignment boundary. Any `must be a multiple of N` rejection is a
   defect waiting to be billed.
4. **The compile-term asymmetry makes this the expensive class.** All 28 were `compile_runtime_error`,
   so they cost compile (20 -> 13.0) AND function (30 -> 19.5) AND the performance those cases would
   have carried. Realistic ceiling here is 78/80 ~= 76.5 against a posted 68.35, i.e. **~+8**.

**So the audit's C86 check is a floor, not a ceiling.** Treat "no declared value reaches a rejection"
as the minimum bar, and ask separately: *what does this kernel do one step outside every pinned value?*

### THE MECHANICAL DETECTOR: does our accepted set EQUAL the `实测` column?

Before reading anything else in this section, run this one comparison. `desc.md`'s support-range table
carries a **`cases.csv 实测`** column ("what the visible cases actually exercise") beside each declared
range. Put our source's accepted set beside it:

**If our accepted set equals the `实测` set, the contract was fitted to the visible cases.** That is a
defect with no further argument needed, and it is detectable without a device, a credit or a probe.

Worked, `gqa`, 2026-09-28 -- 26 hidden cases, the single largest gain on the board at **+9.22**:

| axis | declared | `实测` | our gate | verdict |
|---|---|---|---|---|
| `D` | **64 ~ 512, 64-aligned** | 128 / 256 | `if (D != 128 && D != 256) return -3` | **equals 实测** -- 6 of 8 legal values rejected |
| `S_kv` | **1 ~ 8192** | 128 ~ 2048 | `if (Skv % 128 != 0) return -4` | **equals 实测** -- every non-multiple rejected |
| `N_q`·`D` | Nq <= 256, D <= 512 | 32-128, 128/256 | `if (Nq*D > 65535) return -8` | declared max is 131072, so the max is rejected |

Three independent fitted caps on one operator, every one matching the exercised set rather than the
declared one. `mha` has the identical shape: a declared-surface probe returned **476 of 673
combinations REJECTED**, led by `D=192`, which its own `实测` column notes never appears.

**And make the rejection say which axis.** `gqa`'s 26 failures all arrived as one line --
`gqa: shape outside the kernel contract (gqa_plan rejected)` -- while the kernel internally
distinguishes `-1` through `-8`. None of it was surfaced, so a remote run that already knew the
answer told us nothing and the defect had to be re-derived locally. A rejection that names the axis,
the value and the violated bound turns one hidden run into a complete defect list. This is now **C86**.

### `desc.md` often tells you the answer outright

The support-range table frequently carries a **`cases.csv 实测`** ("what cases.csv actually
exercises") note beside each declared range. When it does, the spec is *handing you* the gap
between the declared surface and the visible cases -- the exact axis the hidden set probes.

`mha`'s table is the clearest example seen so far:

| dim | declared | `cases.csv 实测` |
|---|---|---|
| `S` | 1 ~ 2048 | 1 ~ 1024 |
| `S_kv` | 1 ~ 4096 | 128 ~ 2048 |
| `D` | 64 ~ 256, 64-aligned | 64 / 128 / 256 -- **192 never appears** |
| `N` | 1 ~ 64 | 8 ~ 32 |

A measured probe of that operator's declared surface came back **476 of 673 combinations REJECTED**
(71%): `D=192` 148, `S_kv % 128 != 0` 128, `S` outside `{1,2}` and not a multiple of 128, 200.
Every one of those axes is annotated in the table above as wider than what the cases exercise.
**Read that column first** -- it costs nothing and it names the defect before you write any code.

### The VALUE-RANGE axis counts too, and it has no shape to grep for

C76 is usually a shape, attr or dtype cap. It can also be a **magnitude** -- a constant chosen so it
dominates the values the visible cases happen to contain, which stops dominating outside them.

`mha` shipped one. Both fast kernels added a causal mask as a fixed `-3.0e4` bias **in the raw score
domain**. A raw `Q.K^T` over `D <= 256` terms grows like `R^2 * sqrt(D)`, so at `R = 100` the real
scores reach ~1.1e5, the bias stops dominating the row maximum, and **the mask leaks**. Measured on
the shipped build at `vr = (-100, 100)`, fp16, causal:

| shape | MERE | MARE |
|---|---|---|
| `B1 S1024 Skv1024 N2 D128` | **2.02e-01** | 4.41e+03 |
| `B1 S128 Skv128 N2 D256` | **1.29e+00** | 4.63e+03 |

The same shapes at `(-1, 1)` and `(-4, 5)` pass. And `desc.md` declares the input value range as
**any finite real**, while its `cases.csv 实测` column shows only `[-1, 1]` and `[0, 0]`.

The fix was one constant (`-3.0e4 -> -1.0e30`) and **provably free**: the two `.aicore_binary`
sections are the *same size* and differ in 176 and 24 bytes -- a changed immediate, not an added
instruction -- and at `(-1, 1)` every MERE/MARE is *identical* to the pre-fix build.

**So when you audit caps, include the magnitude constants**: mask biases, clamps, epsilons, saturation
limits, "large enough" sentinels. A cap on a *dimension* is visible in a shape probe; a cap on a
*value* is only visible if you probe the declared value range (**C71**), and the declared range is
usually "any finite real" while the cases are all near unit scale.

**The discriminator:** a constant that selects a **tile** is correct engineering; a constant that
gates a **rejection** is a fitted contract. Trace each to whether exceeding it yields *a smaller
tile and a correct answer* or *an error return*. Report both classes; only the second is a defect.

`skillyard-cannbench/tools/declared_vs_cap_audit.py` mechanises the triage. It is a **triage list,
not a verdict** -- most hits are tiling ladders and you still read each one.

**Prefer runtime tiling to a longer list.** Adding `256` to an instantiation list fixes the cases you
were just billed for and leaves the next unlisted width just as broken. If you keep a static ladder
for speed, put a general fallback beneath it so **no declared shape can reach a rejection**.

### The other gates

- **C72 declared-surface probe** -- `tools/declared_surface_probe.py`. Varies the **dtype** axis
  against an accepted baseline case. Note its limit: it is blind to shape and attr, which is where
  C76's five failures all live. The two gates are complements, not alternatives.
- **Check the `compare: false` outputs -- exactly BECAUSE no gate is watching them.** `proto.yaml` marks
  some outputs uncompared, so the evaluator ignores them and they earn no function marks. That makes
  them the one place in the tree where a wrong answer is **free forever**, and it is where the next real
  defect hides. Audit them when they are a **pure function of the shape** (an index, a count, an
  offset table) -- no tie freedom means any mismatch is unambiguously our bug.
  `moe_gating_top_k_softmax`'s `row_idx` was wrong for **892 of the 1024 declared `k` values** (every
  non-power-of-two `k`) in the shipped build, undetected, because `compare: false` and every visible
  case uses a power-of-two `k`. Cost: zero marks, but it is a wrong output we were shipping. Free to
  check: it is derivable on the host.
- **C73 -- a legal no-op must still LAUNCH.** A profiled window with no NPU kernel trips
  `no_npu_kernel_detected`, which **zeroes the entire operator**. `strided_slice` scored 0 on hidden
  for this and recovered to 86.16 -- the campaign's largest single recovery. An early
  `if y.numel() == 0: return y` in the driver is the classic form.
- **C75 -- an rc read after an enqueue is always zero.** If the launch is asynchronous, the return
  code you check is meaningless. Scan for the lambda's *definition*, not forward from the call site;
  a forward-only scan returns zero hits and looks clean.
- **C66 -- query the core cap, never hardcode it.** A hardcoded 48 was 86% of an A2->A3 penalty.
  `torch.npu.get_device_limit()`. Check the clamp inside `call_kernel` too -- a host-side query alone
  is silently reverted by it.
- **Return codes are not all zero-on-success.** Some kernels return the block count. Test `rc < 0`,
  not `rc != 0`, or you zero a working operator.
- **Upload allowlist:** `.hpp .h .cpp .py .cmake .txt .md .sh .gitignore`. **The extension audit is
  the LAST step, after any build.** Running `build.sh` after auditing recreates `build/` and the
  upload is rejected HTTP 400.
- **`submit_kernel` ZIPS THE DIRECTORY -- `.gitignore` protects NOTHING.** This is the trap that has
  caught every agent so far, because "upload-clean" was reported after auditing with *git*
  semantics while the uploader uses *filesystem* semantics. Two trees packaged and declared clean on
  the same day:

  | tree | off-allowlist files actually present | contents |
  |---|---:|---|
  | `softmax` | 5 | `_compile.log`, 2 **extensionless** (`PKG-INFO`, `not-zip-safe`), `.so`, `.whl` |
  | `rms_norm` | **54** | all of the above plus a full `build/` with `.bin`, `.marks`, `.make`, `.yaml`, `.json` |

  `build/` with `.marks`/`.make`/`.bin` is the recorded HTTP 400 signature. Both had passed their
  own audit, because `build/ dist/ *.egg-info/ _compile.log cann_bench/*.abi3.so __pycache__/` are
  all gitignored -- and all of them ship anyway.

  **Audit with `find`, not `git status`:**
  ```bash
  find . -type f ! -name "*.hpp" ! -name "*.h" ! -name "*.cpp" ! -name "*.py" \
    ! -name "*.cmake" ! -name "*.txt" ! -name "*.md" ! -name "*.sh" ! -name ".gitignore"
  find . -type f ! -name "*.*"        # extensionless -- must be zero
  ```
  **An IMPORT is enough to re-dirty it -- a rebuild is not required.** Measured 2026-10-05 on `gru`:
  the audit passed at 502 files / 0 off-allowlist, then a still-running probe imported the tree's
  shipped driver and Python wrote back a `.pyc` -- 503 files, 1 off-allowlist. `__pycache__` is
  gitignored, so `git status` could never have shown it. **So the rule is not "do not rebuild after
  the audit", it is "nothing may touch the tree after the audit":** make the clean pass the last
  action, after confirming every process of yours has exited, then re-verify `--check` exits 0 AND
  the shipped md5s are unchanged.

  **Move the residue aside, do not delete it** (into a scratchpad, so a re-measure is still
  possible) and **do not rebuild afterwards** -- a rebuild is what recreates it.

  `skillyard-cannbench/tools/clean_submission_tree.sh <tree> [--check]` does exactly that and
  never builds. Run it as the last step before every submission; `--check` is read-only.

  **Scale of the problem, measured 2026-09-27: all 49 trees not yet swept carried residue**,
  between 44 and 125 files each (`cross_entropy_loss` 95 -- which is the one that actually hit
  the 400; `grouped_matmul` 125). This is not an occasional slip, it is the **default state of
  any tree an agent has built in**, so treat a tree as dirty until a `--check` says otherwise.
- **STRIP THE 7 `.inl` + 2 `formula_params.csv` UNDER `third_party/pto-isa`. They ARE a rejection
  cause**, confirmed by the server naming all nine:
  `[WEB-ZIP-006] disallowed file extension: .inl (third_party/pto-isa/include/pto/costmodel/perf_sim/pipe_model_impl.inl)`.
  They are safe to remove: our build has **zero** references to `costmodel`/`perf_sim`, they are
  included only from `costmodel/perf_sim/{reporter,pipe_model}.hpp` and read only by a generator
  script, and a post-strip rebuild is **byte-identical**.

  > **Worked error, recorded because the reasoning is the reusable part.** I first concluded these
  > were harmless, from the fact that four already-posted operators had the files sitting in their
  > trees. That inference is invalid: **an mtime survives a copy or a checkout**, so a file being
  > dated before a submission does not show it was in that submission's zip. The one operator that
  > actually succeeded that day had had the nine files **stripped by its own agent** -- which I saw
  > and misread as "that tree just has no `build/`". I then wrote the wrong conclusion into this
  > skill, which would have failed every later submission.
  >
  > **The rule: a successful submission is evidence about the ZIP THAT WAS SENT, not about the tree
  > as it stands now.** To test whether a file class is tolerated, read the rejection message -- it
  > names the field -- rather than inferring from the state of trees that have been rebuilt since.
- **A code change always costs TWO credits, and the arithmetic has a trap.** `submit_kernel` is
  **standard-only**; `rerun_hidden_cases` reuses the **same zip** and needs a terminal standard job
  above 50 on it. So: fix code -> fresh standard -> hidden = 2 credits. A 1-credit hidden re-run is
  possible only when an existing submission already has a terminal standard pass **and** no hidden run
  against it.
  **The trap: check whether the SUBMITTED ZIP contains the fix, not just that a terminal standard job
  exists.** On `conv_3d_backprop_filter` a submission had a terminal 20/20 standard job and no hidden
  run, which reads as "1 credit for +1.21". But that zip was the *four-fix* wheel and the tree had
  moved ahead of it -- the case-90 fix was never built into a wheel -- so hidden case 90 re-fails
  deterministically, the run lands at 79/80, and a partial **posts nothing**. The cheap route was a
  guaranteed zero. Two agents disagreed on this and the one that had read the wheel was right.
- **A hidden run costs TWO credits, not one.** `rerun_hidden_cases` requires a *terminal standard
  job on the same submission scoring above 50*, so a newly fixed kernel needs a standard run first.
  Budget accordingly, and note the standard run is not wasted -- it re-banks the entry and confirms
  the fix cost no performance.
- **A job that dies in the runner's own stage is AUTO-REFUNDED. A hung job is not a lost credit.**
  `conv_3d_backprop_filter` sat in `status: archiving` for ~6h with the runner `online` and the job
  in its own `storage.terminal_unreported_job_ids`, then resolved to `status: timed_out`,
  `error_code: inflight_timeout`, `failure_kind: hard_failure`, `failed_stage: archive`
  (`在途超时：21620s 无阶段事件进展（阈值 21600s）` -- the threshold is **6h** of no stage event).
  `get_credits` then read `charges: 1, refunds: 1, used: 0`. The **score** is lost
  (`has_results: false`, `result_score: null`) and must be re-earned; the **credit** is not.
  So waiting out the 6h is free -- never re-submit in a panic before the timeout resolves, and
  never let a hung job become a sunk cost that justifies a rushed decision.
  **Check whether it is the runner, not the site:** `list_runners` showed 8 online 910c runners,
  all idle, and only runner-3 carried `terminal_unreported_jobs: 1` -- the other seven were at 0,
  including one with *more* workspace jobs (1125 vs 999). Workspace pressure was not the cause and
  a re-submit has ~7/8 odds of landing on a clean runner.
  **And read a failed-to-report job's logs before writing the run off.** This one had already
  compiled and run **20/20 with perf collected on both cards** before dying in archive, which
  empirically retired a flagged C66 risk in that zip: the uncommitted `c10_npu::GetDeviceResLimit`
  core query builds in the remote image. A missing header there would have zeroed the operator.
- **An upload can time out transiently.** Three timeouts on one operator were hypothesised to be
  size (pre-zipped to 1.2 MB -- still failed, so size was falsified) and a 7-minute pause fixed it.
  If a submit hangs, wait rather than re-cutting the zip.

### The evaluation timeout is PER CASE, summed over the TaskUnit -- not per operator

`src/kernel_eval/eval/process_pool.py:690`:

```python
timeout = len(task.cases) * self.process_config.timeout_per_operator
```

So the evaluation child's allowance is **`timeout_per_operator` (300 s) multiplied by the number of
cases** -- on the order of **24,000 s** for an 80-case hidden set, not 300 s for the whole thing.
Measured corroboration: a visible-20 run took **224.85 s against a 20 x 300 = 6000 s timeout**, and
only **29.1 ms** of that was kernel time; the rest is `DataGenerator` and the CPU golden.

**Do not take the 300 s figure from a comment citing `evaluator.py:1241`** -- that is the
**PyPTO-Pro generator** subprocess, a different thing. This exact misreading made `unique` ship a
cost-ceiling refusal at 250 ms/execution that rejected **three hidden cases computing the correct
answer in under a second**, turning winnable cases into a self-inflicted `compile_runtime_error`
class that also billed the compile term.

**Two things that remain true:** the budget is **pooled** across the TaskUnit, so one pathological
case can still consume a neighbour's share -- keep margin rather than spending to the limit; and the
asymmetry still holds, since **a refusal costs only its own case while a timeout can take the unit.**

**A cost model that gates admission must be validated on every flag that changes its cost.**
`unique`'s model over-estimated `return_inverse=False` (conservative, as designed) and
**under-estimated `return_inverse=True` by up to 4.2x** -- that path re-reads the input *and* streams
int64 ranks per window, a streaming coefficient of ~0.0135 ms/win/MiB against 0.0028 -- and **the
error grows with size, so a small-shape probe reads the model as safe.** Raising the budget without
re-measuring that branch would have admitted a 27 s/execution case believing it was 6.4 s. Give every
admitted branch its own measured over-estimate factor; **the branch that is 5x more expensive is
exactly the one nobody measures.**

---

### THE BOARD IS THE CEILING TEST FOR **PERFORMANCE** TOO, NOT JUST ACCURACY

The `roi_align` lesson above -- *do not conclude "unreachable" from "we also fail"* -- has a
performance twin, and it cost more. Before recording any **structure-limited** or
**at-the-floor** verdict, read the top entries' `case_score_mean` and invert the operator's HAP
form to get their runtime. A competitor 3x faster at the **same pass count** means the verdict is
about *our* structure and nothing else.

**Worked failure, 2026-10-09 -- `lstm`.** Our own measurements said serialisation-bound and sitting
at the floor of the structure: the `block_dim` sweep fit **90-94% serial**, a fixed **7.9 us/step**
with 75-96% idle, and an ablation capped a redesign at **+1.030** against a modelled +4.75. Recorded
verdict: vec-resident NO-GO. Then the board:

| entry | score | `case_score_mean` | `avg_speedup` | implied T |
|---|---:|---:|---:|---:|
| SLAI_8 | **61.1405** | 0.2228 | 0.3704 | **~32 us** |
| SLAI | 60.9685 | 0.2194 | 0.3605 | ~33 us |
| SLAI_5 | 60.6023 | 0.2120 | 0.3530 | ~34 us |
| m0_73877465 | 60.4453 | 0.2089 | 0.3390 | ~35 us |
| **ours** | **53.7597** | **0.0752** | **0.0933** | **~112 us** |

All five are **20/20**, so the entire **7.38-point** gap is the performance term. Implied runtimes
come from inverting lstm's floor-case `HAP = 9/(T+8)`. **Four independent teams at the same plateau
is a demonstration, not an outlier.** Every measurement we took of our own kernel was correct; the
inference that the floor we hit was the *problem's* floor was not.

**The checks, all free:**

- `get_operator_rank` on any terminal job, and read `top3` + `personal_best` **together**. A
  `case_score_mean` beside ours at an equal pass count is a direct ratio of runtimes, with the
  baseline cancelling out.
- Invert the HAP form for their `case_score_mean` and compare the implied runtime against your
  **measured** floor. If their time is below the floor you recorded, the floor is a property of
  your schedule, not of the operator.
- **Re-read your own `RESULTS.json` before pricing a redesign.** It already carried
  `best_A3 = 58.36, their_speedup = 0.251, gap = -4.09` for lstm; nobody looked, and the redesign
  was priced against a model instead. The gap is now -7.38.
- Separate the two boards before saying "frozen". A **hidden** ceiling below the posted entry
  freezes what *hidden* can do; it says nothing about the entry, which a faster kernel improves
  directly. Conflating them retires live headroom -- see
  "There are TWO boards" above.

### Score it locally first

Use cann-bench's own evaluator, never a hand-rolled gate:

```
pip install --no-deps --target <private dir>   # then PYTHONPATH
ASCEND_RT_VISIBLE_DEVICES=<card>  run_evaluation.sh --operator <Name>  # NO --source-dir
```

- **Never pass `--source-dir`** -- it triggers `uninstall_packages` and breaks every concurrent
  sibling agent. This is the single worst multi-agent footgun here.
- The evaluator **ignores `--device-id`**; pin with `ASCEND_RT_VISIBLE_DEVICES`.
- It calls bare `python` from PATH.
- **Exactly 50.00 with `avg_speedup` 0.0 is the concurrency-collision signature.** Re-run isolated.
- The `综合得分` line prints **only on a full pass** -- read the score from `cann_final_eval_*.json`.
- Use `compare_tensors` and pass `native_output`, or it is over-strict and rejects correct kernels.
- `Config.perf_rotate_inputs` defaults True, so a harness that does not rotate reads ~6.7% optimistic.
- **Three runs, report the median and the spread.** A spread of exactly 0.00 is suspicious.

---

## Phase 2.5 -- THE TWO GATES THAT COST THE MOST, MEASURED 2026-10-05

**Five hidden runs in one day, zero full passes, four of them 78-79/80.** Both rules below come from
that day and each would have prevented several of them.

### GATE: never submit while a named open item exists

In **every one** of those five, the failing class was already written down and measured before the
submission:

| operator | hidden | the item that was already known |
|---|---|---|
| mla | 78/80 | 2 value-range `NaN位置不匹配` cases, found by its own 169-case sweep, filed as "known, priced, accepted" |
| conv_3d | 79/80 | the 135-spec combination battery was generated and **never run** |
| roi_align | 79/80 | fp32 `samp >= 5` at ~17% per draw -- its own report said **"DO NOT SUBMIT"** |
| gru | 79/80 | the `nL=1` precision tail, quantified at **0.93% per draw**, named as the weakest assumption |
| lstm | 29/80 | the eviction branch and the dtype cast, both labelled **`UNMEASURED`** |

**A per-draw rate is not a margin. Convert it to `P(any failure over 80)` before calling it
acceptable** -- 0.93%/draw is ~48% across 80 cases; 17%/draw is a certainty. And **an agent's own
"do not submit" outranks a coordinator's read of the economics**: it measured the surface.

**If a report contains an open item, a `DO NOT SUBMIT`, or an `UNMEASURED` label, that is the
answer.** Not "small", not "bounded", not "hidden-only". Empty.

### GATE: run the `full_pass` projection against the LIVE entry before **any** hidden rerun

The loop already says to do this before *fix* work. It applies identically to the **rerun itself**,
and it matters most on an operator that already holds a full-pass entry. Worked error: gru banked a
standard **57.995**, and a hidden rerun was fired on the pattern that hidden usually scores higher.
It returned 79/80 at 55.542, and decomposing it gives

    function = 30*79/80 = 29.625 ;  perf = 55.542-20-29.625 = 5.917 ;  avg_hap = 0.1198
    FULL PASS = 20 + 30 + 50*0.1198 = 56.0      <-- BELOW the 57.995 already posted

So even a perfect hidden run could not have improved the entry. **Three distinct reasons a hidden run
can be unspendable, all seen the same day:** a **ceiling** (a case no kernel can pass -- mla_prolog at
78/80 because the comparator needs 224 GiB of host RAM), **HAP** (the full pass scores below the live
entry -- gru, lstm, mla), and **both** (unique).

### And a cache that evicts its own working set zeroes the operator

A shape-keyed workspace cache keeping "the 2 most recent" keys suffices for a single-chunk call and
**fails for a multi-chunk one**, which touches one key per chunk plus ping-pong buffers. Each miss
re-runs the zero-fill, which is **one `aclrtMemcpy` per buffer**: measured **96 / 200 / 304** memcpys
in an 8-call window at `numLayers` 4 / 8 / 8-bidirectional, against the budget of **5**. The
default-budget control was 0 before and after, which is exactly why a 20/20 never saw it.

Fix with an **epoch stamp**: one top-level call = one epoch, every key it touches is stamped, and
eviction refuses a current-stamp key -- and **a cache HIT must stamp too**, or a multi-chunk call
whose first chunk was cached leaves that key evictable by its own later chunks.

**Corollary: before replacing a missing remote op with a host round-trip, count the memcpys it costs
in the profiled window.** The `torch.empty` + `copy_`-from-CPU idiom is safe only for a **cached,
once-per-shape** allocation. For a per-call value it is one memcpy per call, and trading a
`precision_mismatch` for a `cpu_fallback_detected` loses the operator instead of a few cases.

---

## Phase 3 -- Submit

### GATE A (hard, do this before anything else): does the local PASS hold in the RUNNER's environment?

**A local PASS is evidence only for the environment it was measured in.** Before submitting, read
the record's own Blockers section and repro instructions. If they name an environment dependency
(an optional extra, an env var, a path), confirm the runner has it -- `get_job`'s
`setup_info.environment` lists what the image actually provides. If it does not, the local number
does not transfer and the operator is NOT submittable, however clean its accuracy line looks.

**Worked failure, 2026-09-28 -- `roi_align`.** The run's own `README.md` said, under **Blockers**:

> The accuracy gate is unreachable without the optional `torchvision` extra [...] All accuracy
> numbers here are with torchvision 0.24.0 installed.

and its repro block required `export ROI_TV_PATH=<a torchvision 0.24.0 install>`. The record was
`packaged: false, submitted: false`. It was submitted anyway on the strength of its `20/20`.
Remote image reports `torchvision: null`. Result: **32.62, 6/20**, 1 credit, for a failure the run
directory had already measured case-by-case weeks earlier -- its F2 table gives case 2 as
`621/0/688` small-value errors and case 8 as `47/0/52`; the remote run returned `624/0/688` and
`47/0/52`. Nothing new was learned.

**The mechanism is worth knowing, because it is a general shape.** `golden.py` prefers
`torchvision.ops.roi_align` and silently falls back to pure Python whose first line is
`x_fp32 = x.float()`. On the fallback the fp64 golden and the same-precision `native` reference are
computed by the SAME fp32 code, so `native_output` comes out **bit-identical** to `golden_truncated`
(measured: `cpu_diff_max == 0.0` on all 18 finite cases). Every band denominator is then pinned at
`max(0,1)=1`.

**But the bands are the PENALTY, not the gate -- do not mistake one for the other.**
`compare_tensors` has a stage-1 fast path at `compare.py:519-535`: if `overall_mere < threshold`
**and** `overall_mare < 10*threshold`, it returns PASS immediately with **all band counters zeroed**
and never runs the band analysis at all. So the real gate is:

```
mare < 0.1    (fp32, threshold 1e-2)
mare < 1.0    (fp16, threshold 1e-1)
```

and the brutal `2^-30` small-value criterion is only what you are judged by *after* missing it. This
distinction is load-bearing in both directions. Reading the bands as the gate makes the problem look
like "be bit-identical to the golden's fp32 expression", which is not achievable and leads straight
to writing the operator off. Reading the fast path correctly makes it "get `mare` under 0.1", which
is an ordinary numerical-accuracy target. On `roi_align` a CPU model of our kernel's exact algebraic
identity -- separable, merged weights, folded `1/count` -- **passed with 3x margin**, which is only
visible once you know the fast path exists.

Two corollaries. A zeroed band counter does not mean a band was clean; it may mean the case never
reached band analysis. And a case failing on **1 element out of 8,349,952** (case 12) is not
evidence of a one-element bug -- it is `mare` being a **max**, so a single position sets it.

**Do not conclude "unreachable" from "the vendor also fails".** The run measured
`torch_npu.npu_roi_align` failing 5/5 probed cases through that gate and concluded no
non-bit-identical implementation could pass. The leaderboard refutes it: on 910c, `jiaqian` 65.21 at
**20/20** (updated 07:40 the same day, same image, same benchmark 1.1.2), `dontknow` 64.12 at 20/20,
`SLAI_9` 63.11 at **80/80**. The benchmark scores a kernel against a baseline *time*, not against
the vendor's accuracy, so the vendor failing says nothing about reachability.

**Secondary check, same gate:** `evaluator_verified` distinguishes a score produced by cann-bench's
evaluator end-to-end (shim + `--skip-install`) from one produced by the agent calling
`compare_tensors` itself, from one produced by a hand-rolled comparator. The first is the only one
that also proves the packaging and harness work. **8 of 15** records carry `false`, four of them
queued as ready-to-send (`cummin`, `grid_sampler_3d`, `top_k`, `gru`) -- `cummin`'s note does record
real `compare_tensors` use, so `false` is not by itself evidence the numerics are unmeasured. Read
the note, not just the flag.

### GATE B -- the pass count, then credits and the quality floor

**First, the pass count -- this is the check that decides whether a credit is spendable at all.**

Write down, for this exact tree:

1. the **projected pass count** on the case set you are about to run (`N` = 20 standard, 80 hidden);
2. **every case still known to fail**, by id, with its cause;
3. for each, whether it is closable by kernel work at all.

**If the projected count is not N, do not spend the credit.** A `partial_pass` posts nothing, so the
delta you computed is uncollectable. See the pass-count gate in Phase 0 for the four operators this
was got wrong on in one day, and for the "permanently unspendable" verdict that applies when the
benchmark's own golden cannot run a shape.

Two corollaries that are easy to miss:

- **A standard full pass does not imply a hidden one.** The hidden set is 4x and probes outside
  `desc.md` entirely, including axes the spec marks 固定. An operator at a clean 20/20 whose declared
  surface has never been walked has an **unknown** hidden pass count, not a good one -- so the honest
  entry in the table above is "unknown", and walking the surface is the next task, not a submission.
- **A code change costs TWO credits** (fresh standard, then hidden), so the pass-count check applies
  to *both* runs. Passing standard and failing hidden spends two credits to post one entry you
  already had.



- `get_credits` first. Base 10/day, `bonus_cap` 30, resets 16:00Z, `max_active_jobs_per_user: 3`.
  **`bonus_earned` above the cap means bonus_balance 0 and no refunds** -- every credit is one-way.
- Standard first. Spend a hidden credit when the Phase 0 projection says a full pass would beat the
  posted **operator** entry, **or** when the projected hidden score beats our current hidden score
  -- the latter lands on the solution hidden column even as a partial (see "There are TWO boards").
- **A NEVER-SENT OPERATOR STILL HAS TO CLEAR A QUALITY FLOOR. Compute its implied HAP first:**

  ```
  avg_HAP ~= (local_score - 50) / 50        # for a 20/20 run
  ```

  **HAP below 0.5 means the kernel LOSES to the vendor** -- but that is NOT automatically a reason
  not to send. **Check the operator's own leaderboard bar first** (below): on eight of sixteen
  operators measured, the WINNING entry is itself below vendor parity, because the leaderboard ranks
  against the FIELD, not the vendor. Optimise before submitting only when the bar says you are far
  from it. In practice, with the measured local->remote transfer gap of
  **2.7 to 7.5 points**, that is a floor of roughly **77 local** to land 70 remote.

  **Worked failure, 2026-09-27.** Two operators were screened for *hidden safety* with the C76 cap
  audit, came back clean, and were sent on that basis:

  | op | local | remote | implied HAP | headroom to parity |
  |---|---:|---:|---:|---:|
  | `sigmoid` | 70.64 | **63.12** | 0.262 | +11.9 |
  | `gelu` | 70.08 | **62.73** | 0.255 | +12.3 |

  Both were already sitting at the **top of the performance-headroom table** built an hour earlier.
  The error was using the tool that answers *"will the hidden pass?"* to answer *"is this worth
  sending?"* -- two different questions, and the cap audit says nothing about speed.

  Mitigating fact worth knowing: an entry is the best **fully-passing** run, so a better run
  **replaces** it. A weak entry is a placeholder, not permanent damage -- the cost is the credit and
  the interim standing.

  **And a low-HAP cluster is a lead, not just a loss.** `exp` scores **HAP 0.868** on the same
  elementwise-transcendental archetype where `sigmoid` and `gelu` score **0.26**. Same problem class,
  3.4x apart, so "transcendentals are slow here" is falsified by our own board. When several
  operators of one archetype cluster low, diff them against the archetype's best performer before
  starting separate campaigns.
- Tag the submission with what changed and the local evaluator score, so the job list stays readable.
- **Submitting is the user's call.** Package and report; do not submit unasked.

---

## Phase 2.5 -- READ THE OPERATOR'S LEADERBOARD BAR (free, no credits)

Before deciding whether an operator is worth a credit, find out what actually wins it.

```
get_operator_rank(job_id=<ANY terminal job of yours>, operator_name=<CamelCase>)
```

**The job need not involve that operator** -- the board is looked up independently, so one old job
is enough to query all 53. It costs **no credits**.

**Three gotchas, each of which cost a wrong conclusion:**
1. **The name must be CamelCase, not snake_case.** `TopK` returns 33 entries; `top_k` returns 0.
   Four operators first read as "no competitors" purely because of this.
2. **The board MIXES HARDWARE** -- 910c (A3), 950pr and 950dt appear in one ranking. Filter on
   `hardware == "910c"` for an A3 comparison; several #1 entries are 950pr and are not comparable.
3. **`total_entries == 0` does NOT mean uncontested.** `masked_scale` returns 0 on its own job where
   we scored 94.55. Cause unknown -- do not read 0 as "no competitors".

### The bar is PER-OPERATOR, and it ranges 58 to 94

Measured across 18 operators, 2026-09-28. It tracks **how beatable the vendor baseline is**:

| operator class | leaders' speedup | the bar |
|---|---|---|
| recurrent L4 (`lstm`, `gru`) | **0.25 - 0.32** -- nobody beats the vendor | 58 - 60 |
| gather/scan (`top_k`, `scatter`, `cummin`, `grid_sampler_3d`) | 0.48 - 0.67 | 62 - 66 |
| conv / quantised matmul | 1.1 - 2.1 | 75 - 84 |
| elementwise transcendental (`exp`, `sigmoid`) | **4.5** | 93 - 94 |

**So "vendor parity" is the wrong yardstick.** A rule of the form "HAP below 0.5, do not send"
excluded sixteen operators of which eight were within 1.6 points of their own bar. Compare to the
BAR, not to 0.5.

**And remember the arms differ:** the bar is a REMOTE score, your local is A2. Subtract the measured
local-to-remote transfer gap (2.7 to 7.5 points on this fleet) before concluding you are competitive.

## Phase 4 -- Retrieve and decompose

`get_job` and `list_jobs` **overflow the context window**. They save to a file; use `jq` on it.

```bash
F=<saved tool-result path>
jq -r '.result.job.results.summary' "$F"
jq -r '[.. | objects | select(has("case_id") and .status=="failed")] | .[]
  | [(.case_id|split("_")|last), .failure_type,
     ((.failure_reason//"")|gsub("\n";" ")|.[0:170])] | @tsv' "$F" | column -t -s$'\t'
```

Then pull `shape / dim / O / R / I / dtype` out of the rejection messages into a table. **The table
is the deliverable** -- it is what a fix agent needs and what you cannot reconstruct later.

### A DEVICE FAULT NAMES ITS OWN CAUSE -- READ THE `aicore error` BLOCK, NOT `failure_reason`

When a job reports a device fault, the `get_job` payload contains the **AI Core exception**, and it is
far more specific than the per-case `failure_reason` string you see first. Worked failure, 2026-09-28:
`conv_3d_backprop_filter` returned 31/80 with 42 `cascade_device` skips, and I decomposed it from
`failure_reason` alone, concluded the cause was a hardcoded core count, and briefed an agent on that.
**The payload I had already downloaded contained:**

```
aicore error, error code = 0x40000
errorStr: The address for VEC to read L0C conflicts with that for CUBE to write L0C
blk: 0..23   (all 24 Cube blocks, 2 hits each)
```

The hardware named the exact hazard. The core-count hypothesis was then **refuted** two independent
ways -- a static GM-address audit at six `(NAIC,NAIV)` settings gave identical in-bounds ranges, and 100
on-device runs at five core counts inside poisoned guard regions gave 0 numerical failures and 0 guard
violations, with a forced-wrong 7/7 build scoring **60.18 at 20/20**. A wrong core count costs
performance, not correctness.

**So: on any `cascade_device` / `AclrtSynchronizeDeviceWithTimeout` / ACL 507015 / 507035 result, grep
the saved payload for `aicore error` and `errorStr` before forming a hypothesis.** It is free, it is
already on disk, and the string is usually the answer. Reading `failure_reason` and stopping cost a
wrong root cause, a wrong agent brief, and a round trip.

**And reproduce the fault locally before believing a fix.** Forcing the hazardous path on for a small
tile reproduced the identical error code, error string and ACL code on our own A2 -- and that reproducer
is the only reason two plausible-but-wrong fixes were caught rather than shipped. See **C87**.

### Failure taxonomy

| `failure_type` | prefix | meaning | hits compile term? |
|---|---|---|:--:|
| `compile_runtime_error` | `AI算子执行失败` | our kernel rejected or crashed | **yes** |
| `precision_mismatch` | `精度不达标` | comparison failed | no |
| -- | `Golden执行失败` | **benchmark defect, not ours** | -- |
| anti-cheat | -- | `no_npu_kernel_detected` etc. | zeroes the operator |

### THE `native_output` DISCRIMINATOR DOES NOT APPLY ON THE NaN-POSITION GATE

A correction to how this campaign has been triaging, and it cost a wrong "already fixed" conclusion.
The usual discriminator -- *if the same-precision `native_output` sides with the fp64 golden the
defect is ours; if it sides with us, no fp32 kernel can pass* -- rests on `compare_tensors` reaching a
`native_output` fallback. **On the NaN-position gate it never gets there.** `compare.py` returns at
that gate *before* computing any magnitude statistic and before consulting any band, so **there is no
same-precision escape hatch**. A NaN-position divergence the fp32 reference *shares with us* still
loses the whole case.

So "native sides with us, therefore it is an fp32 limit and not our defect" is a **fairness**
judgement, never a **scoring** one. Measured on `apply_adam_w`, both builds, same probe:

| gate | pre-fix | post-fix | discriminates? |
|---|---|---|---|
| `our-defect` (native sides with the fp64 golden) | 0 slots -> PASS | 0 slots -> PASS | **NO -- blind** |
| cann-bench's own `compare_tensors` | 36/4536 FAIL | 0/4536 PASS | yes |

The earlier fix round's headline -- "not one remaining failure in which a plain fp32 evaluation of
golden.py agrees with the fp64 golden and this kernel does not" -- was **true, and case 68 was still
failing.** A regression gate built on that criterion passes a build the benchmark rejects.

**Rule: the verdict gate is always cann-bench's own comparator.** Keep an `our-defect` arm if useful,
label it a diagnostic, and never let it decide whether a fix is done.

### Two readings that are traps

- **`MERE=0.000000, MARE=0.000000` with `NaN位置不匹配` does NOT mean bit-exact.** `compare.py:389-399`
  returns dataclass **defaults** when the NaN-position check fails -- the comparison exited *before
  measuring anything*. `total_count: 0` likewise means "exited", not "empty tensor". Inf mismatches
  saturate and continue, so they do produce real numbers.
- **A tiny `baseline_perf_us` means a small tensor.** Use it to tell whether a numerical failure is
  related to the large-shape defect you are already chasing or is an independent bug. In `softmax`,
  a `baseline_perf_us` of 59.28 proved the one precision failure had nothing to do with the R cap.

### Read the structure, not just the messages

Group the failing shapes and look for what they share. From `softmax`'s eight rejections:
`I == 1` on every one (no strided path needed), three at only 1.33x the cap (no streaming needed),
and `O == 1` on five (the row-per-core grid is the actual defect, not the constant). None of that is
in the error text; all of it changes the fix.

---

## Phase 4.5 -- TRIAGE THE WHOLE BOARD BEFORE YOU FIX ANYTHING

**A hidden failure is not a work item. It is a candidate for one.** Run
`skillyard-cannbench/tools/triage_hidden.py` across every open front and act on the ranking, not on
whichever failure arrived most recently. It is free and takes seconds.

**Two dimensions, and ranking on one of them is how you waste a campaign:**
1. **Worth** -- the Phase 0 projection: `full_pass - posted_entry`.
2. **Fixable** -- is it ours to fix at all?

Ranking on worth alone put `moe_finalize_routing` **top of the list at +7.66** when its golden
reference *crashes on the six failing cases* and no kernel change can recover them.

Measured on 11 open fronts, 2026-09-27:

| gain | operator | verdict |
|---:|---|---|
| +7.66 | `moe_finalize_routing` | **not ours** -- benchmark defect, gain not realisable |
| +6.83 | `apply_adam_w` | **worth it** -- diagnosed, never implemented |
| +6.64 | `rms_norm` | **adjudicate** -- may be a comparator band gap |
| +6.38 | `grouped_matmul` | **adjudicate** -- CPU ref errs MORE on one case |
| +4.23 | `adaptive_avg_pool_3d` | **worth it** -- already fixed, just needs submitting |
| +1.59 | `engram_gate_fusion` | marginal |
| -3.05 | `cross_entropy_loss` | **close it** |
| -3.25 | `grouped_matmul_swiglu_quant` | **close it** |
| -8.89 | `resize_bilinear` | **close it** |
| -18.07 | `sparse_flash_attention` | **close it** -- its problem is performance, not correctness |
| ? | `conv_2d` | re-run needed; fixed since, hidden never repeated |

**Eleven "open fronts" resolve to two pieces of actual work.** Five are *decisions*: a negative gain
means the hidden run, even perfect, scores below the standard entry already posted. Fixing those is
correctness and paper work, never board work, and leaving them "open" manufactures a backlog that
does not exist.

**Close a front explicitly.** Write the projection into `RESULTS.json` with the verdict. An
unresolved failure with no recorded decision gets re-diagnosed by the next session, which is how
the same six operators got re-analysed three times in one campaign.

**A `MARE`-fails/`MERE`-passes failure can still be a GROSS bug of ours.** `rms_norm` looked exactly
like benign cancellation -- `MERE 3e-6` against a `9.8e-4` bound -- and was a **nondeterministic data
race** that replaced a whole output row's normalisation with `1.0`. Two tells separate them:
- `compare.py` has **three** fallbacks, not two: small-value, cancellation, **and a normal-band**
  same-precision check (all `npu/max(cpu,1) <= 2`). A failure landing in the *normal* band, with the
  CPU reference clean, cannot be excused by any of them -- and in that branch the displayed MERE/MARE
  are the normal-region values, which is why MERE reads small.
- **`MARE < 1` strictly** is the whole-row-scale fingerprint: `rel = |1 - scale|` at every element.
  A dropped or stale element gives `rel -> 1.0` exactly.
So dump **all six** counters (`small_value`, `cancel`, **`normal`**, each npu/cpu) and read the band
of the worst element. `band=NEITHER` with an O(1) `|golden|` and a 70-85% error is not cancellation.

**The adjudicate class is the one to be careful with.** A failure where MERE passes by 19-300x and
only MARE (the *max* element) fails is usually near-zero cancellation, and the comparator's own
fallbacks may or may not cover it. Dump `small_value_error_count / small_value_cpu_error_count /
cancel_error_count / cancel_cpu_error_count` and compare against the CPU reference **before**
writing any kernel code:
- our errors within 2x of the CPU reference's -> **benchmark-side**, do not touch the kernel
- thousands against the reference's zero, with MERE itself above 0.5 -> **ours**, and gross

`engram_gate_fusion` case 98 is the worked positive example: `2575/0` and `17791/0` with
`MERE 0.84` and `1.66`. No ambiguity. `rms_norm` cases 52 and 58 are the worked ambiguous example --
`MERE 3e-6` against a `9.8e-4` bound with `MARE 0.61` -- and it **adjudicated to OURS**: `normal`
band `31/0` and `128/0`, an infinite ratio, root-caused to a C78 tmp-alias race. Ambiguous is not a
synonym for benign; it means measure the counters.

**When conventional probing finds nothing, change the QUESTION, not the sample size.** rms_norm
survived 675 shape points, 5184 dtype-range points, 72 natural draws at 33M elements and 816 planted
near-zero probes. What found it was a **sensitivity** probe -- recover each row's implied scale factor
and flag `|c-1| > 1e-3` -- plus running **one shape per process**, because 15 of 88 failures needed a
cold launch. A tolerance gate asks "is this within bounds"; a sensitivity probe asks "did the value
change by a structured amount", and only the second sees a whole-row scale error.

## Phase 5 -- Fix

Classify against the known classes before designing anything: **fitted contract (C76)** ·
**missing declared feature (C67/C68)** · **degenerate or non-finite operand (C69)** ·
**device fault on a large shape (C70)** · **dtype extreme (C71)** · **no launch (C73)** ·
**async rc (C75)** · **core cap (C66)** · **benchmark defect (not ours)**.

**Any run that generates or modifies kernel code goes to `subagent_type: "npu-skillyard:stage-pipeline"`.**
Never general-purpose. Check the argument, not the brief.

Hand the agent the **per-case table from Phase 4**, the projection from Phase 0, the device protocol
from Phase 1, and the packaging traps from Phase 2. An agent without the table re-derives it wrong.

**Do not stop at "it passes now."** Performance is 50 of the 100 points. A fix that restores
correctness and leaves the operator slow has recovered a third of what was available.

**Watch for the fix that regresses the passing cases** -- they carry the entire performance term
today. Keep the existing fast path intact and add the general path beneath it.

---

## Phase 6 -- Record

1. A section in `skillyard-cannbench/RESULTS.json`: job id, score, decomposition, the per-case
   table, root cause, the projection, and **what you got wrong** if a prior claim was retracted.
2. Refresh `jobs_cache.json` from a live `list_jobs` **before** regenerating the scoreboard -- it
   goes stale silently, and a stale cache produced a scoreboard that was ten jobs behind.
   `python3 gen_scoreboard.py > SCOREBOARD.md` (it writes to stdout).
3. A durable memory only for findings that generalise beyond this operator.

**Never report a provenance or status column you inferred.** A fabricated "which generator built
this" column in this campaign survived until it was challenged, and it was wrong in both directions.
Report evidence tiers and say what the evidence is.

---

## Verification discipline

Every scan in this campaign that was not controlled was wrong. Six separate bad scans:

- a zsh `nomatch` glob silently killed a whole grep -- 3 sites reported, 10 real
- a grep matched a **comment** and nearly mislabelled a live code path
- a single-line grep missed a two-line pattern across four operators
- a probe synthesised tensors that omitted required attrs -- ~100% rejection across 45 operators, all artifact
- a cross-product treated per-input dtype lists as independent axes -- 15 of 16 "findings" were spec-forbidden
- a scan looked forward from a call site when the definition was above it -- zero hits, looked clean

**Run a positive control on a known-bad case before believing any scan returned zero.** That is what
exposed the last one. And when an agent reports an ISA behaviour or a root cause, probe it before
writing it into the cookbook -- several such claims have been falsified.

## THE FULL-PASS PROJECTION IS ACCURATE TO ~0.1 -- WHEN THE CONVERTED CASES FAILED ON PRECISION

`projection = 50 + 50*case_score_mean` has now been validated end to end. `grouped_matmul`'s 77/80
partial carried `case_score_mean` 0.6903051, giving **84.5153**. The fixed kernel's hidden full pass
delivered **84.6175** -- within **0.10**, and it took #1 of 45 entries.

**But the accuracy of that projection depends on WHY the cases were failing**, and this refines the
"newly converted cases earn little HAP" rule in the optimizer skill:

| failure class of the converted cases | what to project |
|---|---|
| **precision only** (compile already a full 20) | the run's **own `case_score_mean`** -- the cases already ran at full speed and were timed; only the comparator rejected them |
| **cap / shape rejection** (`compile_runtime_error`) | **discount it** -- the case never ran, so after the fix it executes a path that was never tuned, often a slow fallback |

grouped_matmul's three were all `precision_mismatch` on the small-value band, and
`case_score_mean` actually **rose** 0.6903051 -> 0.6923509: the converted cases scored slightly
*above* the mean of the other 77. So for a precision-only conversion, project at the mean and expect
to land on it; discount only when a rejection is being converted.

## A CORRECTNESS GAP PRICED OFF A SUPERSEDED PARTIAL IS A PHANTOM

When surveying for "correctness-recoverable" prizes, the tempting arithmetic is

```
gain = (arithmetic full-pass ceiling of the kernel that produced the partial) - (partial's posted score)
```

**That number is meaningless if the operator's best hidden run is now a FULL PASS.** The prize was
already collected; you are subtracting a current score from a *superseded* kernel's ceiling.

Measured, `weight_quant_batch_matmul`: a survey priced it at **+5.83**, from a recorded control
ceiling of 76.58 minus a 2026-09-29 partial's 70.7754. But the very next day
`job_dbbef1bd4ed4` **succeeded at 80/80, 72.1974** -- and the operator is absent from
`hidden_partials.json` for exactly that reason. **There was no partial left to recover and the
correctness prize was ZERO.** An agent spent a session establishing that.

Worse, the ceiling it was priced against was computed on the **pre-fix, wrong-answer**
configuration, and the fix that converts those cases is **not free**: `WQ_PLANES 2->3` +
`WQ_RADIX 127->254` costs **-2.218 points** (exact permutation p=0.0286, complete separation), and
the re-association a further -0.2..-0.35. So `76.58 - 2.22 - 0.3 ~= 74.1`, the session projected
74.58, and it **delivered 72.1974**. A second clean instance of accuracy being anti-correlated with
score, and another case where deltas do not decompose because the score saturates.

**The gate, and it is free:** before pricing any correctness gap, check whether the operator's best
hidden run already passed. `hidden_best_per_op.tsv` / absence from `hidden_partials.json` answers it.
If the best run is a full pass, the correctness column is **closed** and the only remaining prize is
performance.

## VERIFY THE TREE IS THE CONFIGURATION THAT EARNED THE BANKED ENTRY

A submission tree is a working directory, and sessions leave candidates in it. **Check the shipped
source's hash against the recorded shipped identity before packaging anything.**

Measured: `weight_quant_batch_matmul`'s tree held sha1 `d0ee8d0e` -- the unsubmitted **a03** tiling
variant -- while the recorded shipped identity was `b00ecc68`. a03 had never been through the hidden
set. **The next person to package that operator would have shipped it silently**, with no step in
the normal flow catching it, because the packaging audit checks *file extensions*, not *identity*.

Why it matters even for a "pure tiling" change: **a tile change is a summation-order change.**
Hidden cases whose M or N selects a different tile round differently -- 2 of 20 visible cases moved,
so ~8 of 80 edge-probing hidden cases would -- and that operator had two cases sitting at only
**1.2x** headroom on the comparator's stage-1 fast path. A banked full pass is guaranteed **only for
the configuration that earned it** (band totals pin a CONFIGURATION, never an IMPLEMENTATION).

**So:** record the shipped sha1 with every banked entry, diff it before packaging, and keep rejected
candidates in a `variants/` directory *outside* the build path rather than in place. If a candidate
is worth shipping, that is a priced decision with its own credits -- a03's was +0.704 local
projecting to ~72.90 against a live 72.20, i.e. inside the coin-flip band and not worth risking a
22-point full pass.
