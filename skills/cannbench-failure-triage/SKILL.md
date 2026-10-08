---
name: cannbench-failure-triage
description: Turn a failed cann-bench job into a decision, deterministically. Dispatch a failure signature to its known class, cause and action; count real defects rather than failed cases; decide whether a credit is spendable; and name the controls that must exist before any result is believed. Triggers - a hidden run failed, a job scored 0, decompose this job, what does this error mean, is this fix worth a credit, why did N cases fail.
---

# cann-bench failure triage

**Every entry below cost real credits, real device-hours, or a wrong conclusion that was later
retracted.** The point of this file is that a failure should be *dispatched*, not *recognised* --
read the signature, look it up, take the named action.

Use with `cannbench-campaign-loop` (the end-to-end loop and the scoring model). This file is the
decision procedure for when something has already gone wrong.

---

## STEP 0 -- COUNT REAL DEFECTS, NOT FAILED CASES
> **A class marked CLOSED is closed only over the shapes that were probed.** A 264-run sweep declared a
> device-fault class closed; a later session re-opened it and found a **second, different** fault
> **present in the posted wheel** -- because the group that would have exposed it *rejected* those
> shapes, and no other group combined a large `W` with a pad above 2. **9190** declared combinations
> reach it and **8628 were already reachable in the shipped build**. So "do not re-open X" is an
> instruction about *priority*, never about *possibility*: if a new probe reaches shapes the old sweep
> did not, the class is open again.


**Do this before anything else. It changes the answer more than any other single step.**

One device fault makes every later case fail. Pull the per-case list and classify:

```
failed_cases  =  first_real_fault  +  stream-sync casualties  +  skipped cases
```

Signatures of collateral, which are **NOT defects**:
- `ACL stream synchronize failed, error code:507015` / `:507035` *after* an earlier fault
- `设备不可恢复，用例跳过` ("device unrecoverable, case skipped")

**Worked examples:**

| operator | reported | real defects |
|---|---|---|
| `conv_3d` | 49 failures | **1** (42 cascade + 6 compile + 1 precision) |
| `group_norm` | 22 failures | **1** (case 79 faulted; 80/81 sync; 82-100 skipped; **58 of 58 before it passed**) |

**Find the FIRST failing case id and diagnose only that one.** Sort by case id; the fault is the
lowest id with a device error. Everything above it is noise until that one is fixed.

Then compute, from the payload and not by hand:

```
avg_hap   = case_score_mean                       # cross-check: perf/(50/N)/passed
full_pass = 20 + 30 + 50*avg_hap                  # N = 20 standard, 80 hidden
```

and go to **STEP 3** before proposing any work.

---

## STEP 0b -- GET THE EXACT FAILURE SPLIT FROM `get_job`, NOT FROM THE LOG

**`get_job_logs`' `log_tail` TRUNCATES and cannot be counted.** On lstm's 80-case hidden run it
omitted **32,761 of 40,953 bytes** -- 80% of the log -- while still *looking* like a case listing,
which is exactly how a partial tail gets mistaken for the whole story.

**`get_job(job_id)` carries the complete per-case array** at
`results.operators[0].cases`, and each entry has a **`failure_type`**. One count classifies every
case:

```bash
# definitive split: our-defect vs unwinnable vs recoverable-runtime
python3 -c "
import json,collections,sys
cases=json.load(open(sys.argv[1]))['result']['job']['results']['operators'][0]['cases']
print(collections.Counter(c.get('failure_type') for c in cases))" <saved get_job output>
```

`precision_mismatch` / `compile_runtime_error` / `None` (passed). Worked example, lstm
`job_af0654ceaf45`:

```
precision_mismatch 49   compile_runtime_error 2   None 29
```

**Why this is the highest-value free check in the file.** The two numbers price completely different
work. `compile_runtime_error` cases are a **recoverable class** -- a missing runtime op, a rejected
shape -- and each one carries the **compile** term as well as function and performance (see the
scoring asymmetry in the campaign skill). `precision_mismatch` cases already hold all their compile
marks, and most of them may be **unwinnable** (run STEP 2i's floor probe).

In that example the split collapsed a **+0.85-to-+9.30** estimate to its **floor of +0.85**, because
the recoverable class was **2 cases, not ~22**: the larger figure had assumed a hidden-set
distribution over `numLayers` that nothing measured. **Never pick a point inside a range the case
array can resolve** -- and per the rule that an unexercised branch voids a projection, never widen
the recoverable class by assuming how the hidden set samples an axis.

The array also carries `elapsed_us`, `baseline_perf_us`, `t_hw_us` and `perf_score` per case, which
is where the HAP inversion in the campaign skill gets its `Tb`/`Tc`/`Th`.

---

## STEP 0b-2 -- `failure_type` IS TOO COARSE. SPLIT `precision_mismatch` BY **WHICH CLAUSE** FAILS.

**`failure_type: precision_mismatch` is one label over two unrelated defect classes, and conflating
them sent a full day of work at the smaller one.**

The gate is `mere < threshold AND mare < mare_threshold`. **`MERE` is the MEAN relative error and
`MARE` is the MAX.** So which of the two fails tells you what kind of problem you have:

| clause that fails | what it is | how it looks |
|---|---|---|
| **`mere >= threshold`** | **a WRONG ANSWER** | the *mean* is out; tens of percent of positions mismatch |
| `mere` inside, `mare >= mare_threshold` | a precision **outlier** | a handful of positions out of millions |

A mean relative error above 1 cannot be rounding. It means the output is wrong by a factor.

```bash
python3 -c "
import json,sys,collections
cases=json.load(open(sys.argv[1]))['result']['job']['results']['operators'][0]['cases']
cnt=collections.Counter()
for c in cases:
    outs=(c.get('accuracy') or {}).get('output_results') or []
    bad=[o for o in outs if not o['passed']]
    if c['status']=='success': k='PASS'
    elif not outs: k='NO_OUTPUTS'
    elif any(o['total_count']==0 for o in bad): k='NANPOS'          # STEP 0c
    elif any(o['mere']>=o['threshold'] for o in bad): k='WRONG'
    else: k='MARE_ONLY'
    cnt[k]+=1
print(cnt)" <saved get_job output>
```

**Worked example, lstm `job_179867269adf` (30/80).** `failure_type` said `precision_mismatch` for all
50. The clause split said:

| class | n | MERE | positions mismatched |
|---|---:|---|---|
| **WRONG** | **26** | **0.25 to 168** | **27% to 92%** |
| MARE_ONLY | 21 | 1.8e-4 to 0.038 (inside the gate) | 2 to 122 of 51,200 to 1,638,400 |
| NANPOS | 3 | -- | never compared |

The distribution is **bimodal with nothing between**: passing cases sat at `MERE` ~4e-9, the
MARE_ONLY class at ~1e-4, the WRONG class at ~1e1. A precision story predicts a continuum and does
not get one.

**The cost of not doing this split.** An fp64 control, eight association variants, the
k-accumulation order, a tanh decomposition and an mkldnn on/off arm all returned null -- every one of
them aimed at the 21-case MARE_ONLY class, while the larger 26-case class was a wrong answer nobody
had examined. The operator was twice called unwinnable on that evidence, against a competitor
passing **69/80** on the same hidden set while running *slower*. **An external entry that beats your
pass count is a standing refutation of "unwinnable"; reconcile against it before concluding.**

**Two further reads the same array gives you free:**

- **Mismatch FRACTIONS are diagnostic.** `mismatch_count / total_count` clustered at ~90%, ~67%,
  ~45%, ~33%, ~9%. "Right for the first N steps, wrong after" is the fingerprint of a loop or
  recurrence, not of noise -- so dump the error profile **along the loop axis** (timestep, layer,
  tile index) rather than sweeping shapes blindly.
- **A size frontier, if there is one, is visible in `total_count`.** Every lstm PASS had
  `y total_count <= 51200` and `hn total_count <= 1024`, while the WRONG class ran to 33,554,432 and
  131,072. Treat that as an axis to probe, **not** as a shape: a count field is a count (STEP 0c),
  and two cases with identical counts and dtype can differ by an attribute -- lstm cases 39 and 90
  matched exactly on both and one passed while the other was 87.5% wrong.

**Read the thresholds out of the payload, never assume them.** They are per-operator and appear as
`output_results[i].threshold` (and in the log as `threshold=`/`mare_threshold=`). lstm's are
**0.05 / 0.5**, not the 1e-2 / 1e-1 default -- assuming the default misprices every case in the set.

**Operational note:** `get_job` can fail on a large hidden payload with
`IncompleteRead(81558 bytes read, N more expected)` and it is **reproducible, not transient** -- the
80-case lstm payload is ~850 KB. Retrying does not help. Save the payload once when a call succeeds
and work from the file; a previously saved payload for an earlier run of the same operator answers
most questions, since the failure *set* moves slowly between builds.

---

## STEP 0c -- `MERE=0, MARE=0` CAN MEAN "NEVER COMPARED". CHECK `total_count`.

**A failure reporting `MERE=0.000000, MARE=0.000000` is NOT a case whose values are bit-identical.**
`compare.py:390-399` checks **NaN positions first**, before any numeric work, and on a mismatch
returns a **default `CompareResult`** -- `mere=0.0, mare=0.0, max_diff=0, total_count=0`. Those zeros
are **dataclass defaults**. Nothing about the finite values was ever computed.

**The tell is `total_count == 0`** in `accuracy.output_results[i]`. Always read it before interpreting
a zero metric:

| reading | means |
|---|---|
| `mere=0, mare=0, total_count=0` + `NaN位置不匹配` | **never compared** -- a NaN-position early return |
| `mere=0, mare=0, total_count=N>0`, `passed=true` | genuinely bit-identical, or the "all valid positions are NaN/Inf and matched" early return at `compare.py:473-491` |

I misread the first as the second on lstm and priced three cases as a cheap NaN-propagation fix. They
are "we emit a NaN where the reference does not", which is a different and harder defect.

### The small-value band is unreachable when the golden is lossy

`compare.py:586-590`: `small_value_passed = sv_error_count / max(sv_cpu_error_count, 1) <= 2`.

When the operator's `golden.py` casts (e.g. `lstm_layer.float()`), `native_output` **is**
`golden_truncated`, so `cpu_diff == 0` and **`sv_cpu_error_count == 0`** -- which was true in **all
51** of lstm's failures. The denominator collapses to `max(0,1) = 1`, so the band allows exactly
**2** errors, where an error is `|ours - golden| > 2^-30` (fp32; `2^-16` for 16-bit) at any element
with `|golden| < 2^-14`.

Measured `sv_error_count` on lstm's near-miss class: **408, 378, 437, 858, 363, 1957, 45, 221,
1944, 9, 19, 57 ...** against an allowance of **2**, over populations of 945 to 36,048.

**So "the band might rescue it" is not a route for such an operator** -- clearing it demands
bit-exactness over the entire small-value population. **The only route is the stage-1 fast path**
(`mere < threshold AND mare < 10*threshold`). Compute the band's actual allowance from
`sv_cpu_error_count` before proposing a band-based recovery; a clean native reference makes the band
*stricter*, not more forgiving (see also the band-allowance note in STEP 2a).

---

## STEP 0d -- RECOVERING THE HIDDEN REGIME: THE FREE OBSERVABLES CAN CONSTRAIN THE WRONG PARAMETER

The hidden cases are bit-reproducible from their seeds, so a CPU-only sweep can in principle
recover each case's generator settings -- **shapes, dtype and `value_range`** -- by matching the
payload's reported counters. This is free: no device, no credit. But it has a failure mode that
produces a confident, wrong verdict, and it must be checked before the recovered regime is used
for anything.

### First: separate the GOLDEN-ONLY fields from the output-dependent ones

Only fields computed from the **reference alone** can be matched by a CPU sweep that does not run
our kernel. Measured on lstm's payload:

| field | depends on | usable to pin a regime without the device? |
|---|---|---|
| `y_numel`, `hn_numel`, `cn_numel` | shapes only | **yes** |
| `small_value_total_count` = `count(|golden| < sv_thr)` | the golden | **yes** (`sv_thr`: fp32 `2^-14`, fp16 `2^-11`, bf16 `2^-8`) |
| `cancel_total_count` | tests `|output|` | **no** |
| `normal_total_count` = total - sv - cancel | derived from the above | **no** |
| `max_diff`, `mere`, `mare`, `mismatch_count` | ours vs golden | **no** (but see the discriminator below) |

Getting that table wrong is the first way to go wrong: `normal_total_count` *looks* like a pure
shape/golden quantity and is not.

### The trap: a 1-parameter fit silently ALIASES two parameters

A generator usually has more than one magnitude knob. lstm has two that matter -- `w_ih` (input
weights) and `w_hh` (recurrent weights) -- and they are **not symmetric** in what they control:

| held fixed | swept | `small_value_frac` | the control's MERE |
|---|---|---|---|
| `w_hh` | `w_ih` 0.1 -> 2.0 | **0.0004 -> 0.275 (3 orders)** | barely moves |
| `w_ih` | `w_hh` 0.75 -> 3.0 | barely moves | **2e-6 -> 122 (8 orders)** |

**The free observable constrains the parameter that does not matter and is nearly blind to the one
that does.** A sweep over a single combined "scale" fitted the band counters, recovered a gain of
~6.5, and concluded the class was winnable. Two-parameter, the *same* counter fingerprint is
matched by regimes whose error differs by **seven orders of magnitude**. The verdict flipped.

**So: sweep every magnitude knob independently and report which observable responds to which.**
If an observable moves with a parameter the failure does not depend on, it cannot pin the failure,
however exactly it matches.

### The discriminator: OUR OWN measured metric, against the payload's reported one

When several regimes share the free fingerprint, the payload carries one more number that
separates them -- **the `mere` it reports for that case**. Run our kernel at each candidate regime
and keep only the regimes that reproduce it. This costs device time but no credit, and it is
golden-independent in the sense that matters: it tests *our* behaviour, not a modelled reference.

Worked example (lstm case 41, fingerprint `y_sv=0.09598, cn_sv=0.00854`, payload `mere=130`):

| candidate regime | ours | in-family CPU fp32 | exact fp64 | matches 130? |
|---|---:|---:|---:|---|
| `w_hh=1.0, w_ih=1.0` | **1.69e-05** | 1.02e-05 | 1.43e-05 | **no -- excluded** |
| `w_hh=2.0, w_ih=0.25` | 58.3 | 61.5 | 59.9 | no |
| `w_hh=3.0, w_ih=0.10` | **119** | 120 | 123 | **yes** |

The branch the 1-parameter fit had chosen is **excluded by our own measurement**: we read 1.7e-05
there, not 130. The branch that does reproduce 130 has the **exact fp64 answer at 123 and the
in-family fp32 arm at 120** -- we are within 1-3% of both, which is the signature of a
chaos-limited case, not of a defect (STEP 2i).

### What a recovered regime is good for

Two legitimate uses, and one illegitimate one:

- **Legitimate:** showing the hidden set probes *outside* the visible value convention. Across
  lstm's 26 wrong-answer cases, `small_value_frac` runs 0.0000 to 0.663, mapping `w_ih` to roughly
  **0.05 - 2.0**, against a visible convention of `+/-0.1`. That retires "no visible case hits it,
  so it cannot happen" (see [[hidden-set-ignores-desc-md]]).
- **Legitimate:** choosing where to run a reproduction, so the probe is not run at a regime past
  the point where the reference itself fails -- which is where a reproduction can no longer
  discriminate our defect from chaos.
- **Illegitimate:** declaring a class winnable or unwinnable from the recovered regime alone. The
  parameter that decides that is, on this operator, **not recoverable from golden-only data at
  all**; the only handle on it is a measured metric.

---

## STEP 1 -- DISPATCH THE SIGNATURE

### A. Accuracy / metric signatures

| signature | class | cause | action |
|---|---|---|---|
| `MERE=MARE=0.000000` + `NaN位置不匹配` | **NaN-position gate** (C100) | Control flow expressed as arithmetic: `0*Inf` or `Inf-Inf`. The comparator returned dataclass defaults -- it exited **before measuring**. NOT bit-exactness. | Diff **NaN masks only**, CPU-fp64 golden, over the four distributions the generator emits: `[-inf,inf]`, `[nan,nan]` (~50% NaN), dtype-boundary magnitudes, zeros. Fix with **selects, not multiplies**. |
| `MERE=MARE=0.000000` + `bit-exact 比较失败: 1/N 个元素字节不等` | **bit-exact operator** (threshold is `0.000000e+00`) | Representative selection. `-0.0 == +0.0` is true but the bit patterns differ; same for NaN payloads and Inf sign. | Find which element the reference emits among equals, and match its bits. |
| `相消兜底` / small-value band, `NPU/CPU = N/0` | **absolute-bound band** | Gate is `count_npu / max(count_cpu,1) <= 2`, i.e. **up to 2 offenders are tolerated**. `CPU=0` is the fingerprint: the reference is exactly clean there, so the ratio branch never runs. | Target **count <= 2**, not zero, and not "get MARE down". Fix the **denominator**: more precision (one int8 plane = 127x), or compensated summation. |
| `normal` band, `ours/cpu = 1/0` | **strict band** | With `normal_cpu_error_count == 0`, **any** non-zero count fails. | Here the target genuinely **is** zero. A case with a *worse* MARE can pass if its CPU count is non-zero -- so MERE/MARE margins do not predict this. |
| passes at a suspiciously wide threshold | **per-operator override** | `proto.yaml` overrides thresholds and is merged **last**. Defaults can be ~80x tighter. | Read `proto.yaml` before reasoning about any margin. |

### B. Scoring / anti-cheat signatures -- these look like success

| signature | class | action |
|---|---|---|
| `通过率 100.00%` with `综合得分 0.00` | **`aclrtMemcpy` anti-cheat**: >5 host round-trips in the profiled window zeroes compile+function+perf | No per-call host round-trip is possible. Data the kernel needs must arrive via **GM**. Clean: D2D `copy_`, `Tensor.mul(float)`, a cached one-time H2D. Fatal: `.item()`, `float(t.max())`, `.cpu()` per call. |
| `score_error_code = cpu_fallback_detected` | same threshold, `copy_` count | Cache outside the profiled window; a shim that re-copies every weight per call trips it. |
| exactly `50.00` with `avg_speedup 0.0` | **concurrency collision** | Re-run isolated. Not a result. |
| a score that cannot be reproduced | **device 0** | The evaluator always runs logical 0. Device 0 computes wrong answers **and produces false passes** (zeroed UB hid a padding bug: recorded 19/20, real 7/20). Pin `ASCEND_RT_VISIBLE_DEVICES` and re-measure before hunting a regression. |

### C. Device faults

| signature | class | action |
|---|---|---|
| `error code = 0x40000`, *"The address for VEC to read L0C conflicts with that for CUBE to write L0C"*, ACL 507015 | **C87 L0C hazard** | Coverage needs `chain_distance >= edge_distance`. Compute **S** (L0A/L0B slot count) and **D** (edge distance) per edge; `D < S` is exposed. A `pipe_barrier(PIPE_M)` clears it (**measured**: barrier-off faulted 7/19 shapes, barrier-on 19/19). A same-iteration `MTE1->M` ring is **not** a discharge. |
| ACL 507035, *"The vector core execution is abnormal"* / *"UB address accessed by the VEC instruction is not aligned"* | **Vec fault** | UB alignment or out-of-range. Suspect the newest code path, especially one with a single exercising case. One shape per process. |
| `retCode=0x31`, *"MTE accesses an invalid GM address"* | **C107 capacity guard that gives up** | A shrink loop that floors and then proceeds. Look for a sibling path that has a blocking loop this one lacks -- the asymmetry is the defect. |
| any fault, region looks incoherent in the swept axis | **a cost model gates the path** | The failure set is **not an interval**: measured `33 FAULT, 64 PASS, 256 PASS, 257 FAULT, 300 FAULT, 384 PASS, 512 PASS`. A powers-of-two sweep returns CLEAN and is wrong. Sweep **off-by-one around every threshold in the chooser**. |
| small probe clean, large probe faults | **pressure, not tile size** | `M1 K512 N32` read clean; `K=4096` fired immediately. Small shapes **mis-compute**; only high pressure **faults**. |

### D. Rejections and environment -- the expensive class

`compile_runtime_error` drains **compile AND function AND performance**. A `precision_mismatch`
leaves compile at a full 20. Check the split first; it decides the value.

| signature | class | action |
|---|---|---|
| our own rc message naming an axis and bound | **fitted contract** (C86) | Is the accepted max == the `cases.csv 实测` max? Then it is fitted. **Degrade, never reject**: keep the fast ladder, add a runtime-tiled general path beneath. The hidden set probes **outside `desc.md`**, including axes marked 固定. |
| `aclnnInplaceZero ... 561103`, or `Op ZerosLike does not has any binary` | **remote image lacks the op** | A device-side zero-fill. Use `torch.empty` + `.copy_(torch.zeros(...))` from a **CPU** tensor. **Invisible locally** -- our image has the binary. Sweep the **live** path: a `TORCH_LIBRARY` registration makes a Python driver dead code. |
| `no_npu_kernel_detected` | **C73** | A legal no-op must still LAUNCH. Zeroes the whole operator; one operator went 0 -> 86.16 on this. |
| `Golden执行失败` at `op_runner.py:304` (vs ours at `:395`) | **NOT OURS, BUT IT STILL BILLS COMPILE** | The golden crashed and our kernel was never invoked -- the golden runs FIRST (`evaluator.py:318`, returns at `:321`). **But `evaluator.py:332` sets `failure_type = FAILURE_TYPE_COMPILE_RUNTIME_ERROR`**, so the case costs compile AND function AND performance just like one of ours. It is **unwinnable, not free**: price it as a permanent `cf` and exclude it from any recoverable count. Worked example: a hidden `projSize >= hiddenSize` lstm case, because `torch.nn.LSTM` itself raises at `proj_size >= hidden_size`. |
| `DefaultCPUAllocator: can't allocate N bytes` | **harness host OOM** | The reference cannot run that shape. If it is a hidden case, the operator is **permanently unspendable on hidden** -- record once and stop revisiting. |
| `OverflowError: cannot convert float infinity to integer` | **evaluator crash on non-finite data** | Corroborates that the band carries Inf/NaN. Upstream. |
| a local PASS that fails remotely | **environment-bound** | A local pass is evidence only for that image. Read the tree's README Blockers and `get_job`'s `setup_info.environment`. One operator was sent on a 20/20 needing `torchvision`, which the runner lacks: 32.62. |

---

## STEP 2 -- BEFORE BELIEVING ANY RESULT, NAME THE CONTROL

**A measurement without its control is not evidence.** Each of these caught a wrong conclusion.

| claim | required control | what it caught |
|---|---|---|
| "this is faster/slower" | a **same-binary** arm (rebuild the control, confirm identical `.aicore_binary` md5) | a sign test at **p=0.0026** was fully null -- device payload was byte-identical, so a device-time difference was impossible |
| "no measurable change" | **per-case** non-overlap, not just the aggregate | an aggregate that went **UP** hid a real regression: 16 of 19 cases slower, p=0.0044. Also: 3 small cases at +20/+30/+12% cancelled by 3 large ones, while the aggregate sign test read p=0.82 |
| "the scan found nothing" | a **positive control** -- show the probe firing on a known-bad input | nine uncontrolled scans returned zero and were wrong |
| "the constant is absent" | the **corrected** grep, tested against a decoy | `NAME = 48` matched, `#define NAME 24u` did not; that negative cost 2 credits and two shared-card faults |
| "the buffer is zeroed" | **poison** it first, then run the fill | `torch.empty` does not zero; a no-op `copy_` onto a fresh zero page looks identical to a working one |
| "the fix is in the build" | `nm -D` on the **built wheel**, or the object md5 | a fix was `#define`-gated OFF by default and `build.sh` never set the flag -- no `_staged` symbols shipped |
| "this cap is architectural" | find the measured evidence for the claim | one *was* (SYNCALL deadlocks above 24: with a barrier, `block_dim` 25 and 32 never return) -- removing it would have caused a silent hang |
| **"device time did not change"** | **read `elapsed_us`, NOT `t_hw_us`** | `t_hw_us` is the benchmark's **roofline constant**, not a measurement of your kernel: it came back **bit-identical across both arms and all four runs, zero variance**, which read naively is a *perfect null* (0 slower / 20 faster, +0.00%) while `avg_speedup` had moved **0.765 -> 0.662**. `op_times.device_kernels` even names the instantiation, so use it to confirm which build ran |
| "the score belongs to this tree" | the shipped object md5 == the measured arm's | a score that does not belong to the shipped bytes is not a score |
| **"the ablation arm measures the technique"** | the toggle's **definition** -- grep the `#ifdef`'s *body*, not its name | `LSTM_NO_TANHFIX` was **comment-only**: the code it gated had been deleted months earlier, so the arm was byte-identical to base **by construction**. Its clean null is a control on the harness, not evidence about tanh -- and read as evidence it retires the technique. See [[unwired-lever-retires-the-technique]] |

**And the arithmetic trap:** `avg_hap` from a **partial** run is biased -- it is measured on the
cases that ran *before* the fault, which are the early ones. One such value came out **above** the
standard set's, which is backwards, and 42 of 80 cases were unobserved.

---

## STEP 2a -- THE `mare` THE HARNESS PRINTS IS A BAND-DEPENDENT **DISPLAY** VALUE

`compare_tensors` decides a stage-2 failure by **three band tests**, and the `mare` it prints means
different things depending on which band failed. Two consequences, both of which produced a wrong root
cause on the same operator:

**1. The element you name as "the cause" may live in a band that contributed nothing.** One verdict
blamed an element whose golden was `1.55e-7`, reasoning that a denominator of `min(surviving |golden|)`
made the metric a coincidence. Measured: that element lives in the **small-value** band, which had
**0 offenders** against a bound its diff was **150x inside**. The real failure was the **cancel-band
count rule** (`ours <= 2*max(native,1)`, ours 3 vs native 1), and the normal band was 10x inside its
gate. The probe's `worst_golden` was the **overall argmax** while `cr.mare` was the band-dependent
display value -- **two different elements.** Read the **per-band counters**, never the headline.

**2. A PROBE THAT OMITS `native_output` FORCES THE STRICT BRANCH AND IS SYSTEMATICALLY PESSIMISTIC.**
`native_output=None` forces `normal_cpu_error_count = 0`, which selects the strict branch -- but the
**real evaluator supplies it** (`evaluator.py:392-406`; `golden_precision` defaults to `fp64_cpu`).
Measured on one shape: the strict gate failed **5 of 8** draws while the real gate passed **8 of 8**. A
recorded "P(pass) ~ 0.75, no deterministic route" was an artefact of that single omission, and the case
was in fact **already closed** by a fix sitting in the tree (0 of 10 draws over the gate, worst 1.23x
inside).

**A harness that under-reports is as dangerous as one that over-reports** -- it retires real work as
impossible. Mirror the evaluator's construction: `native_output` supplied and computed on the **CPU**
(C89), and run a known-*passing* case through your probe before trusting any failure it reports.

## STEP 2e -- A FITTED CONTRACT CAN LIVE IN A **COMBINATION**, NOT AN AXIS

A per-axis walk cannot see it. Two defects on one operator were invisible to every single-axis sweep:

- `projSize` x `numLayers` -- wrong only when **both** `P` is not a multiple of 16 **and** `L >= 2`.
  Every single-layer `projSize` probe passes, because the corrupted row is only read from layer 1 on.
- `bias=False` x `numLayers` -- a precision floor that fails the real gate only from `L >= 6`, and
  **no visible case combines `bias=False` with `numLayers > 1`.**

**So after walking each axis, walk the cross terms of any axis pair the visible set never varies
together.** Read `cases.csv` for which pairs are *jointly* constant -- that is the blind spot, and it is
cheap to enumerate.

**And off-by-one still matters inside a combination.** The bidirectional half of that first defect fires
at `P = 17..31, 63, 127` and **not** at `32, 48, 49, 64, 96, 112` -- a **non-monotone** set a
powers-of-two sweep calls clean (C107/C108 again, one level up).

## STEP 2f -- YOUR REPLICA OF THE EVALUATOR CAN BE STRICTER THAN THE EVALUATOR

An exact-looking replica of the comparison reported **15/20 on a kernel the real evaluator scores
20/20**. Cause: `golden.py` forces `.float()`, so for that operator the fp64 golden and the
same-precision `native_output` collapse to the **same fp32 CPU run** (verified bitwise equal) -- the
small-value denominator pins at 1 and the gate becomes *"at most 2 near-zero elements may differ by more
than 2^-30"*, unsatisfiable for any fp32 kernel at scale.

**The real evaluator passes them, so the replica's construction of that one input is wrong.** Pair this
with STEP 2a's `native_output=None` trap: **a local gate can be wrong in either direction, and the
under-reporting direction retires real work as impossible.**

**What to do instead when you need a local gate:** calibrate it against the cases the benchmark itself
**certifies**. A CPU-fp64 Frobenius band tuned so the 20 known-passing cases pass is a defensible
instrument; a hand-rolled reproduction of the band logic is not. And keep a **positive control on the
gate itself** -- the shipped kernel failing the same cases your probe flags is what tells you the gate
is mis-calibrated rather than the kernel broken.

## STEP 2g -- AN AVAILABILITY PROBE IS NOT A LOCALITY PROBE

**A host op can return the right answer *because torch_npu silently ran it on the CPU*** -- a D2H round
trip per call, i.e. an `aclrtMemcpy`, i.e. the anti-cheat zeroes the operator at 100% pass.

A pre-build probe verified that `>> 23` **returns the correct value** on an npu tensor. It does. The
build then scored **0.00 at 100.00% pass**, and the evaluator log named it:

```
CAUTION: The operator 'aten::bitwise_right_shift.Tensor_out' is not currently supported
on the NPU backend and will fall back to run on the CPU.
```

**The correct probe measures LOCALITY, not correctness:** run each candidate op in **its own
subprocess**, grep the log for `npu_cpu_fallback`, and include a **known-falling-back op as the positive
control** -- if it does not fire, the probe is measuring nothing. Of **28 ops** checked that way,
exactly **two** fall back: `bitwise_right_shift` and `frexp`. On-device and safe: `aminmax`, `amax(dim)`,
`stack` of 0-dim, `clamp*`, `abs`, `log2`, `floor`, `nan_to_num`, `maximum`, `exp2`, `pow`, `ldexp`,
`Tensor.view(dtype)`, integer floordiv/mul/sub/neg, casts, sliced D2D `copy_`. The fix was
`// 8388608` for `>> 23` -- exactly equivalent after `abs()`, and on-device.

**Run this for every operator with a Python driver, before submitting.** A CPU fallback is invisible to
every correctness gate and presents only as the anti-cheat zero, which a correctness-only gate reads as
finished.

**Two related measurements worth keeping:**
- **`torch.exp2` is NOT bit-exact over the integers 0..123.** Where exactness is load-bearing, assemble
  `2^e` by writing the fp32 biased-exponent field instead. `ldexp` *is* exact but profiles as
  `Cast + PowTensorTensor + Mul`.
- **Driver overhead is usually LAUNCH-bound, not bandwidth-bound.** `torch.aminmax` emits **two**
  kernels (`ReduceMin` *and* `ReduceMax`), so two calls are **four launches and four full passes** --
  ~43 us on a case whose entire kernel is 39 us. When pricing a host-side pre-pass, count **launches**,
  not bytes: a reduction over a tiny tensor still costs the ~21 us aclnn launch floor.

## STEP 2h -- EVERY REMOTE RUNNER IS AN **A3**; EVERY LOCAL MEASUREMENT IS AN **A2**

`list_runners`: all eight online `910c` runners are **A3**. **There is no A2 remote runner and there
never was.** So every remote score this campaign has received is an A3 score, while every local
"20/20" -- including one operator's *fourteen* arms -- is an **A2** measurement.

```
Ascend910B2.ini    (A2, local)   ai_core 24   cube 24   vector 48
Ascend910_9362.ini (A3, RUNNER)  ai_core 20   cube 20   vector 40
```

Both under `/usr/local/Ascend/cann-*/aarch64-linux/data/platform_config/`, **readable with no device
at all.**

**So "it passes locally" is a statement about a 24/48-core part, and the thing you submit to has
20/40.** For most operators that transfers. It stops the moment anything depends on the core count --
and the sharpest case is an all-core **`SYNCALL`**, where launching more blocks than the part has cube
cores is a **guaranteed deadlock** (`retCode=0x25 [aicore timeout]`, 507014), not a slowdown. One
operator shipped `block_dim = 24` into exactly that and hung **two shared A3 dies**; 13 of its 20 cases
launch above 20, and `bd = 8, 10` passed while `bd = 24` faulted on two different dies.

**Mandatory Preflight for any kernel containing `SYNCALL`:** read both `.ini` files, take the
**minimum** cube count across the parts you may run on, and confirm every reachable `block_dim` is
`<= that`. Costs nothing, needs no hardware.

**And state which part a local result came from.** An A2 20/20 is not evidence about the runner; it is
evidence that the kernel is correct on 24 cores.

## STEP 2b -- A `MANUAL` VERDICT IS NOT A PASS

The audit tool returns `MANUAL` for every class it cannot decide by grep. **A carried `MANUAL` is an
open question, not a clean result**, and they accumulate silently because a run with only MANUALs
"exits 0".

**Measured cost of ignoring one:** the `C69/C72` item reads *"degenerate + non-finite operands
(Inf/NaN/subnormal/FLT_MAX on every operand)"*. It had been carried, un-discharged, for the entire
campaign on an operator whose hidden run then failed **12 cases** to exactly that class. **The tool
named the bug and the item was carried, not closed.**

**Rule: before declaring an operator ready, every `MANUAL` is either discharged by a device probe or
explicitly deferred with a written reason.** "It only reported MANUAL" is not a clearance. Write the
probe to disk so the next session inherits the discharge rather than the question.

**And a `[clean]` can be untrustworthy too, in two ways seen:**
- **path-dependence** -- the same file at the same SHA-1 reported `FAIL` under one path and `clean`
  under another. A `[clean]` is evidence about the run, not about the file.
- **a check whose trigger is narrower than its title.** One read `[clean]` on a real site because its
  pattern demanded fp32 plus a literal wide tile, while the defect was fp16 with a **runtime** stride.
  When a check clears something you have reason to doubt, read the check's pattern, not its name.

## STEP 2c -- DECIDE WHOSE DEFECT IT IS, AND WHETHER THE REFERENCE IS EVEN SPECIFIABLE

Some cases cannot be passed by **any** implementation. Establish this by measurement, then stop.

**The test: run the reference many times on the same distribution and see whether IT is stable.**

Worked example. Two cases failed a **bit-exact** comparison on one element, `0x0000` vs `0x8000` --
a `+0.0`/`-0.0` sign. `torch.unique` sorts with `std::sort` (introsort, **unstable**) and keeps the head
of each equal run; `+0.0` and `-0.0` compare **equal**, so which bit pattern survives is a partition-order
artefact. Over 60 seeds the reference kept `+0.0` **27/60** and `-0.0` **33/60**, matching
first-occurrence 31/60 and last-occurrence 30/60. **A coin flip. There is no rule to implement** -- and
`desc.md` asserted both a bit-exact comparison *and* a first-occurrence rule, and was factually wrong
about the second.

**So `desc.md` is not authoritative about reference behaviour. Measure the reference.** Other members
of this class already recorded: the benchmark's golden OOMing at 16 GiB, its golden crashing before our
kernel runs, and an `inverse` assignment that is order-stable at `n=6` and not at `n=16129`.

**When a case is unspecifiable: record it, report it upstream, and do not count it against the
kernel.** It also caps the pass count permanently, which decides STEP 3.

**Corollary -- re-derive a recorded plan before trusting it.** A stored fix plan for the above
("use the min-origidx of the zero group") came from a probe with 8 copies of each value and is false at
scale: min-origidx would not have fixed it either. A plan validated on a toy shape is a hypothesis.

---

### One harness trap that silently produces a meaningless score

**`--skip-install` REQUIRES `--source-dir`.** Without it the CLI silently evaluates **all 53
operators**, and it scores whatever `cann_bench` happens to be **importable** -- so each arm needs its
own `pip --target` on `PYTHONPATH` first. A first attempt this way scored an unrelated package and
reported `LSTM`, 0.00, in five seconds. If a score arrives implausibly fast or names the wrong
operator, this is why.

### STEP 2d -- WIDENING A DECLARED SURFACE CREATES NEW PRECISION EXPOSURE

**A shape fix and an accuracy fix are not independent.** Opening a cap makes larger shapes *reachable*,
and larger shapes are worse conditioned -- so closing a `compile_runtime_error` class can convert it
into a `precision_mismatch` class rather than into a pass.

Worked example. An operator's `hiddenSize` cap was widened 256 -> 1024, taking the declared-surface
ACCEPT rate from **63.2% to 100%** with two mechanical hidden-failure classes provably gone. But its
sharpest residual is now a *precision* one, and it was already visible in the evaluator's own per-output
JSON: **one visible case passes with `MARE` 1050x the limit and `MERE` 1.8x**, and its pass rests
**entirely** on the band relaxation -- `normal ours/cpu = 2984/22149`, i.e. on the fp32 CPU reference
*also* being dirty there. Accumulation error grows with the axis that was just opened, so the largest
hidden cases are now reachable **and** are exactly where that relaxation is thinnest. One data
distribution away from `N/0` and the strict branch fires.

**So when you widen a cap, re-ask the accuracy question on the shapes you just made reachable.** Two
checks that cost nothing:
- **Scan the passing cases' own counters for a thin margin.** A case that passes only because
  `cpu_error_count` is large is a case that fails when the reference happens to be clean. Grep the
  per-output JSON for `ours/cpu` ratios, not just for failures.
- **Probe at the new maximum, not at the old one.** "Accepted" is not "correct": verify against a
  CPU-fp64 golden at the newly opened extremes, since a widened path can be accepted and silently wrong.

**And distinguish the two kinds of cap while you are there** -- they need different fixes:

| the cap is | symptom when exceeded | fix |
|---|---|---|
| **fitted** (a staging tile over a chunkable extent) | a smaller tile and a correct answer | chunk it -- pure offsets |
| **architectural as a tile width** (alignment x extent exceeds UB) | no wider tile is possible at all | **block the OUTPUT**, in-kernel if the contraction spans the axis |

Measured instance of the second: `kKC >= 8` for 32-byte alignment meant that at `H=1024` two tiles each
needed 98304 B -- **196608 against a 188416 UB**. No widening exists. Blocking the output rows was
exact because the gate algebra is elementwise in that index, and it had to be **in-kernel**: the
contraction spans all of `H` and the state is produced by the same recurrence, so no cross-launch
decomposition exists.

## STEP 2f-LENIENT -- `compare_tensors(threshold=...)` IS INERT, AND THIS ONE BLESSES A WRONG KERNEL

Every other trap in this file makes a local gate **stricter** than the evaluator, so it rejects a
correct kernel -- annoying, but it fails safe. **This one fails the other way.**

`compare_tensors(out, gold, threshold=0.0)` **does not select a bit-exact path.** `compare.py:822-824`
recomputes the per-output threshold from **`custom_thresholds` only**; the `threshold=` argument sets
nothing but the top-level result field. With `custom_thresholds` empty, an fp16 comparison falls back
to its ~1e-3 MERE/MARE default -- where `+0.0` vs `-0.0` has **zero** relative error. So a candidate
that is provably wrong **PASSES**.

Measured on golden `[-1.0, -0.0, 1.0]`:

| gate | verdict |
|---|---|
| `threshold=0.0` | **PASS** -- wrong |
| `custom_thresholds={"float16": 0}` | **FAIL** -- right |

**How to apply:** when you need an exactness gate, pass **`custom_thresholds`**, never `threshold`.
And give every gate you build a **positive AND a negative control at import time** -- a known-bad
candidate must FAIL and a known-good one must PASS, or the gate is unvalidated. A gate that has only
ever seen good inputs cannot distinguish "my kernel is right" from "my gate is inert".

## STEP 2i -- THE FLOOR PROBE: RUN IT BEFORE TOUCHING NUMERICS

### A FLOOR THAT RETURNS EXACTLY `0.000` IS A TAUTOLOGY, NOT A PASS

**Check the floor's VALUE, not just its verdict.** `compare.py:383` does
`golden_truncated = golden.to(output.dtype)`, so when the operator's `golden.py` casts
(e.g. `lstm_layer.float()`) and there is **no `_bench` hook**, then
`golden_truncated == native == your floor arm` **bit for bit**, and the floor reports
`MERE = MARE = 0.000` **by construction** -- on every row, at every value range, forever.

**A genuine floor reads something like `1e-7`, never `0.000`.** Exact zeros across a whole ladder are
the tell that you are comparing the reference against itself.

Worked example, which retired a real item for a week on a wrong basis: lstm's floor returned
`MARE = 0` on all 32 rows out to `|w| = 50`, with a firing all-NaN negative control. Read naively
that says "every case is winnable". It actually says nothing at all. And the **opposite** earlier
reading was equally wrong -- a 2026-09-30 record claimed *"above `w` about +/-2 the FLOOR ITSELF
fails, so the operator has no specifiable fp32 answer there"*, which closed the item as impossible.
Measured, the floor passes at `|w|` = 2, 3, 5, 10, 50.

**The two framings are not equivalent and the difference decides whether work is possible:**

| reading | implication |
|---|---|
| "no target exists" | the item is closed forever; stop |
| **"the target is the REFERENCE'S BITS"** | the only route is matching its operation order -- a definite, pricable statement |

So when the golden is lossy, restate the verdict as: **winnable only by bit-matching the reference's
operation order; never by being more accurate.** See
[[use-the-operators-own-golden-convention]] for how the same confusion inverted a 7-arm study.

### A PERTURBATION BELOW THE DTYPE'S ULP IS A NO-OP, AND THE PROBE REPORTS `0` FOR IT

The floor probe perturbs an arm and asks whether the perturbed arm still passes. **If the
perturbation is smaller than one ulp of the tensor it is applied to, it rounds away and the
"perturbed" arm is the SAME computation** -- so the probe reports `mere = 0` and a confident
`RECOVERABLE`.

Measured: a floor arm multiplied the **case-dtype** tensor by `(1 + 2^-23)`. That is one fp32 ulp,
but at fp16 (ulp `2^-10`) and bf16 (ulp `2^-7`) it rounds back to the identical value. Three
`RECOVERABLE` verdicts came out of that arm and **all three dissolved** when the probe was re-run at
a precision where the perturbation survives.

**The guard is one assert, not a convention:** after perturbing, check the perturbed tensor actually
**differs** from the original (`(a != b).any()`), and record it as a column in the result table. A
probe that cannot show its own perturbation landed is not evidence. This is the fourth distinct way a
gate or probe reads clean on a wrong input, after
[[threshold-arg-is-inert-use-custom-thresholds]],
[[negative-control-must-not-sit-on-the-boundary]] and
[[negative-control-must-perturb-the-whole-tensor]].

### THE RIGHT FLOOR FOR A CHAOTIC OPERATOR: PERTURB THE REFERENCE'S **INPUT** BY ONE ULP

Both standard floors fail on a chaotic, lossy-golden operator. Substituting an **exact** arm is
out-of-family (next section) and substituting the **reference rounded to case dtype** is a tautology
that returns `0.000`. There is a third construction with neither defect, and it needs **no kernel, no
device, no oracle**:

> Run the operator's own `golden.py` **against itself**, with one input nudged by a single ulp.

That measures the quantity that actually decides the case -- **how much the reference's own answer
moves under a perturbation no implementation can avoid** -- and the comparison is in-family by
construction, because both arms *are* the reference.

Worked example (lstm, `S100 B64 In256 H256` fp32, `a` = the recurrent weight half-range):

| `a` | golden vs golden-with-input-nudged-1-ulp | verdict |
|---|---|---|
| 0.1 | MERE 2.2e-06 | PASS |
| 0.3 | MERE 3.1e-06 | PASS |
| 0.5 | MARE 5.09 | **FAIL** |
| 0.85 | 2.25 / 21.5% mismatch | **FAIL** |
| 1.0 | **8.94 / 37.04% mismatch** | **FAIL** (our kernel there: 8.836 / 37.6%) |

The knee is at `a` ~ 0.3-0.7 and **it moves down with H and S**: at `H1024, S200-512, a=0.5` the
one-ulp floor is already MERE 5.1-16.1 at 71-81% mismatch -- the exact signature of four hidden cases
-- with our kernel inside 1% of it.

**A case whose reported error sits at this floor is not winnable by being more accurate, and the
measurement costs nothing.** Run it before any numerics work, and before pricing an accuracy fix:
"ours is 1% from the one-ulp floor" retires a fix that "ours is 615x the fp32 epsilon" would have
funded.

### THE REFERENCE IMPLEMENTATION CAN CHANGE WITH AN ATTRIBUTE -- CHECK WHICH GOLDEN A CASE GETS

"Bit-match the reference's operation order" presumes there is **one** reference. There may not be.
PyTorch emits, at `aten/src/ATen/native/RNN.cpp:1473`:

```
UserWarning: LSTM with projections is not supported with oneDNN. Using default implementation.
```

So `proj_size > 0` cases are scored against the **plain ATen per-timestep loop** and `proj_size == 0`
cases against **oneDNN's fused RNN** -- two different operation orders, in one operator, selected by
an attribute. A matching program aimed at the wrong one does nothing.

**Two consequences, both of which bit us:**

- A conclusion established on one branch does not transfer. `|y| <= 2` holds for `proj_size == 0`
  (`y = o * tanh(c)`), so `max_diff_y` near 2.000 there is a saturated sign disagreement. With
  projection, `y = W_hr @ (o * tanh(c))` is **unbounded** and the same cases read `max_diff_y`
  **13.35 / 13.47**. The out-of-range-emission family is cleared only on the non-projection branch.
- A reference quirk is branch-local. oneDNN suppressing a NaN that IEEE produces
  ([[onednn-suppresses-nan-that-ieee-produces]]) cannot explain a `proj_size > 0` case, because
  oneDNN never ran for it.

**Grep the probe logs for dispatch warnings.** They are free, they appear on stderr where nobody
reads them, and they name which reference the case was actually scored against.

### THE CONTROL MUST BE IN-FAMILY: fp64 IS NOT A CONTROL FOR AN fp32 KERNEL

**"The control passes and ours fails" is evidence of our defect ONLY when the control shares our
arithmetic family.** An exact fp64 arm injects no per-step rounding, so in any expansive or recurrent
regime it passes where a correct fp32 kernel fails -- **by construction**. Using it as the control
measures the precision gap, which the regime then amplifies for free.

Worked example, which manufactured a cap bug that does not exist: three lstm configurations
(`H1024` at `S=200/512/2048`) showed `control=PASS, ours=FAIL` with H128/H256/H512 passing and H2048
failing for everyone -- a textbook ours-only band bracketed by passing controls, and it was promoted
as the strongest signal on the operator. The control was the fp64 arm. Adding a CPU **fp32** arm
dissolved it: at `H1024_S200` ours is **4.90e-6** against the in-family arm's **4.92e-6**, so **ours
is MORE accurate and still fails** -- a single-element MARE coin flip with MERE four orders *inside*
the gate. An H ladder at `S=4` (no amplification) over 128..2048 including 255/256/257, 511/512/513
and 1023/1024/1025 returns ratio 0.9-1.4 at every H; a static cap would bite there and does not.

**So name the control's dtype and operation order before believing any ours-only failure.** Keep the
fp64 arm only as the third point for the triangle check in STEP 4, never as the pass/fail control.
This is the same error as using the wrong golden convention, one level up: not the golden's
convention but the **control's**.

**And in an EXPANSIVE recurrence, accuracy is actively counterproductive** -- measured per timestep on
lstm: at `t=0` the error is `6.5e-07` (fp32 epsilon, i.e. the per-step arithmetic is correct), and
past `|w| ~ 1.4` the Jacobian's spectral radius exceeds 1 and amplifies *any* rounding difference
geometrically -- growth `4.7x` at `|w|=0.5`, `2.2e5` at `1.4`, `5.1e9` at `2.0`. Being *more* accurate
diverges from a lossy reference exactly as fast as being less accurate. That is the mechanism behind
an exact-GEMM arm scoring **0 fail->pass / 19 pass->fail**.


The cheapest control for **"is this case even winnable"**, and it costs no card and no credit.

Construct the candidate as the **exact fp64 reference rounded to the case dtype** and push it
through cann-bench's own `compare_tensors`. That is the best any implementation can return. Then:

| floor | reading |
|---|---|
| floor **FAILS** | no kernel -- ours, the vendor's, anyone's -- can pass. The failure is the comparator against the case's conditioning, **not our defect**. Record it as unwinnable and stop. |
| floor **PASSES**, we fail | a real accuracy deficit of ours, and the floor's margin tells you how much room you have to find. |

**Why this is not optional.** Without it, a failing hard case is ambiguous between our numerics and
the case's conditioning, and the default reading is "fix our kernel". On lstm it retired a residual
that had been carried as an open defect for a day -- "bf16 with `w` +-10 is 0.884 relative error" --
by showing the floor there is `mare` **1.9e+02**, i.e. the exact fp64 answer misses the gate by
380x. It also bounded the whole value-range axis in one run: 95/95 configurations at or below the
visible set's value scale pass on all three dtypes, and **beyond `w` ~ +-2 the floor itself starts
failing**. That converts "we might be exposed on value range" into a named, measured boundary.

**It also splits `unspecifiable` from `unwinnable-but-specifiable`,** which STEP 2c asks you to
decide and which is otherwise a judgement call. Run the floor against **both** of the benchmark's
own references: if its fp32 golden and its fp64 oracle **disagree** on the case, the reference is
unspecifiable and no target exists (lstm: `+-inf` in weights -- fp32 golden returns 0 NaN where the
fp64 oracle returns 352 of 384). If they agree and the floor passes, the case is a legitimate
target however far away it looks.

**CAVEAT, and it reverses the conclusion on some operators: when the ORACLE and the NATIVE
reference are the SAME lossy run, the fp64 floor is NOT the ceiling.** An operator with a
`get_input` hook makes `evaluator.py:315-317` discard the fp64 upcast, so the oracle and the
same-precision reference are one fp32 CPU run (`cpu_diff_max == 0.000e+00` on every row). The target
is then the reference's **bits**, not the true value, and **bit-matching it BEATS the floor**.
Measured on roi_align: the exact fp64 answer rounded to dtype **FAILS 16 of 96** fp32 draws, while
golden's own association is **bit-exact 96/96** -- and the same on fp16, 36/36. So a failing floor
there means "unwinnable *by being more accurate*", which is a different claim from "unwinnable", and
reading it the first way **retires real work**. Check for a `get_input` hook before using a floor
failure to close a case.

**Pair it with the value-range question**, because `desc.md` almost always declares no value range
while every visible case sits near unit scale -- so the floor is what tells you how far outside that
a correct kernel could survive. One structural prior worth knowing before you panic about extremes:
across 53 operators / 1060 cases the `[-100,100]` / `[-65504,65504]` extremes battery is applied to
35 elementwise, normalisation and indexing operators and to **none** of the seven float level4
FusedComposite operators, all of which cap at 1 or 2.

## STEP 2j -- ENUMERATE HOST OPS BY WHAT **DISPATCHES**, NOT BY WHAT LOOKS LIKE A CALL

The `561xxx` family is "the runner has no binary for this op". Three confirmed members:
`aclnnInplaceZero` and `aclnnCast` (**561103**), and **`Slice` (561000)**. **Every one is invisible
locally** -- our image has the binary, so a green local run is no evidence at all.

An enumeration that lists things which *look like* kernel calls will miss them. One agent enumerated
`gather`, `sort`, `scatter_`, `clone` and `cat`, built a three-tier fallback for all five, and still
shipped a build that failed 3 cases -- because the op that broke was **`narrow`**, which reads as a
view.

**The instrument, measured:**

| probe | launches |
|---|---|
| `torch.empty` | **NONE** -- allocator only |
| `.contiguous()` on an already-contiguous tensor | **NONE** |
| same-dtype same-device `.to()` | **NONE** |
| `reshape` on a contiguous tensor | **NONE** |
| `.clone()` | **NONE** -- async D2D memcpy, not an op |
| **`narrow(0, ...).contiguous()`** | **NONE** -- a dim-0 narrow of a contiguous tensor is still contiguous |
| **`narrow(d != 0, ...).contiguous()`** | **`aclnnInplaceCopy_SliceAiCore_Slice`** |
| `transpose().contiguous()` | `aclnnInplaceCopy_TransposeAiCore_Transpose` |
| `torch.gather` | `aclnnGather_GatherElementsV2_...` + `aclnnGather_CastAiCore_Cast` |
| `torch.sort` (plain and stable) | **`aclnnSort_SortAiCpu_Sort`** (AI **CPU**) + `aclnnSort_CastAiCore_Cast` |
| `scatter_` | `aclnnInplaceScatterValue_...` |
| `torch.cat` | `aclnnCat_ConcatD_ConcatD` |

**How to run it:** one op per **subprocess** under `torch_npu.profiler`, then read the **`Name`**
column (column 2) of `ASCEND_PROFILER_OUTPUT/kernel_details.csv` -- it carries the full
`aclnnEntry_AiCoreOp_OpType` triple, the same vocabulary the remote error uses (`opType: 5, Slice`).
**Trap:** some profiler builds emit **no** `op_statistic*.csv`, and reading only that file reports
"no ops" for calls that did dispatch. Pair it with an **AST walk** for slicing nodes and tensor
subscripts -- in a clean driver every `Subscript` indexes a Python list or dict, never a tensor.

A rescued census lives at `skillyard-cannbench/reports/dispatch_census_2026_10_05/`.

**And a FAILED REMOTE JOB IS ITSELF A PROBE OF THE REMOTE OP SET:** read what it got *through*, not
only what it died on. One job dispatched `aclnnGather_TransposeAiCore_Transpose` and got past it
before dying at `aclnnInplaceCopy`, which proves `Gather`, `Sort`, its `Cast` **and** `Transpose` are
all present remotely.

**Design rule: allocate the exact output shape up front and fill it; never produce a larger tensor
and cut it down.** And before swapping a missing op for a host round-trip, **count the memcpys** --
see the campaign loop's budget-of-5 rule. A fix that trades a `precision_mismatch` for a
`cpu_fallback_detected` loses the whole operator instead of a few cases.

## STEP 3 -- DECIDE. THE GATE IS THE PASS COUNT.

**Send only if the hidden can PASS -- a projected 80/80, not a projected score.**

```
1. projected PASS COUNT on the case set you would run       <- the gate
2. every case still open, by id, with its cause
3. for each: closable by kernel work, or not
4. only then:  full_pass  vs  the CURRENTLY POSTED entry
```

- **Any case open => do not spend the credit.** A `partial_pass` posts **nothing**, so the delta is
  uncollectable. Four operators were each proposed for a credit on deltas of +6 to +10 while every
  ceiling had already been computed as 78/80.
- **A standard full pass does not imply a hidden one.** The hidden set is 4x and probes outside
  `desc.md`. An unwalked declared surface means the hidden count is **unknown**, not good.
- **`|delta| < ~1` is zero.** Calibrated against six realised outcomes: median absolute projection
  error 0.44, worst 4.87.
- **Price against the LIVE entry.** Our own later run may have superseded the baseline a recorded
  delta was measured against (a `+4.76` became `+1.21` that way).
- **A code change costs 2 credits** (fresh standard, then hidden). Check whether the **submitted zip**
  contains the fix -- a terminal standard pass on an older zip is not the same submission.
- **Fixing is NOT gated. Only submitting is.** Repair silent wrong answers whatever the board pays.

**Record the verdict.** An operator decided as unspendable gets that written down with its evidence,
so the next session does not re-derive an uncollectable delta.

---

## STEP 4 -- DEVICE AND HYGIENE PROTOCOL

- **A `pgrep -f` wait loop can match ITS OWN command line and deadlock.** One agent lost a card to
  `pgrep -f 'kernel_eval.cli eval --source-dir'` while it spun; another session claimed the card in the
  gap. Anchor the pattern on something your own process cannot contain, or filter your own PID and PPID.
- **Claim a card**: "No process in device" **twice, 60 s apart**, *and* `/proc/<pid>/environ`
  `ASCEND_RT_VISIBLE_DEVICES` checked on every live `kernel_eval` process. **Hand off directly from
  winning to starting** -- a card was lost twice *inside* its own confirmation window; the gap was the
  bug, not the gate. Sibling PIDs look like yours unless you log cmdlines.
- **Never device 0. Never card 5.** Verify by **physical** card. Discard a contended timing arm rather
  than explaining it away.
- **Do not SIGTERM a running eval** to free a card -- it leaks a device allocation (933 MB held by a
  PID absent from `/proc` for 25+ minutes).
- **NEVER SIGKILL A RUNNING KERNEL. It leaves the card SILENTLY RETURNING WRONG ANSWERS.** An agent
  killed a slow probe on card 1; for the next **~9 experiments** that card returned answers **~300x
  worse than correct -- consistently and reproducibly**, which is precisely why the results were
  believed. `npu-smi` reported it free and Health OK throughout. A byte-identical 47-config plan read
  **47/47 dirty**, then **0/47** on a clean card. Deliberate contention does **not** reproduce it, so
  it is the aborted-kernel state, not contention, and it persists **across processes** for minutes.
  Four separate "findings" came out of that window and all were false: a universal 257-793x error
  excess, a `block_dim<=3` clean / `>=4` dirty 500x effect, period-3 non-determinism over identical
  calls, and a mean error 615x the fp32 floor. **Let a probe finish or time out; never kill it.** This
  generalises the known bad-card problem from one card to **any** card -- the silent wrong-answer
  state is reachable by your own actions and the health check cannot see it.
- **PUT A TRIANGLE-INEQUALITY CHECK IN FRONT OF EVERY ACCURACY TABLE.** Print all three of
  `|ours-golden|`, `|ours-exact|`, `|exact-golden|`. If `|ours-golden|` exceeds the sum of the other
  two, that is **arithmetically impossible**: the card is poisoned and every number in the run is
  void. This is what finally caught the case above (`7.9e-5` against `8.0e-8` and `1.2e-7`) after
  nine experiments had been accepted. It costs one extra CPU reference per row and it is the only
  cheap detector for a silently-wrong card.
- **If your card faults, STOP and report.** Never a host-wide `pkill`; own PIDs by number.
- **`kernel_eval.cli eval --source-dir <arm> --skip-install` is SAFE** with siblings running. Only
  `run_evaluation.sh --source-dir` triggers `uninstall_packages`.
- **Package LAST**: `clean_submission_tree.sh`, **no rebuild after**, audited with `find` not
  `git status` -- `submit_kernel` zips the directory, so `.gitignore` protects nothing. Off-allowlist
  and extensionless both 0; the 9 `third_party` `.inl`/`.csv` stay deleted.
- **Do not patch the audit tool while agents run against it** -- it invalidated two before/after pairs
  in one session. Queue the patch.
- **Never `json.dump` straight to a record file** -- serialise to a string, validate, then
  `os.replace`. A partial write truncated an 890 KB record mid-key.

---

## STEP 5 -- REPORT SHAPE

1. **Real defect count** and the first failing case id, with the cascade separated out.
2. **Signature -> class** for each distinct defect, by the dispatch table above.
3. **The projected pass count**, then `full_pass` vs the live entry.
4. **Every number with its control**, and every unmeasured thing labelled UNMEASURED (remote cost
   always is: the local->remote gap has run 1.35 to 7.5 points).
5. **The verdict**: close it, fix-but-never-send, or permanently unspendable -- with the evidence.
