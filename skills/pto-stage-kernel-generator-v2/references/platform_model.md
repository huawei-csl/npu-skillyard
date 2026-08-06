# Ascend NPU Platform Model (A2/A3)

PTO-ISA hardware model for NPU architecture versions 220x (Atlas A2 training /
Atlas A2 inference). Use this to reason about what tile data-movement and
compute patterns are architecturally legal on A2/A3.

For the A5 (351x / Atlas 350) platform, see `platform_model_a5.md`.

Section IDs use `PLAT-§` prefix for cross-referencing from SKILL.md.

---

## PLAT-§Hierarchy: Memory Hierarchy

```
GM (Global Memory / HBM)  — off-chip DRAM, several GB, all cores share it
  │
  ├── MTE2 (DMA load)  ──→  UB (Unified Buffer / Vec SRAM)  — 192 KB
  │                              │
  │                              ├── Vec engine (SIMD ALU) — reads/writes UB only
  │                              │
  │                              └── MTE3 (DMA store) ──→  GM
  │
  └── MTE1 (DMA)  ──→  L1 (Cube buffer)  — 512 KB per AIC core
                               │
                               ├── L0A (left operand)  — 64 KB, TEXTRACT source
                               ├── L0B (right operand) — 64 KB, TEXTRACT source
                               └── L0C (accumulator)   — 128 KB
                                    │
                                    └── TSTORE ──→  GM
```

**Critical rule**: Cube (L0A/L0B/L0C) and Vec (UB) run on SEPARATE physical cores
(AIC vs AIV). They communicate ONLY through GM + cross-core FFTS flags.
Cube output in L0C cannot be read by Vec without a round-trip through GM.

---

## PLAT-§Topology: Core Topology

```
One AI Core cluster (A2/A3):
  ┌────────────────────────────────────────────┐
  │  AIC (Cube core)                            │
  │    - TMATMUL, TEXTRACT, TSTORE(L0C→GM)     │
  │    - L1 512KB, L0A 64KB, L0B 64KB, L0C 128KB │
  │    - Communicates with AIV via GM + FFTS    │
  ├────────────────────────────────────────────┤
  │  AIV-0 (Vec sub-block 0, vid=0)            │
  │    - TMOV, TADD, TMUL, TEXP, TCVT, TLOAD   │
  │    - UB 192KB PRIVATE to this sub-block     │
  ├────────────────────────────────────────────┤
  │  AIV-1 (Vec sub-block 1, vid=1)            │
  │    - Same capabilities, its OWN 192KB UB    │
  └────────────────────────────────────────────┘
```

Cross-core sync via FFTS (Fast Fine-grained Task Synchronization):
- `set_cross_core_flag<PIPE>(flag_id)` — signal from one core to another
- `wait_flag_dev(flag_id)` — block until flag is set
- A2/A3: Cube broadcasts to both Vec sub-blocks; Vec reduces to Cube

---

## PLAT-§Pipelines: Pipelines (within one AIV core)

```
PIPE_MTE2  — DMA load engine  (GM → UB), async, 128 B/cycle for Vec tiles
PIPE_V     — Vector SIMD ALU  (UB → compute → UB)
PIPE_MTE3  — DMA store engine (UB → GM), async
PIPE_S     — Scalar processor (address calc, control flow)
PIPE_MTE1  — DMA for Cube     (GM → L1, L1 → L0A/L0B), on AIC core
```

**Pipe concurrency**: MTE2, Vec, and MTE3 can all execute in parallel on the SAME AIV
core. This is the basis for double-buffering and pipeline overlap.

---

## PLAT-§Movement: Legal Data Movement Paths

| Operation | Direction | Engine | Notes |
|-----------|-----------|--------|-------|
| `TLOAD` (Vec) | GM → UB | MTE2 | Vec tile only; async; zero-pads tails |
| `TLOAD` (Mat) | GM → L1 | MTE1/MTE2 | Mat tile load into Cube buffer |
| `TSTORE` (Vec) | UB → GM | MTE3 | Vec tile only; async |
| `TSTORE` (Acc) | L0C → GM | MTE3 | Cube result store |
| `TMOV` (V→V) | UB → UB | Vec | Same-dtype copy within UB |
| `TMOV` (M→L/R) | L1 → L0A/L0B | MTE1 | Mat→Left/Right/Bias/Scaling |
| `TMOV` (A→M) | L0C → Mat | Fixpipe | Acc→Mat with optional quant/relu |
| `TCVT` | UB → UB | Vec | Dtype conversion in-place in UB |
| `TADD/TMUL/TSUB/TEXP/TRELU/TLOG` | UB → UB | Vec | Element-wise math in UB |
| `TROWEXPAND/TCOLEXPAND` | UB → UB | Vec | Broadcast row/col values across a tile |
| `TEXTRACT` | L1 → L0A/L0B | MTE1 | Cube operand load |
| `TMATMUL` | L0A,L0B → L0C | Cube | Matrix multiply, fp16→fp32 accumulate |
| `TMATMUL_ACC` | L0A,L0B,L0C → L0C | Cube | Fused accumulate |
| `TGEMV` | L1 → Acc | Cube | Matrix-vector multiply |
| `TRESHAPE` | L1 → L1 | — | View reinterpretation (no data copy) |
| `set_flag/wait_flag` | — | Any | Intra-core pipe sync (same AIV) |
| `set_cross_core_flag/wait_flag_dev` | — | FFTS | Cross-core sync (AIC ↔ AIV) |
| `pipe_barrier(PIPE_V)` | — | Vec | Stall Vec until all pending Vec ops complete |
| `pipe_barrier(PIPE_ALL)` | — | All | Stall ALL pipes; required after TLOAD/TSTORE |

---

## PLAT-§Illegal: Explicitly ILLEGAL Operations

| Attempt | Why illegal |
|---------|-------------|
| Vec reading Cube L0C output directly | Different cores; must go through GM |
| Cube reading Vec UB output directly | Different cores; must go through GM |
| TLOAD from UB to GM | Wrong direction; TLOAD is GM→UB only |
| TSTORE from GM to UB | Wrong direction; TSTORE is UB→GM only |
| TMATMUL on Vec tiles | TMATMUL requires L1/L0A/L0B operands |
| TADD/TMUL on L0C tiles | Vec ops work on UB only |
| Relaxing `pipe_barrier(PIPE_ALL)` around TLOAD/TSTORE | Causes data corruption; required for MTE↔V ordering |
| Using auto mode with `-DMEMORY_BASE` | Crashes at runtime; manual mode is mandatory |
| Exceeding UB 192KB peak | Silent corruption or device trap |
| Exceeding L0C 128KB peak | Silent corruption or device trap |
| Reusing event ID while previous signal/wait is in flight | Race condition, data corruption |
| `const` / `constexpr` qualifying an `event_t` variable | Compile error: "the 3rd parameter must be a type 'event_t'" |

**Removed from this table: "scalar indexing of a `__gm__` pointer".** It used to
say "NPU Alarm crash requiring hardware reset". **That is false** — probed on
A2/dav-c220 (`isa_probes/probe_gmscalar.cpp`): scalar read and scalar write, on
both the Vec and the Cube core, all return exact values with the device healthy
before and after. It is legal and is the supported way to read a runtime scalar;
see C1 for when to use it and for the `volatile` + `dcci` requirements. The false
rule blocked two independent `grouped_matmul` runs, because a runtime-determined
tile schedule cannot be built without it.

---

## PLAT-§Events: Event IDs and Synchronization

- **8 event IDs per core**: EVENT_ID0 through EVENT_ID7
- Each `set_flag/wait_flag` pair consumes one ID for one dependency edge
- Same ID can be reused across iterations if the previous use has fully retired
- Double buffering typically uses IDs 0,1 for even/odd slots
- Cross-core flags use a separate namespace (flag_id 0-15 on A2/A3)

**Standard TLOAD→compute→TSTORE pattern:**
```cpp
TLOAD(tile, gm_src);                          // MTE2 starts loading
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);       // MTE2 signals: data ready
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);      // Vec waits for data
TADD(dst, tile, bias);                        // Vec computes
pipe_barrier(PIPE_V);                         // Vec completes
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);       // Vec signals: result ready
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);      // MTE3 waits for result
TSTORE(gm_dst, dst);                          // MTE3 stores
pipe_barrier(PIPE_ALL);                       // Full barrier before buffer reuse
```

---

## PLAT-§UB: UB Budget Math

> **The 192 KB is PER AIV SUB-BLOCK, not per core. Do NOT partition UB between the two
> sub-blocks -- that halves your budget for nothing.**
>
> This was an open contradiction in this project for a long time, and several kernels
> resolved it the expensive way: `ub = vid * UB_VID_STRIDE` with
> `static_assert(2 * UB_VID_STRIDE <= 184*1024)`, i.e. ~92 KB each. Those kernels are
> CORRECT -- partitioning is merely wasteful, not wrong -- but they were built on half
> the buffer that exists. In one case it forced rows-per-item down from 4 to 2 to make
> room for a prefetch slot, costing 1.120x -> 1.327x before the prefetch repaid it.
>
> **Probed directly** (`skillyard-runs/isa_probes/probe_ubpriv.cpp`): both sub-blocks
> `TASSIGN` a tile at the SAME UB address and hammer it concurrently for up to 20,000
> iterations, then each stores what it sees. If the buffer were shared, both would read
> the last writer's value.
>
> | UB address | result |
> |---|---|
> | 0x0 | sub0 reads 1.0, sub1 reads 2.0 -- **PRIVATE** |
> | 0x10000 (64 KB) | **PRIVATE** |
> | 0x20000 (128 KB) | **PRIVATE** |
> | 0x2c000 (176 KB) | **PRIVATE** |
>
> Deterministic across repeats. 176 KB matters: it is far above the ~92 KB half-budget
> the partitioning scheme assumes, so a single sub-block demonstrably addresses the
> whole buffer.
>
> **Positive control, so "always private" cannot mean "the probe is blind":** the same
> hammer-then-read logic pointed at one GM slot -- unambiguously shared -- reports
> SHARED on every run (both sub-blocks read 2.0). The probe detects sharing when
> sharing exists.
>
> Practical consequence: budget the FULL 184 KB usable (192 KB minus the 8 KB PTO
> reserves at `TMP_UB_OFFSET`) for each sub-block independently. The one thing the two
> sub-blocks DO share is that `TMP_UB_OFFSET` library scratch -- which is why
> `SaturationMode::ON` must be passed explicitly on the fp16->int8 `TCVT`, since the
> default path routes through that shared region and can race.

```
UB capacity: 192 KB = 196,608 bytes PER AIV SUB-BLOCK (184 KB usable below TMP_UB_OFFSET)
Alignment: 32 bytes

Double-buffered fp16 tile: max ELEMENTS_PER_TILE <= 196608 / (4 * 2) = 24,576
Double-buffered fp32 tile: max ELEMENTS_PER_TILE <= 196608 / (4 * 4) = 12,288

UB budget check formula:
  sum(bytes_per_tile * live_buffers) <= 196608
```

---

## PLAT-§L2: L2 capacity, measured

Not in any datasheet we have; bracketed by experiment on 910B2. A kernel whose GM
working set grew with its sweep axis was L2-served up to **195 MB** of footprint and
HBM-bound by **387 MB**, so L2 capacity lies between those. The signature: implied
bandwidth of 1288 / 1527 / 1673 GB/s at the small sizes -- impossible against a measured
**811 GB/s** HBM streaming ceiling, so a cache was absorbing it -- collapsing to 695 GB/s
once the footprint no longer fit.

Practical rule: keep `block_dim * per_core_workspace` under ~190 MB. If a stage's
workspace scales with a swept dimension, that budget is what decides the largest size it
runs well at, and exceeding it costs far more than the parallelism you gain from more
cores (measured: 1.4x at one size, 1.23x at another, purely from lowering `block_dim`).

---

## PLAT-§L0C: L0C Budget Math

```
L0C capacity: 128 KB = 131,072 bytes
Alignment: 64 bytes

L0C budget guard:
  static_assert(L0C_BYTES <= 131072, "L0C overflow on A2/A3");
```

---

## PLAT-§Bandwidth: Bandwidth Model

| Path | Rate | Formula |
|------|------|---------|
| GM → UB (Vec tile) | 128 B/cycle | `ceil(bytes / 128)` |
| GM → UB (Mat tile) | 256 B/cycle | `ceil(bytes / 256)` |
| UB → UB (Vec tile) | 128 B/cycle | `ceil(bytes / 128)` |
| L1 → L0A/L0B | per-element | `ceil(bytes / 32)` |
| L0C → GM | burst | `ceil(bytes / 32)` |

These are per-cycle *issue* rates. They do NOT bound a kernel that streams a working set
larger than L2 -- for that, see the measured aggregate ceiling below.

### PLAT-§ReadCeiling: the aggregate GM read ceiling (A2, measured)

**A PTO `TLOAD` extracts ~920 GB/s from HBM-resident bytes, and nothing you can write in
PTO changes that.** Measured on 910B2 by streaming 469.8 MB of int8 through pure load-only
kernels (`grouped_matmul_swiglu_quant_v0170/reports/bandwidth_shortfall.md`):

| variable swept | result |
|---|---|
| ND→NZ conversion vs plain ND→ND copy | 856 → 899 GB/s (+5%) |
| contiguity, burst length (512 B … 4096 B) | flat, 880-899 |
| ring depth 2/3, and **no handshake at all** | flat, 916-919 |
| descriptor size 8 / 16 / 32 / 64 KB | flat, 897-919 |
| `block_dim` 24 / 32 / 40 / 48 | flat, 857-916 |
| interleaved vs contiguous per-lane partition | flat, 913-919 |
| GM→L1 (24 AIC) vs GM→UB (48 AIV) | 899 vs 919 |
| **both engine classes reading concurrently** | **911 -- bandwidth does NOT add** |

The ceiling is HBM-side, not an MTE2 or descriptor limit. The identical load path run over a
smaller footprint goes far faster:

| footprint | 8 MB | 32 MB | 128 MB | 470 MB |
|---|---|---|---|---|
| GB/s | 4600 | 3393 | 1659 | **919** |

**When to apply the bypass alias — the decision rule is measured.** Bypass pins the read
rate at ~1530 GB/s *regardless of footprint*; the cached path is faster than that only
while the working set fits L2. Same code, same 469.8 MB of load traffic, varying only the
address range touched:

| streamed footprint | 8 MB | 32 MB | 128 MB | 470 MB |
|---|---|---|---|---|
| cached | 4618 GB/s | 3406 | 1664 | 920 |
| L2-bypass | 1539 | 1533 | 1528 | 1530 |
| effect | **0.33x LOSS** | 0.45x LOSS | 0.92x | **1.66x WIN** |

**The discriminator is whether the operand is already resident in L2 WHEN IT IS READ — not
its size.** Two campaign cases settle this, and they point opposite ways at similar
footprints:

| case | footprint | state at read | alias |
|---|---|---|---|
| `quant_matmul` | 6-45 MB | L2-**resident** (re-read operands) | **0.45-0.54x LOSS** |
| `group_norm_silu` | 67.5 MB | **cold** (streamed once) | **1.23x WIN**, bit-identical |
| `group_norm_silu`, L2-warm | 67.5 MB | warm | neutral |

An earlier version of this section led with "only above ~150 MB". That is **wrong** and was
falsified from both directions: a 45 MB resident operand loses, a 67.5 MB cold operand wins.
Size only ever worked as a proxy for residency, and it is a bad one.

**So: alias an operand iff it is read cold and not re-read. Confirm with an alias-off
control on the same run — every case that has run one has changed somebody's mind.**

**Two refinements measured on a second operator (`flash_attention_grad`), which correct the
over-simple form of this rule:**

* **The crossover is NOT fixed at ~150 MB.** There, aliasing won from ~34 MB upward
  (1.230x at S=1024) — far below L2. The threshold is not the working-set size alone; it is
  whether the aliased tensors are actually *re-read*. Decide from reuse, not from a number.
* **"3x pessimization below L2" does not generalize.** The measured cost of a wrong alias
  there was **4%**, not 3x. The 0.33x figure in the table above is one access pattern
  (a tight re-read loop over a small footprint), which is the worst case, not the typical one.
* **In a CHAIN, a producer-consumer intermediate is NOT cold.** An intermediate written by
  an earlier stage is still L2-resident when the next stage reads it, so aliasing it bypasses
  a cache that was about to hit. Measured on `flash_attention_grad`: aliasing `s_gm` gains
  **1.15-1.20x standalone** and **loses in the chain** at S<=512 (0.917x at S=128) -- and the
  sign flips exactly at L2 capacity, where the intermediate stops fitting. **Benchmark the
  alias in the composed chain, never only standalone**; the standalone number has the
  opposite sign in the regime that matters.
* **The alias is for READS. A never-re-read WRITE is not a candidate.** On
  `moe_token_permute`, aliasing the 128 MiB output -- written once, never re-read, the
  textbook "no reuse" tensor -- was a **LOSS**, while aliasing the read-side `tokens` was
  worth **1.225x** (alias-off control, reproduced 1.23-1.54x at all 16 contract points).
  The rule is about avoiding a useless L2 *fill* on a streamed read, not about write traffic.
* **Aliasing MORE tensors can destroy a winning alias set.** Adding two redundancy-2.0
  tensors to a winning redundancy-1.0 set took 1.230x down to 1.039x. Alias the
  *never-re-read* tensors only, and re-measure after each addition rather than assuming the
  set composes.

**Cross-checked on an unrelated kernel, and it predicts the magnitude, not just the sign.**
`quant_matmul_a8w8` (different operator, dtype path and generation; activation aliased):

| M | working set | cached | bypass | effect | bit-exact |
|---|---|---|---|---|---|
| 1024 | 6.4 MB | 24.0 us | 44.5 us | **0.54x LOSS** | yes |
| 4096 | 23.1 MB | 62.2 us | 130.5 us | **0.48x LOSS** | yes |
| 8192 | 45.4 MB | 115.4 us | 254.4 us | **0.45x LOSS** | yes |

against the table above predicting 0.45x at 32 MB. Note the outputs stay **bit-exact even
when bypass is slower** — the alias is always semantically safe, so a wrong decision here
costs performance only, never correctness.

**Scope, stated honestly.** Of the suite cases whose contracts were checked
(`grouped_matmul` ~10 MB, `quant_matmul` ~45 MB, `flash_attention` ~50 MB at S=2048,
`grouped_matmul_swiglu_quant` 470 MB), **only one exceeds L2.** This rule is a large win on
a narrow class: weight-streaming operators at MoE/LLM scale. Check the working set before
reaching for it — for most kernels the correct action is to do nothing.

**Two further consequences for optimization.**

1. **Know when to stop.** If a stage is streaming an out-of-L2 working set and measures
   ~900 GB/s, it is *at the PTO ceiling*. Further load tuning (wider bursts, deeper rings,
   more cores, NZ pre-layout, contiguous ABI) is measurably worthless -- all of it was swept
   and is flat. Record a hardware-limit gate and spend the remaining attempts elsewhere.
2. **The only lever that works is shrinking the footprint.** The curve above is the whole
   optimization space: block the algorithm so the hot working set fits L2 and the same code
   goes 1.8-5x faster. Re-reading bytes that stay in L2 is nearly free -- a schedule that
   issued 1.94x the essential weight bytes cost only 9.5% more wall-clock.

**Reporting rule.** If the schedule issues more bytes than are essential (e.g. one weight
re-read per row-pass), GB/s computed on *essential* bytes understates the achieved rate.
Report both, or the kernel looks slower than it is and you will chase a phantom.

**Calibration.** On the same device: `zero_` writes at 1456 GB/s; torch's own `w.max()`
*reads* at 862 GB/s (i.e. a generic vendor-library read is stuck near our ceiling too); a
hand-written vendor fused operator streams at **1493 GB/s marginal**. So ~920 GB/s is a
property of the PTO load path against HBM, not of the silicon. A vendor kernel beating a
generated one by up to ~1.6x on a pure streaming stage is expected today and is not a
kernel-quality defect -- say so in the report rather than recording it as unexplained.

**The ceiling is NOT PTO's abstraction — measured.** PTO hardcodes the DMA `sid` argument
to `0` on every load and exposes no override, which made it the obvious suspect. It is not:
calling `copy_gm_to_ubuf_align_b8` **directly, bypassing `TLOAD`**, with byte-identical
parameters and only `sid` varied, gives

| sid | 0 (control) | 1 | 2 | 3 | 4 | 7 | 15 |
|---|---|---|---|---|---|---|---|
| GB/s | 923 | 918 | 922 | 916 | 922 | 917 | 921 |

The `sid=0` control reproduces `TLOAD`'s 919 GB/s, so the hand-rolled call is measuring the
same thing. **`sid` is inert, and PTO's wrapper neither adds nor removes anything** — the
raw hardware DMA instruction hits exactly the same wall. So do not attribute this ceiling to
the tile library, and do not propose a PTO change to lift it.

**Root cause: ~920 GB/s is the L2 miss-and-allocate fill rate, not HBM peak.** Hardware
counters (`msprof --aic-metrics=L2Cache`) on both arms under the same flush:

| | duration | L2 read hits | misses | hit rate |
|---|---|---|---|---|
| vendor fused op | 372.6 us | 502,453 | 30,799 | **94.2%** |
| our load probe | 508.8 us | 47 | 3,670,017 | **0.0%** |

Each event is one 128-byte line (validated: the same probe over an 8 MB footprint reports
identical total events and misses = 65,537 x 128 B = exactly 8.4 MB, i.e. compulsory
misses only). The vendor's 533,252 events cover just 68 MB of its 538 MB of GM reads, so
**~87% of its read traffic never enters the L2 read path** — it streams the weight around
L2 and spends L2 on the operand it actually reuses. We push 470 MB of never-reused data
through L2 with allocate-on-miss and hit 0%.

That single fact explains why every knob above is flat: **none of them changes whether the
stream allocates in L2.**

> **SCOPE CORRECTION -- that flatness is a property of the CACHED path only.** Every sweep in
> the table above was run without the uncached alias. Re-measured on `grouped_matmul_swiglu_quant`
> **through the alias**, two of those knobs are emphatically NOT flat:
>
> | knob, on the ALIASED path | result |
> |---|---|
> | 256 B -> 512 B bursts | 550.5 -> **446.5 us (1.233x)**; load-only probe 909 -> 1140 GB/s |
> | ring depth 3 -> 1 | 526.7 vs 446.5 us -- **1.18x SLOWER** |
> | 512 B strided -> fully contiguous | 428.4 vs 429.7 us -- 1.003x, flat again past ~512 B |
>
> The reading is consistent once stated properly: **while the stream is being absorbed by an
> L2 miss-and-allocate at ~920 GB/s, nothing upstream of that bottleneck matters. Remove the
> bottleneck with the alias and the DMA parameters become the constraint again.**
>
> **So the order is: alias FIRST, then re-tune burst width and ring depth against the aliased
> baseline.** Tuning them on the cached path measures nothing, and quoting the flatness after
> aliasing costs ~1.2x twice over. An alias-off control on the same run confirms the alias
> itself is worth **1.42x** at that geometry.

**So the rule is: do not tune the load, change what the stream does to L2.**

### PLAT-§L2Bypass: the fix — stream through the uncached address alias

On A2/A3 the L2 policy is **not** a DMA flag (which is why `sid` is inert). It is an
**aliased address window**: the same physical memory mapped at `addr + offset` with L2
disabled. CANN's own AscendC layer does exactly this
(`L2CacheAlter` in `asc/impl/basic_api/utils/kernel_utils_macros.h`):

```c
if (mode == CacheMode::CACHE_MODE_DISABLE)
    return (__gm__ T*)((uint64_t)addr + l2CacheOffset);
```

and the offset is a public runtime query:

```c
rtError_t rtGetL2CacheOffset(uint32_t deviceId, uint64_t *offset);  // rt_preload_task.h
```

On this 910B2 it returns **`0x80000000000`**. Measured on the identical kernel binary,
469.8 MB streamed, changing only the pointer handed to the launch:

| weight pointer | time | rate |
|---|---|---|
| `w` (cached, what we generate today) | 513.5 us | 915 GB/s |
| `w + rtGetL2CacheOffset()` | **307.5 us** | **1527 GB/s** |

**1.67x, and it beats the vendor's 1433 GB/s on the same operator.**

**Correctness is verified, not assumed.** Running the real `grouped_matmul_i32` kernel both
ways and comparing bit-exactly against a CPU int64 reference: identical in all three of
`fresh`, `settled`, and an adversarial `rewrite` case where the weight is dirtied in cache
immediately before the launch. Zero mismatching elements. The two views are coherent.

**Apply it to the STREAMED operand only.** L2 is not the enemy — a 0% hit rate is. Alias
the operand that is read once and never reused (here, the 470 MB weight); leave the reused
operand (here, `x`) on the normal cached path, which is precisely what the vendor does
(94% hits on its ~68 MB of cached traffic). Aliasing a reused operand will make it slower.

**"Never reused" includes reuse created by your own SCHEDULE.** Bypass removes L2 from the
path, so any byte the schedule re-reads now costs full HBM price. Measured on the same
load-only kernel, varying only how many times the schedule re-reads each expert's weight:

| schedule redundancy | cached | L2-bypass | effect |
|---|---|---|---|
| 1.00x (each byte read once) | 560.2 us | **367.9 us** | **1.52x faster** |
| 1.50x | 593.2 us | 512.5 us | 1.16x faster |
| 1.94x | 625.2 us | **667.7 us** | **0.94x — SLOWER** |

So a redundant schedule wastes much of the bypass gain, and **past ~1.9x redundancy bypass
is a pessimization** — check your schedule's redundancy before enabling it.

**But do NOT assume the converse: "remove redundancy and the load gain follows" was tried
and FAILED.** Those numbers are a *load-only* probe. Rebuilding the real `grouped_matmul`
Cube kernel for one row-pass per expert (`kRT=4`, `kRPP=64`, redundancy 1.50x -> 1.00x,
validated `max|diff|=0` on all 12 cases) bought **1.3%**, not the ~40% the probe implied:

| stage-1 kernel | cached | L2-bypass |
|---|---|---|
| `kRT=2`, 1.50x redundant (shipped) | 611.6 us | **533.3 us** |
| `kRT=4`, 1.00x redundant | 692.4 us | 526.5 us |
| `kRT=4, kNSUB=2` (lower L0C pressure) | 830.0 us | 657.0 us |

Load imbalance was excluded (`block_dim=16` gives `kRT=4` a perfect 8.00 items/lane and is
*slower*, 669.0 us) and so was L0C pressure (the `kNSUB=2` row above). The schedule change
that removes load work adds Cube-side work of its own, and it roughly cancels.

**The transferable lesson is about the probe, not the schedule:** a load-only probe bounds
the *load*, and once bypass removes the load bottleneck that bound stops predicting the
kernel. Re-derive the floor after any change that shifts the bottleneck, and treat a
projection from an isolated probe as a hypothesis to measure, not a result.

**Plumbing.** No kernel-side PTO change is needed — it is pointer arithmetic. Query the
offset once on the host, and either alias the pointer in the harness before the launch or
pass the offset as an extra scalar arg and add it to the streamed operand's base inside the
kernel. PTO has no API for this (`TLOAD` has no cache-policy parameter, and the only cache
control it ships is `TPREFETCH_ASYNC`, which pulls *into* L2) — exposing a `CacheMode` on
`GlobalTensor` is the natural upstream request.

---

## PLAT-§Manual: Manual Mode Constraints (`-DMEMORY_BASE`)

All kernels in this workflow are compiled with `-DMEMORY_BASE` (manual mode):

1. **Must use `TASSIGN`** to bind every tile to a fixed UB/L1/L0 address. Auto-allocation is unavailable.
2. **Must use explicit `set_flag`/`wait_flag`** for all MTE↔Vec synchronization.
3. **`pipe_barrier(PIPE_ALL)` is required** after every TLOAD and TSTORE to maintain memory consistency between MTE and Vec engines.
4. **Cube↔Vec communication requires `set_cross_core_flag`/`wait_flag_dev`** with explicit flag IDs.
5. **Double buffering is the standard pattern**: ping-pong two buffer slots with distinct event IDs.
6. **Do NOT switch to auto mode**. Auto-mode kernels compiled with `-DMEMORY_BASE` crash at runtime.

---

## PLAT-§Align: Tile Alignment Rules

PTO tile dimensions must be aligned to specific boundaries for correct memory access:

### Vec tiles (UB)

| Layout | Alignment requirement | Example |
|--------|----------------------|---------|
| `BLayout::RowMajor` | `cols` must be multiple of 32 bytes | fp32: cols % 8 == 0; fp16: cols % 16 == 0; int8: cols % 32 == 0 |
| `BLayout::ColMajor` | `rows` must be multiple of 32 bytes | fp32: rows % 8 == 0; fp16: rows % 16 == 0; int8: rows % 32 == 0 |

### Cube tiles (L1/L0)

| Tile type | Alignment requirement |
|-----------|----------------------|
| Mat tiles (L1Mat, L1MatZN) | Both `rows` and `cols` must be multiples of 16 (fp16) or 8 (fp32) |
| Left/Right tiles (L0A/L0B) | Inherit from parent Mat tile alignment |
| Acc tiles (L0C) | `rows` multiple of 16, `cols` multiple of 16 (fp16 accumulation in fp32) |

### TMATMUL dimension constraints

For `TMATMUL(C, A, B)` where A is M×K and B is K×N:
- **M**: multiple of 16 (both fp16 and fp32)
- **K**: multiple of 16 (fp16 inputs) or 8 (fp32 inputs)
- **N**: multiple of 16 (both fp16 and fp32)
- Runtime m/k/n ∈ [1, 4095]

**A2/A3 fractal/layout constraints for TMATMUL:**
- Left (A): `Loc == Left`, layout target-dependent
- Right (B): `Loc == Right`, layout target-dependent
- Acc (C): `Loc == Acc`
- Static shape constraints: `TileLeft::Rows == TileRes::Rows`, `TileLeft::Cols == TileRight::Rows`, `TileRight::Cols == TileRes::Cols`

### Fractal format summary

For matrix multiply `A × B = C`:

| Matrix | Fractal | Internal Order | Fractal Shape |
|--------|---------|----------------|---------------|
| A (L0A) | ZZ | Row-major internal, row-major inter-fractal | 16 × (32B/sizeof(T)) |
| B (L0B) | ZN | Col-major internal, row-major inter-fractal | (32B/sizeof(T)) × 16 |
| C (L0C) | NZ | Row-major internal, column-major inter-fractal | 16 × 16 |

### Alignment helper formula

```cpp
constexpr int AlignUp(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

// Example: align columns to 32 bytes for fp32 RowMajor
constexpr int CTC = ((ColTile + 7) / 8) * 8;  // 32-byte alignment for fp32
```

---

## PLAT-§Subblocks: Vec Sub-block UB (CORRECTED -- it is PRIVATE)

**UB is PRIVATE to each Vec sub-block. Each of `vid=0` and `vid=1` has its own
192 KB (184 KB usable below `TMP_UB_OFFSET`).** This section previously said the
opposite; see PLAT-§UB for the hardware probe and its positive control.

- Static `TASSIGN` addresses **are** private per sub-block -- both vids may use the
  SAME addresses without interfering
- Do NOT carve disjoint address ranges per vid: that halves your budget for nothing
- Do NOT `return` on nonzero vid to avoid a sharing hazard that does not exist; you
  lose half the Vec throughput. (Returning early is still WRONG for a different and
  real reason in cross-core stages -- both AIVs must reach an FFTS handshake or it
  deadlocks. See C8 and COOK-§8.6.)
- `get_subblockid()` returns the current vid
- The ONE genuinely shared resource is the `TMP_UB_OFFSET` library scratch at the top
  of UB, which is why `SaturationMode::ON` is load-bearing on the fp16->int8 `TCVT`

**Standard Vec-only preamble:**
```cpp
#if defined(__DAV_C220_VEC__)
  auto vid = get_subblockid();
  if (vid != 0) return;
  set_mask_norm();
  set_vector_mask(-1, -1);
```

---

## PLAT-§CrossCore: Cross-Core Flag Rules

- Do not emit a first-iteration `wait_flag_dev()` unless the matching producer
  sets that flag before the wait can occur
- On A2/A3 V→C, a Cube-side `wait_flag_dev` waits for both Vec subblocks;
  a `vid != 0` early return means Cube cannot safely wait on that V→C flag
- Bootstrap free-slot signals before the first consumer wait:
  ```cpp
  set_cross_core_flag<PIPE_MTE3>(FREE_FLAG_0, 2);
  set_cross_core_flag<PIPE_MTE3>(FREE_FLAG_1, 2);
  ```

---

## PLAT-§DataTypes: PTO-ISA A2/A3 Data Type Constraints

### TMATMUL — supported type triples (A2/A3)

| A type | B type | Acc type | Notes |
|--------|--------|----------|-------|
| int8_t | int8_t | int32_t | — |
| half | half | float | — |
| bfloat16_t | bfloat16_t | float | — |
| float | float | float | — |

### TLOAD — A2/A3 constraints

- Vec loads: ND→ND, DN→DN, NZ→NZ layouts only
- Mat loads: ND→ND, DN→DN, NZ→NZ, plus **ND→NZ** and **DN→ZN** conversions
- For ND→NZ or DN→ZN: `GlobalData::staticShape[0..2] == 1` and `TileData::SFractalSize == 512`
- For `int64_t/uint64_t`: only ND→ND or DN→DN supported
- Supported dtypes: int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, half, bfloat16_t, float
- Destination tile location: `TileType::Vec` or `TileType::Mat`

### TSTORE — A2/A3 constraints

- Source tile location: `TileType::Vec`, `TileType::Mat`, or `TileType::Acc`
- Vec/Mat dtypes: int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, half, bfloat16_t, float
- For `int64_t/uint64_t`: only ND→ND or DN→DN supported
- Acc source: int32_t or float; dest layout ND or NZ
- Acc static shape: `1 <= Cols <= 4095`; ND: `1 <= Rows <= 8192`; NZ: `1 <= Rows <= 65535` and `Cols % 16 == 0`

### TEXTRACT — A2/A3 constraints

- Supported element types: int8_t, half, bfloat16_t, float
- Source layouts: `(SFractal==ColMajor && isRowMajor)` or `(SFractal==RowMajor && !isRowMajor)`
- In GEMV scenarios targeting Left, also allows `(Rows==1 && isRowMajor)`
- Destination: `TileType::Left` or `TileType::Right` with target-supported fractal

### TMOV — A2/A3 constraints

- Shape must match: `Src::Rows == Dst::Rows` and `Src::Cols == Dst::Cols`
- Supported tile-type pairs:
  - `Mat → Left/Right/Bias/Scaling`
  - `Vec → Vec`
  - `Acc → Mat`
- Bias: supported dtype pairs are `int32→int32`, `float→float`, `half→float`; source row must be 1; `Cols * sizeof(SrcType)` aligned to 64B
- Scaling: destination dtype must be `uint64_t`; source row must be 1; `Cols * sizeof(SrcType)` aligned to 128B

### Vec element-wise (TADD etc.) — A2/A3 constraints

- **Data types**: int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, half, bfloat16_t, float
- Tile layout must be row-major (`TileData::isRowMajor`)

---

## PLAT-§Arity: Tile constructor arity == number of DYNAMIC extents

Verified in `pto_tile.hpp`; the overloads are SFINAE-gated, so a wrong count produces a
confusing template error rather than a clear one:

```cpp
Tile()                                                  // no DYNAMIC extent
Tile(VR, VC)   if RowValid == DYNAMIC && ColValid == DYNAMIC   // BOTH -> 2 args
Tile(VR)       if RowValid == DYNAMIC && ColValid  >  0        // row only -> 1 arg
Tile(VC)       if RowValid  >  0      && ColValid == DYNAMIC   // col only -> 1 arg
```

**Pass exactly as many runtime extents as you declared `DYNAMIC`.** `COOK-§1.65`'s example
declares both (`..., DYNAMIC, DYNAMIC, ...`) and correctly passes two — it was reported as a
compile error by a pipeline run and re-checked against the source: **the example is right**,
the failure mode is declaring one `DYNAMIC` and passing two (or vice versa).

---

## PLAT-§A2Gaps: instructions the MCP documents that A2/A3 cannot run

The MCP serves the PTO-ISA docs, which cover **all** backends. `documented: true` does not
mean "available on your target". Verified against the pinned pto-isa tree:

| instruction | reality on A2/A3 (`dav-c220`) |
|---|---|
| `TDEINTERLEAVE` | **Not implemented at all** — zero occurrences anywhere in `pto-isa/include/`. The MCP returns a constraints block tagged `backend: "a5"` only. A generator that reads `documented: true` and emits it gets a compile failure. |
| `TAND` / `TANDS` | **16-bit and 8-bit only.** `a2a3/TAnd.hpp` carries `static_assert((sizeof(T) == 2) \|\| (sizeof(T) == 1), "Fix: TAND has invalid data type.")`, so `uint32_t` masking is rejected. The MCP page has no A2A3 constraint block. Use `TSHRS` + `TSHLS` to build the mask instead. |

**Rule: before emitting any instruction, check that the constraints block you got back is
tagged for YOUR backend.** An `a5`-only block is an evidence gap, not a green light.

> **ROOT CAUSE, FOUND: the MCP indexes a DIFFERENT pto-isa than we compile against.**
>
> | | MCP's doc corpus | what our kernels link against |
> |---|---|---|
> | remote | `gitcode.com/cann/pto-isa` | `github.com/hw-native-sys/pto-isa` |
> | revision | `3b6fefaa` (branch `master`) | **`109c9f72`** (our pin) |
> | `TDEINTERLEAVE` docs | present (`docs/isa/TDEINTERLEAVE.md`) | **absent -- 0 files** |
>
> The bundled `.mcp.json` passes no `docs_path`, and the server then **auto-clones pto-isa
> into `~/.cache/npu-coding-mcp/pto-isa`** (README: "Zero-setup for PTO-ISA"). So the
> instructions below are not "documented but unimplemented" in any deep sense -- they exist
> in the revision the MCP read and not in the one we build against. **The MCP is extracting
> correctly, from the wrong repository.**
>
> **Fixes, in order of value:** (1) pass an explicit `docs_path` pointing at the pinned
> checkout (`npu-coding-mcp serve /path/to/pto-isa/docs`); (2) have the server report the
> pto-isa revision it indexed so skew is detectable rather than silent; (3) until either
> lands, keep grepping `include/pto/npu/` -- it is the only source that reflects the library
> you actually compile against.

**Working rule, unchanged in practice: THE MCP IS NOT AN EXISTENCE ORACLE.** It documents
instructions absent from our pinned tree. Confirmed absent
from the entire pinned `pto-isa` tree (zero occurrences in `include/`):

| documented by MCP | occurrences in pto-isa |
|---|---|
| `TDEINTERLEAVE` | 0 |
| `THISTOGRAM` | 0 |
| `TMULADDDST` | 0 |
| `TFUSEDMULADD` | 0 |
| `TFUSEDMULADDRELU` | 0 |
| `TPairReduceSum` | 0 |
| `SET_QUANT_SCALAR` | 0 -- the real path is `TSTORE(dst, acc, uint64_t preQuantScalar)` |
| `TADDC` | 0 NPU backend files (present in the generic header, cost model, CPU sim and README -- so it *compiles* for the CPU simulator and cannot run on device) |
| `TSUBC` | 0 NPU backend files, same shape |

`documented: true` means "there is a doc page", not "you can call it". **Seven and counting**, found by three independent runs. **Before building a
design around any instruction you have not already used, grep the pinned `pto-isa` tree for
it.** One grep; it has now cost two separate runs a redesign — one of them lost the whole
radix-select path and had to fall back to a merge sort.

### PLAT-§Precision: `TRSQRT`'s 2-arg form is a hardware APPROXIMATION

`TRSQRT(dst, src)` lowers to the hardware `vrsqrt` (`a2a3/TUnaryOp.hpp:271`). Only the
**3-arg** form (with a scratch tile) is the exact `vsqrt` + `vdiv` path.

Measured in an fp32 RMSNorm: the 2-arg form gives **1.194e-03** relative error against a
CPU-fp64 reference -- **119x over a 1e-05 tolerance** -- and it was **not faster**. It is a
silent accuracy bug in exactly the normalization kernels that reach for it. The MCP page
documents neither overload's precision.

**Use the 3-arg form unless you have measured that the approximation is inside your
contract's tolerance.**

**Worked substitution (kv_rmsnorm_rope_cache).** With `TDEINTERLEAVE` unavailable, a RoPE
pair-deinterleave over bf16 was done by viewing 64 `bfloat16` as 32 `uint32` and computing
`even = bitcast(w << 16)`, `odd = bitcast((w >> 16) << 16)` — an **exact** bf16->fp32
widening *and* the deinterleave in three integer vector ops, replacing `TCVT` entirely.

---

## PLAT-§Instructions: PTO Instruction Quick Reference

| Category | Instructions | Memory | Core |
|----------|-------------|--------|------|
| DMA load | `TLOAD`, `TPREFETCH` | GM → UB (MTE2) / GM → L1 | AIV / AIC |
| DMA store | `TSTORE` (Vec, Mat, Acc) | UB → GM (MTE3) / L0C → GM | AIV / AIC |
| Vec element-wise | `TADD`, `TSUB`, `TMUL`, `TMULS`, `TADDS`, `TDIV`, `TEXP`, `TLOG`, `TRELU`, `TSQRT`, `TMOV`, `TMAX`, `TMIN`, `TPRELU` | UB ↔ UB (Vec) | AIV |
| Vec scalar ops | `TADDS`, `TMULS`, `TDIVS`, `TSUBS`, `TEXPANDS`, `TSELS`, `TMAXS`, `TMINS` | UB ↔ UB (Vec) | AIV |
| Vec broadcast | `TROWEXPAND`, `TCOLEXPAND`, `TCOLEXPANDADD`, `TROWEXPANDADD` | UB ↔ UB (Vec) | AIV |
| Vec dtype | `TCVT` | UB ↔ UB (Vec) | AIV |
| Vec fill | `TEXPANDS` (scalar fill), `TFILLPAD` (zero pad) | UB (Vec) | AIV |
| Vec reduction | `TROWSUM`, `TCOLSUM`, `TROWMAX`, `TCOLMAX`, `TROWMIN`, `TCOLMIN`, `TROWPROD` | UB ↔ UB (Vec) | AIV |
| Cube↔Vec FIFO | `TPUSH`, `TPOP`, `TALLOC` (via `TPipe`, dir `DIR_C2V`/`DIR_V2C`/`DIR_BOTH`) | slot ring, GM- or UB-staged | AIC↔AIV — see COOK-6.6 |
| Vec complex | `TGATHER`, `TSCATTER`, `TSORT32`, `TMRGSORT`, `TPARTADD`, `TQUANT` | UB (Vec) | AIV |
| Cube extract | `TEXTRACT` | L1 → L0A/L0B (MTE1) | AIC |
| Cube reshape | `TRESHAPE` | L1 → L1 (view reinterpretation) | AIC |
| Cube move | `TMOV` (Mat→Left/Right/Bias/Scaling) | L1 → L0A/L0B/BT/FP | AIC |
| Cube matmul | `TMATMUL` | L0A,L0B → L0C (Cube) | AIC |
| Cube matmul accumulate | `TMATMUL_ACC` | L0A,L0B,L0C → L0C (Cube) | AIC |
| Matrix-vector | `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS` | L1 → Acc | AIC |
| Tile insert/extract | `TINSERT`, `TINSERT_FP`, `TCONCAT` | UB (Vec) | AIV |
| Inter-NPU comm | `TPUT`, `TGET`, `TBROADCAST`, `TREDUCE`, `TTEST`, `TWAIT` | GM ↔ GM (remote) | AIC/AIV |
| Async comm | `TPUT_ASYNC`, `TGET_ASYNC`, `TNOTIFY` | GM ↔ GM (remote) | AIC/AIV |
| Sync | `set_flag`, `wait_flag`, `pipe_barrier`, `TSYNC` | Intra-core (all pipes) | Any |
| Cross-core | `set_cross_core_flag`, `wait_flag_dev` | AIC ↔ AIV (FFTS) | AIC/AIV |


## PLAT-§C33Col: C33's last-fractal rule governs the COLUMN axis too

`C33` is written about rows: a boxed tile is correct iff `ceil(Valid/16) == ceil(Rows/16)`.
**The same constraint applies to the column extent, and violating it is silently wrong** --
no error, no fault, just incorrect data.

Probed sharply on `dav-c220` (8/8 runs each side of the boundary):

| valid column extent | result |
|---|---|
| 120 | **PASS** |
| 112 | **FAIL** |

The boundary lands exactly where the C33 fractal arithmetic says it should, which is what
distinguishes this from a vague "make it a multiple of the tile width" rule -- it is C33,
applied to the other axis.

**Apply the C33 test to BOTH extents of every boxed tile**, and lock a contract dim rather
than silently accepting a value that violates it. A generator that checks rows only will
emit a kernel that validates on friendly shapes and returns wrong answers on others.

## PLAT-§FixWAR: `PIPE_M <-> PIPE_FIX` flags do not block the scalar thread

`set_flag(PIPE_M, PIPE_FIX, id)` / `set_flag(PIPE_FIX, PIPE_M, id)` do **not** stall the
scalar thread. The loop therefore runs ahead and the next work item's `TLOAD` overwrites L1
while a `TEXTRACT` from the previous item is still in flight -- a WAR window between work
items.

Measured: **2 wrong results in 30 runs at M=8192**, while **passing an 86-case validation
sweep**. Two properties make this especially dangerous:

* it is **inter-item**, so it only appears once a lane owns more than one work item -- the
  rule-31 blind spot again;
* **a determinism check does not catch it**, because the race is between work items and
  reproduces identically run to run.

That is the third distinct instance of the same trap in this project: *a check that compares
a thing to itself cannot see a fault that affects both sides equally* (the others being the
benchmark null control and single-run validation). Guard the L1 buffer with a flag class that
actually stalls the producer, and validate with repeated runs at a size where lanes own
multiple items.


## PLAT-§RWSerial: read and write bandwidth do NOT overlap -- roofline on TOTAL traffic

A kernel that both reads and writes does not get its loads for free behind its stores.
Measured (PROBED) on `dequant_swiglu_quant`:

| | time |
|---|---|
| loads alone | 28.73 us |
| stores alone | 7.87 us |
| **serial sum** | **36.60 us** |
| **both together, measured** | **36.83 us** |

The two add. **A read+write roofline must use `(bytes_read + bytes_written) / peak`, never
`max(read_time, write_time)`** -- the `max()` form silently understates the floor by the
smaller side and will make a kernel look like it has headroom it does not have.

### Scope note on the ~920 / ~1530 GB/s figures in PLAT-§ReadCeiling

Those numbers were measured on a **pure-read, fully out-of-L2 stream of ~470 MB**. They are
not universal. On a 37.7 MB read+write stage the same probes give **574.7 GB/s cached** and
**1313.9 GB/s aliased**. The *shape* of the finding holds -- aliasing a cold stream is worth
roughly 2x, and the cached path is the one that is capped -- but **measure your own floor
rather than quoting the headline numbers**. Also probed there: read *stride* is free (four
access patterns within 0.3%), so a stride hypothesis is not worth an attempt.


## PLAT-§CmpsInt32: `TCMPS` SILENTLY IGNORES `CmpMode` on an int32 source

**Silent wrong answers. Verified in the A2/A3 backend** (`pto/npu/a2a3/TCmps.hpp`,
`GenCmpCall`):

```cpp
if constexpr (std::is_same<TIN, int32_t>::value) {
    vcmpvs_eq(dst, src0, src1, repeat, ...);       // cmpMode taken, then NEVER USED
} else {
    vcmp_dispatch(dst, src0, src1, cmpMode, ...);  // every other dtype honours it
}
```

The int32 specialization accepts `cmpMode` and unconditionally emits `vcmpvs_eq`. So
`TCMPS(dst, src_i32, scalar, CmpMode::GE)` computes **`==`**, not `>=` -- no error, no
warning, wrong mask.

Isolated cleanly in a campaign run: an `idx >= ip` predicate became `idx == ip`, and with
`ip = V/2` the mask kept exactly `1662 + 1` elements -- the single matching index.

**Do not use `TCMPS` with an int32 source for anything but equality.** Convert to fp32 first
(exact for magnitudes below 2^24, which covers any index into a tile), or build the predicate
arithmetically. This is a defect in the library rather than a platform limit, and it is a
strong upstream bug-report candidate: the parameter is in the signature and silently dropped.


## PLAT-§ValidColTrunc: a tile narrower than its GM view silently TRUNCATES the load

If a `Tile`'s `ValidCol` is narrower than the `GlobalTensor` view being loaded, `TLOAD`
transfers only `ValidCol` columns and **the remaining columns of the destination are never
written** -- no error, no warning. In a fused kernel this surfaces as a numerical error that
looks like a math bug: one campaign run chased a 1.61 relative error that was simply the RoPE
half of a fused `[kTok, 576]` load never arriving.

**The tile's valid extent is the transfer size, not a window onto a larger transfer.** When
one GM view feeds two logical halves, either load the full width into one tile and slice in
UB, or issue two loads with correctly-sized tiles. State the widths in the StageSpec so the
mismatch is visible at review rather than at 1.61.

## PLAT-§MCPIdentity: the MCP is not an IDENTITY oracle either

Already established: it is not an existence oracle (nine documented instructions have no NPU
backend, `PLAT-§A2Gaps`). It is also not reliable about *which* instruction a page describes.

`TGATHER` is the confirmed case, and the page is a **hybrid of two different instructions**:

| part of the page | which instruction |
|---|---|
| summary ("gather/select elements using an index tile or mask pattern") | the in-UB element gather |
| operands `["dst", "src0", "src1"]` | the in-UB element gather |
| C++ signature, constraints, both examples | `comm::TGATHER(parallelGroup, dstGlobalData, stagingTile)` -- the **collective multi-NPU** gather across ranks |

A generator looking for "gather elements by index" finds a summary that matches its need and
a contract that belongs to a distributed collective. **Check that the `cpp_intrinsic` header
and signature on an MCP page match the operation you think you are reading about** -- here the
header is `include/pto/comm/pto_comm_inst.hpp`, which gives it away immediately.


## PLAT-§LoadTail: `TLOAD` REPLICATES the first element past the valid extent

`TLOAD` writes `ceil(n/8)*8` floats, not `n` -- the slots between the valid extent and the
8-element boundary are written, clobbering anything pre-placed there. **What lands in them is
a replica of the loaded span's FIRST element** (46-config probe), not the next row's data as
first reported.

That distinction decides whether it hurts you, and it explains why two workers reported
contradictory symptoms of the same defect:

* **benign for a max reduction** -- a replica of an in-span element cannot exceed the max;
* **corrupting for count or sum** -- the replica is counted, and a count-based cutoff is then
  off by the number of pad slots.

Either zero the pad explicitly after the load, or size the valid extent to a multiple of 8.

## PLAT-§LaneRounding: `TCMPS` writes 64-rounded, `TSELS` writes exactly `validCol`

`TCMPS` writes its mask rounded up to 64 lanes; `TSELS` writes exactly `validCol`. Splitting a
tile mid-width between the two therefore corrupts the tail of the earlier partition unless the
split is **64-aligned**. Align mid-tile splits to 64, or place the `TCMPS` partition last.
