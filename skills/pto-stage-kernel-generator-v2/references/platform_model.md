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

**Two consequences for optimization.**

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

**Untested lead (do NOT state as fact).** Every GM→L1/GM→UB load in
`pto/npu/a2a3/TLoad.hpp` hardcodes the DMA `sid` argument to `0`, with no PTO-level
override. Whether `sid` carries a QoS/cache hint that would change the HBM stream is
undocumented in the MCP corpus and unprobed.

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
