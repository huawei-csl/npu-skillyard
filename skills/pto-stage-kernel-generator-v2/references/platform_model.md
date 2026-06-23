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
  │    - UB 192KB shared with AIV-1             │
  ├────────────────────────────────────────────┤
  │  AIV-1 (Vec sub-block 1, vid=1)            │
  │    - Same capabilities, same UB space       │
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
| Scalar indexing of `__gm__` pointer (`ptr[idx]`) | NPU Alarm crash requiring hardware reset |

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

```
UB capacity: 192 KB = 196,608 bytes
Alignment: 32 bytes

Double-buffered fp16 tile: max ELEMENTS_PER_TILE <= 196608 / (4 * 2) = 24,576
Double-buffered fp32 tile: max ELEMENTS_PER_TILE <= 196608 / (4 * 4) = 12,288

UB budget check formula:
  sum(bytes_per_tile * live_buffers) <= 196608
```

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

## PLAT-§Subblocks: Vec Sub-block UB Sharing

UB is shared by both Vec sub-blocks (`vid=0` and `vid=1`).

- Static `TASSIGN` addresses are **not** private per sub-block
- Concurrent vids require disjoint address carving or a proven ping-pong protocol
- Otherwise, return early on nonzero vid before any shared addresses are reused
- `get_subblockid()` returns the current vid

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
