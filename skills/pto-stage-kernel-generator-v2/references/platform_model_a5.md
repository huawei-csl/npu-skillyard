# Ascend NPU Platform Model — A5 (351x / Atlas 350)

PTO-ISA hardware model for NPU architecture version 351x (Atlas 350
accelerator card). Use this to reason about what tile data-movement and
compute patterns are architecturally legal on A5.

Section IDs use `A5-§` prefix for cross-referencing from SKILL.md.

---

## A5-§Hierarchy: Memory Hierarchy

```
GM (Global Memory / HBM)  — off-chip DRAM, several GB, all cores share it
  │
  ├── MTE2 (DMA load)  ──→  UB (Unified Buffer)  — 256 KB per AIV core
  │                              │
  │                              ├── Vec engine (Regbase SIMD ALU) — Register File ↔ UB
  │                              │
  │                              ├── L1 Buffer  — direct UB→L1 path (NEW)
  │                              │
  │                              └── MTE3 (DMA store) ──→  GM
  │
  └── MTE1 (DMA)  ──→  L1 (Cube buffer)  — 512 KB per AIC core
                               │
                               ├── L0A (left operand)  — 64 KB, TEXTRACT source
                               ├── L0B (right operand) — 64 KB, TEXTRACT source
                               └── L0C (accumulator)   — 256 KB
                                    │
                                    ├── TSTORE ──→  GM  (via Fixpipe)
                                    └── L0C → UB       (NEW: direct path to AIV)
```

### New data pathways (vs A2/A3)

| Pathway | Direction | PTO relevance |
|---------|-----------|---------------|
| **L0C → UB** | AIC→AIV | Cube output can reach Vec tiles without GM round-trip |
| **UB → L1** | AIV→AIC | Vec output can reach Cube buffer without GM round-trip |
| **GM ↔ UB** (Loop mode) | Both | MTE2/MTE3 Loop mode: Normal or Compact per iteration |

### Removed data pathways (vs A2/A3)

| Pathway | Impact on PTO |
|---------|---------------|
| **L1 → GM** (direct) | `TSTORE` from `TileType::Mat` is illegal; must go L0C→GM or UB→GM |
| **GM → L0A/L0B** (direct) | `TLOAD` cannot target L0A/L0B directly; must go GM→L1→TEXTRACT |

### Critical rule: Cube and Vec still on SEPARATE physical cores

AIC (Cube) and AIV (Vec) are independent cores. They communicate through:
1. **TPUSH/TPOP/TFREE** — FIFO-pipe-based inter-core data transfer via SSBuffer
2. **GM** — still available for bulk data transfer between cores
3. **`set_cross_core_flag` / `wait_flag_dev`** — cross-core flag signaling

---

## A5-§Topology: Core Topology

```
One AI Core cluster (351x / Atlas 350):
  ┌────────────────────────────────────────────────────┐
  │  AIC (Cube core) — independent Scalar unit          │
  │    - TMATMUL, TMATMUL_ACC, TEXTRACT, TSTORE(L0C→GM)│
  │    - L1 512KB, L0A 64KB, L0B 64KB, L0C 256KB        │
  │    - Fixpipe (quant/dequant/relu/format conversion)  │
  ├────────────────────────────────────────────────────┤
  │  AIV-0 (Vector core 0, vid=0) — independent Scalar  │
  │    - TMOV, TADD, TMUL, TEXP, TCVT, TLOAD, TSTORE   │
  │    - UB 256KB (shared with AIV-1)                   │
  ├────────────────────────────────────────────────────┤
  │  AIV-1 (Vector core 1, vid=1) — independent Scalar  │
  │    - Same capabilities, same UB space                │
  └────────────────────────────────────────────────────┘

  AIC : AIV ratio = 1 : 2
```

Cross-core sync via `set_cross_core_flag` / `wait_flag_dev`:
- A5: 1:1 signaling, 16 physical / 32 addressable semaphores
- AIV0 and AIV1 can independently trigger AIC wait (mode 4)
- AIC broadcasts to both AIV sub-blocks; each AIV reduces to AIC

---

## A5-§A5: A5 Platform Delta (vs A2/A3)

A5 changes from A2/A3 that affect PTO kernel generation:

| Resource | A2/A3 | A5 | Impact |
|----------|-------|----|--------|
| UB capacity | 192 KB (196608 bytes) | 256 KB (262144 bytes) | Larger tiles fit; update `static_assert` bound |
| L0C capacity | 128 KB | 256 KB | Larger accumulator tiles; can avoid K-slicing |
| UB bank structure | 16 groups × 3 banks × 4 KB | 8 groups × 2 banks × 16 KB | Fewer, bigger banks; 2 read ports per group |
| SIMD register width | 128-bit (Membase) | 256-bit (Regbase) | Wider tiles per instruction |
| L0A fractal format | ZZ | NZ | No NZ→ZZ conversion from L1 |
| TSTORE source tiles | Vec, Mat, Acc | Vec, Acc only (no Mat) | Mat tiles cannot be stored directly to GM |
| Cross-core signaling | FFTS broadcast | 1:1 inter-core FIFO pipes | TPUSH/TPOP/TFREE protocol |
| Code branching | `__DAV_C220_VEC__` / `__DAV_C220_CUBE__` | `__DAV_VEC__` / `__DAV_CUBE__` | Different preprocessor macros |

**When generating for A5:**
- Update UB budget guard: `static_assert(MAX_UB_ADDR <= 262144)`
- L0C budget guard: `static_assert(L0C_BYTES <= 262144)`
- Use `TPUSH`/`TPOP`/`TFREE` for inter-core sync (not FFTS `set_cross_flag`/`wait_flag_dev`)
- Use `__DAV_VEC__` / `__DAV_CUBE__` macros for code branching
- The Vec sub-block UB sharing rules (A5-§Subblocks) still apply
- `TSTORE` only from `TileType::Vec` or `TileType::Acc` — never `TileType::Mat`

**When targeting both A2/A3 and A5:**
- Use the smaller capacity (192KB UB, 128KB L0C) as the default
- Add a `#ifdef __DAV_A5__` conditional for larger tiles if beneficial
- Keep cross-core protocol compatible with A2/A3 (the stricter model)
- Or emit separate code paths under `#if defined(__DAV_C220_VEC__)` vs `#if defined(__DAV_VEC__)`

---

## A5-§Pipelines: Pipelines (within one AIV core)

```
PIPE_MTE2  — DMA load engine  (GM → UB), async, 128 B/cycle for Vec tiles
PIPE_V     — Vector SIMD ALU  (Register File ↔ UB)
PIPE_MTE3  — DMA store engine (UB → GM), async
PIPE_S     — Scalar processor (address calc, control flow)
PIPE_MTE1  — DMA for Cube     (GM → L1, L1 → L0A/L0B), on AIC core
```

**Pipe concurrency**: MTE2, Vec, and MTE3 can all execute in parallel on the
SAME AIV core. This is the basis for double-buffering and pipeline overlap.

---

## A5-§Movement: Legal Data Movement Paths

| Operation | Direction | Engine | Core | Notes |
|-----------|-----------|--------|------|-------|
| `TLOAD` (Vec) | GM → UB | MTE2 | AIV | Async; Loop mode supported |
| `TLOAD` (Mat) | GM → L1 | MTE1/MTE2 | AIC | ND→NZ and DN→ZN conversions supported |
| `TSTORE` (Vec) | UB → GM | MTE3 | AIV | Async; Loop mode supported |
| `TSTORE` (Acc) | L0C → GM | Fixpipe | AIC | Via Fixpipe with optional quant/format conversion |
| `TMOV` (V→V) | UB → UB | Vec | AIV | Same-dtype copy within UB |
| `TMOV` (M→L/R) | L1 → L0A/L0B | MTE1 | AIC | Mat→Left/Right/Bias/Scaling |
| `TMOV` (A→M/V) | L0C → Mat/Vec | Fixpipe | AIC | Acc→Mat or Acc→Vec with optional quant |
| `TCVT` | UB → UB | Vec | AIV | Dtype conversion in-place in UB |
| `TADD/TMUL/TSUB/TEXP/TRELU/TLOG` | UB → UB | Vec | AIV | Element-wise math in UB (via Register File) |
| `TROWEXPAND/TCOLEXPAND` | UB → UB | Vec | AIV | Broadcast row/col values across a tile |
| `TEXTRACT` | L1 → L0A/L0B | MTE1 | AIC | Cube operand load; NZ→NZ (no conversion) |
| `TMATMUL` | L0A,L0B → L0C | Cube | AIC | Matrix multiply, fp16/bf16/fp8→fp32 accumulate |
| `TMATMUL_ACC` | L0A,L0B,L0C → L0C | Cube | AIC | Fused accumulate |
| `TMATMUL_MX` | L0A,L0B,scales → L0C | Cube | AIC | MicroScaling format matmul (NEW) |
| `TGEMV` | L1 → Acc | Cube | AIC | Matrix-vector multiply |
| `TRESHAPE` | L1 → L1 | — | AIC | View reinterpretation (no data copy) |
| `TSTORE_FP` (Acc) | L0C → GM | Fixpipe | AIC | Vector-quantized store with scale tile |
| `set_flag/wait_flag` | — | Any | Same core | Intra-core pipe sync |
| `set_cross_core_flag/wait_flag_dev` | — | SSBuffer | AIC↔AIV | Cross-core sync |
| `pipe_barrier(PIPE_V)` | — | Vec | AIV | Stall Vec until all pending Vec ops complete |
| `pipe_barrier(PIPE_ALL)` | — | All | Any | Stall ALL pipes; required after TLOAD/TSTORE |
| `TPUSH/TPOP/TFREE` | L0C↔UB, UB↔L1 | FIFO | AIC↔AIV | Inter-core FIFO data transfer |

---

## A5-§Illegal: Explicitly ILLEGAL Operations

| Attempt | Why illegal |
|---------|-------------|
| `TSTORE` from `TileType::Mat` tile | A5 does not support Mat store; only Vec and Acc source tiles |
| Vec reading Cube L0C output directly | Different cores; must go through TPUSH/TPOP or GM |
| Cube reading Vec UB output directly | Different cores; must go through TPUSH/TPOP or GM |
| `TLOAD` from UB to GM | Wrong direction; TLOAD is GM→UB/L1 only |
| `TSTORE` from GM to UB | Wrong direction; TSTORE is UB→GM or L0C→GM only |
| TMATMUL on Vec tiles | TMATMUL requires L0A/L0B operands on AIC |
| TADD/TMUL on L0C tiles | Vec ops work on UB only |
| int4b_t (S4) in TMATMUL | S4 removed from Cube unit on A5; cast to int8_t on Vec core first |
| 4:2 structured sparsity (sparse matmul) | Hardware removed in A5 |
| Relaxing `pipe_barrier(PIPE_ALL)` around TLOAD/TSTORE | Causes data corruption; required for MTE↔V ordering |
| Using auto mode with `-DMEMORY_BASE` | Crashes at runtime; manual mode is mandatory |
| Exceeding UB 256KB peak | Silent corruption or device trap |
| Exceeding L0C 256KB peak | Silent corruption or device trap |
| Reusing event ID while previous signal/wait is in flight | Race condition, data corruption |
| Scalar indexing of `__gm__` pointer (`ptr[idx]`) | NPU Alarm crash requiring hardware reset |

---

## A5-§Events: Event IDs and Synchronization

- **8 event IDs per core**: EVENT_ID0 through EVENT_ID7
- Each `set_flag/wait_flag` pair consumes one ID for one dependency edge
- Same ID can be reused across iterations if the previous use has fully retired
- Double buffering typically uses IDs 0,1 for even/odd slots
- Cross-core flags use a separate namespace (16 physical / 32 addressable on A5)

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

## A5-§UB: UB Budget Math

```
UB capacity: 256 KB = 262,144 bytes
Alignment: 32 bytes

Double-buffered fp16 tile: max ELEMENTS_PER_TILE <= 262144 / (4 * 2) = 32,768
Double-buffered fp32 tile: max ELEMENTS_PER_TILE <= 262144 / (4 * 4) = 16,384

UB budget check formula:
  sum(bytes_per_tile * live_buffers) <= 262144

UB budget guard:
  static_assert(MAX_UB_ADDR <= 262144, "UB overflow on A5");
```

**UB bank structure (changed from A2/A3):**
```
A2/A3:  16 bank groups × 3 banks/group × 4 KB/bank = 192 KB
         1 read port + 1 write port per bank group

A5:     8 bank groups × 2 banks/group × 16 KB/bank = 256 KB
         2 read ports + 2 write ports per bank group
         (max concurrent: 2 reads + 0 writes, or 1 read + 1 write)
```

---

## A5-§L0C: L0C Budget Math

```
L0C capacity: 256 KB = 262,144 bytes (doubled from A2/A3's 128 KB)
Alignment: 64 bytes

L0C budget guard:
  static_assert(L0C_BYTES <= 262144, "L0C overflow on A5");

Larger L0C enables:
  - Larger accumulator tiles → can avoid K-slicing in many GEMM scenarios
  - More double-buffered accumulator space for pipelining
```

---

## A5-§Bandwidth: Bandwidth Model (A5)

| Path | Rate | Formula |
|------|------|---------|
| GM → UB (Vec tile) | 128 B/cycle | `ceil(bytes / 128)` |
| GM → UB (Mat tile) | 256 B/cycle | `ceil(bytes / 256)` |
| UB → UB (Vec tile) | 128 B/cycle | `ceil(bytes / 128)` |
| UB → L1 (NEW) | DMA rate | Direct path, no GM roundtrip |
| L0C → UB (NEW) | DMA rate | Direct path, no GM roundtrip |
| L1 → L0A/L0B | per-element | `ceil(bytes / 32)` |
| L0C → GM | burst | `ceil(bytes / 32)` |

---

## A5-§Manual: Manual Mode Constraints (`-DMEMORY_BASE`)

All kernels in this workflow are compiled with `-DMEMORY_BASE` (manual mode):

1. **Must use `TASSIGN`** to bind every tile to a fixed UB/L1/L0 address. Auto-allocation is unavailable.
2. **Must use explicit `set_flag`/`wait_flag`** for all MTE↔Vec synchronization.
3. **`pipe_barrier(PIPE_ALL)` is required** after every TLOAD and TSTORE to maintain memory consistency between MTE and Vec engines.
4. **Cube↔Vec communication** uses `TPUSH`/`TPOP`/`TFREE` FIFO protocol (not A2/A3 FFTS).
5. **Double buffering is the standard pattern**: ping-pong two buffer slots with distinct event IDs.
6. **Do NOT switch to auto mode**. Auto-mode kernels compiled with `-DMEMORY_BASE` crash at runtime.

---

## A5-§Align: Tile Alignment Rules

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
| Left tiles (L0A) | **NZ fractal** (changed from ZZ in A2/A3); inherit from parent Mat alignment |
| Right tiles (L0B) | ZN fractal (unchanged from A2/A3); inherit from parent Mat alignment |
| Acc tiles (L0C) | `rows` multiple of 16, `cols` multiple of 16 (fp16 accumulation in fp32) |

### TMATMUL dimension constraints

For `TMATMUL(C, A, B)` where A is M×K and B is K×N:
- **M**: multiple of 16 (both fp16 and fp32)
- **K**: multiple of 16 (fp16 inputs) or 8 (fp32 inputs)
- **N**: multiple of 16 (both fp16 and fp32)
- Runtime m/k/n ∈ [1, 4095]

**A5 fractal/layout constraints for TMATMUL:**
- Left (A): `Loc == Left`, `!isRowMajor`, `SFractal == RowMajor`
- Right (B): `Loc == Right`, `isRowMajor`, `SFractal == ColMajor`
- Acc (C): `Loc == Acc`, `!isRowMajor`, `SFractal == RowMajor`

### Fractal format summary

For matrix multiply `A × B = C`:

| Matrix | A2/A3 Fractal | **A5 Fractal** | Internal Order | Fractal Shape |
|--------|---------------|----------------|----------------|---------------|
| A (L0A) | ZZ | **NZ** | Row-major internal, column-major inter-fractal | 16 × (32B/sizeof(T)) |
| B (L0B) | ZN | ZN (unchanged) | Col-major internal, row-major inter-fractal | (32B/sizeof(T)) × 16 |
| C (L0C) | NZ | NZ (unchanged) | Row-major internal, column-major inter-fractal | 16 × 16 |

**Key benefit**: L1 Buffer uses NZ format. In A2/A3, L1→L0A required NZ→ZZ
conversion via TEXTRACT. In A5, L1→L0A is NZ→NZ (no conversion needed),
reducing overhead.

### Alignment helper formula

```cpp
constexpr int AlignUp(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

// Example: align columns to 32 bytes for fp32 RowMajor
constexpr int CTC = ((ColTile + 7) / 8) * 8;  // 32-byte alignment for fp32
```

---

## A5-§Subblocks: Vec Sub-block UB Sharing

UB is shared by both Vec sub-blocks (`vid=0` and `vid=1`).

- Static `TASSIGN` addresses are **not** private per sub-block
- Concurrent vids require disjoint address carving or a proven ping-pong protocol
- Otherwise, return early on nonzero vid before any shared addresses are reused
- `get_subblockid()` returns the current vid

**Standard Vec-only preamble:**
```cpp
#if defined(__DAV_VEC__)
  auto vid = get_subblockid();
  if (vid != 0) return;
```

**Key difference from A2/A3**: Use `__DAV_VEC__` macro (not `__DAV_C220_VEC__`).

---

## A5-§CrossCore: Cross-Core Flag Rules

- Do not emit a first-iteration `wait_flag_dev()` unless the matching producer
  sets that flag before the wait can occur
- On A5, signaling is 1:1 (no sub-block reduction needed, unlike A2/A3)
- AIV0 and AIV1 can independently trigger AIC wait (mode 4)
- Bootstrap free-slot signals before the first consumer wait:
  ```cpp
  set_cross_core_flag<PIPE_MTE3>(FREE_FLAG_0, 2);
  set_cross_core_flag<PIPE_MTE3>(FREE_FLAG_1, 2);
  ```

---

## A5-§A5InterCore: A5 Inter-Core Sync: TPUSH/TPOP/TFREE Protocol

A5 replaces the FFTS flag-based cross-core sync (A2/A3) with a FIFO-pipe-based
protocol using `TPUSH`, `TPOP`, and `TFREE`. Data flows directly between Cube
and Vec cores through dedicated FIFO buffers, not through GM workspace.

### Core branching macros

```cpp
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif
```

Use `if constexpr (DAV_VEC)` and `if constexpr (DAV_CUBE)` for branching.
Never mix Tile types from one core's path in the other's execution path.

### Communication directions

| Direction | Constant | Data flow | Producer pipe | Consumer pipe |
|-----------|----------|-----------|---------------|---------------|
| Cube → Vec | `DIR_C2V` | L0C → UB | PIPE_FIX | PIPE_V |
| Vec → Cube | `DIR_V2C` | UB → L1 | PIPE_MTE3 | PIPE_MTE1 |
| Bidirectional | `DIR_BOTH` | L0C ↔ UB | PIPE_FIX + PIPE_MTE3 | PIPE_V + PIPE_MTE1 |

### TPipe structure

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum>
using TPipe = TPipe<FlagID, DirType, SlotSize, SlotNum>;

// Parameters:
// FlagID:   inter-core sync flag ID (0-7, 8 available)
// DirType:  DIR_C2V (1), DIR_V2C (2), or DIR_BOTH (3)
// SlotSize: FIFO slot size in bytes
// SlotNum:  FIFO depth (recommended: 2)

// Initialization with buffer addresses:
using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V, sizeof(T) * M * N, 2>;
MatPipe mPipe(
    (__gm__ void *)(uint64_t)GM_SLOT_BUFFER,     // GM FIFO base
    (uint32_t)C2V_CONSUMER_BUF,                   // Cube→Vec consumer UB addr
    (uint32_t)V2C_CONSUMER_BUF                    // Vec→Cube consumer L1 addr
);
```

### TPUSH (producer side) — 3 steps

```
Step 1 — Alloc:  wait for consumer to free space
                  C2V: wait_intra_block(PIPE_FIX, FlagID+1)
                  V2C: wait_intra_block(PIPE_MTE3, FlagID+1)

Step 2 — Store:  write data to FIFO
                  AccTile → VecFIFO: pushAcc2VecFiFo (L0C → UB)
                  VecTile → MatFIFO: pushVec2MatFiFo (UB → L1)

Step 3 — Commit: signal consumer that data is ready
                  C2V: set_intra_block(PIPE_FIX, FlagID)
                  V2C: set_intra_block(PIPE_MTE3, FlagID)
```

### TPOP (consumer side) — 3 steps

```
Step 1 — Wait:  wait for producer data ready
                 C2V: wait_intra_block(PIPE_V, FlagID)
                 V2C: wait_intra_block(PIPE_MTE1, FlagID)

Step 2 — Pop:   read data from FIFO
                 VecFIFO → VecTile: popTileFromVecFiFo
                 MatFIFO → MatTile: popTileFromMatFiFo

Step 3 — Free:  signal producer that space is released (via TFREE)
```

### TFREE — release FIFO slot

```cpp
TFREE<C2VPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
```

Must be called after every TPOP. Without TFREE, the producer deadlocks
waiting for free space.

**Note**: For the TileData TPOP flow, free-space notification is already handled
internally — `TFREE(Pipe&)` is a no-op for API symmetry. For the GlobalData
flow, `TFREE(Pipe&, GlobalData&)` is required to release the slot.

### TileSplitAxis (Vec sub-block partitioning)

| SplitAxis | Description | Vec0 gets | Vec1 gets |
|-----------|-------------|-----------|-----------|
| `TILE_UP_DOWN` | Split along rows | Top half | Bottom half |
| `TILE_LEFT_RIGHT` | Split along columns | Left half | Right half |
| `TILE_NO_SPLIT` | No split | Full tile | Full tile |

### FlagID allocation

A5 provides **8 FlagIDs** (0–7). Each TPipe gets one FlagID.

| FlagID offset | Purpose |
|---------------|---------|
| FlagID | Data-ready signal (producer sets, consumer waits) |
| FlagID+1 | Space-free signal (consumer sets, producer waits) |
| FlagID+16 | Second Vec sub-block signal (when both Vec cores active) |

When both Vec sub-blocks are active, Cube must wait for both:
```cpp
wait_intra_block(PIPE_FIX, FlagID);       // Vec0 done
wait_intra_block(PIPE_FIX, FlagID + 16);  // Vec1 done
```

### A5 inter-core sync rules

- TPUSH and TPOP must be paired: one TPUSH → one TPOP → one TFREE
- Never TPUSH twice without an intervening TPOP+TFREE (FIFO overflow)
- Each TPipe gets a unique FlagID — never share FlagIDs across pipes
- Recommended FIFO depth: 2 slots
- Do not mix A2/A3 FFTS (`set_cross_flag`/`wait_flag_dev`) with A5 TPUSH/TPOP

### Supported tile types for TPUSH/TPOP

| Tile type | Role | Direction |
|-----------|------|-----------|
| `TileType::Acc` | Accumulator produced by Cube | C2V |
| `TileType::Vec` | Vector tile produced by Vec | V2C |
| `TileType::Ctrl` | Control signal from Vec | V2C_CTRL |

### GlobalData FIFO slot workflow

For cases where you need to write custom data into the FIFO slot:
```cpp
// Producer side:
TALLOC<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);
TSTORE(slot, tile);        // write tile data into the GM slot
TPUSH<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);  // commit

// Consumer side:
TPOP<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
TLOAD(tile, slot);         // read from the GM slot into local tile
TFREE<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
```

### A5 fusion operator example (Flash Attention pattern)

```
Phase 1 (Vec):    compute K^T → TPUSH K^T to Cube (V2C)
Phase 2 (Cube):   TPOP K^T → TMATMUL(Q, K^T) → TPUSH Score to Vec (C2V)
Phase 3 (Vec):    TPOP Score → Softmax normalize → TPUSH P to Cube (V2C)
Phase 4 (Cube):   TPOP P → TMATMUL(P, V) → TSTORE output to GM
```

Each phase transition uses TPUSH/TPOP/TFREE with distinct FlagIDs.

---

## A5-§DataTypes: PTO-ISA A5 Data Type Constraints

### TMATMUL — supported type triples (A5)

| A type | B type | Acc type | Notes |
|--------|--------|----------|-------|
| int8_t | int8_t | int32_t | — |
| half | half | float | — |
| bfloat16_t | bfloat16_t | float | — |
| float | float | float | — |
| **fp8_e4m3** | **fp8_e4m3** | float | NEW in A5 |
| **fp8_e5m2** | **fp8_e5m2** | float | NEW in A5 |
| **hifloat8** | **hifloat8** | float | NEW in A5 |

**Removed**: int4b_t (S4) — not supported by Cube unit on A5.

### TLOAD — A5 constraints

- Vec loads: ND→ND (row-major + NoneBox), DN→DN (col-major + NoneBox), NZ→NZ (RowMajor)
- Mat loads: all Vec layouts plus **ND→NZ** and **DN→ZN** conversions
- **MX format loads** (NEW in A5):
  - `MX_A_ZZ/MX_A_ND/MX_A_DN` → ZZ for scale A
  - `MX_B_NN/MX_B_ND/MX_B_DN` → NN for scale B
- For `int64_t/uint64_t`: `PadVal` must be `Null` or `Zero`
- For row-major ND→ND with static shapes: `ValidCol` must equal `staticShape[4]`,
  `ValidRow` must equal product of `staticShape[0..3]`

### TSTORE — A5 constraints

- Source tile must be `TileType::Vec` or `TileType::Acc` (**no `TileType::Mat` store on A5**)
- Vec dtypes: int8_t–float + **fp8_e4m3, fp8_e5m2, hifloat8, fp4_e1m2x2, fp4_e2m1x2**
- Acc: source int32_t or float; dest layout ND or NZ
- ND row-major width in bytes must be multiple of 32
- `AtomicAdd` supported with restricted dest dtypes

### TEXTRACT — A5 constraints

- **Dtypes**: int8_t, hifloat8_t, fp8_e5m2, fp8_e4m3, half, bfloat16_t, float, **fp4_e2m1x2, fp4_e1m2x2, fp8_e8m0**
- Source layouts: `(SFractal==ColMajor && isRowMajor)` or `(SFractal==RowMajor && !isRowMajor)`
- **NEW**: `ScaleLeft` (`SFractal==RowMajor && isRowMajor`) and `ScaleRight` (`SFractal==ColMajor && !isRowMajor`)
- Supports relu-pre, scalar-quantized, and **vector-quantized** forms (`TEXTRACT_FP`)

### TMOV — A5 constraints

- Supported element types: int8_t, hifloat8_t, fp8_e5m2, fp8_e4m3, half, bfloat16_t, float, fp4_e2m1x2, fp4_e1m2x2
- Source/dest dtype must be identical (for CommonCheck paths)
- Supported tile-type pairs:
  - `Mat → Left/Right/Bias/Scaling/ScaleLeft/ScaleRight`
  - `Vec → Vec/Mat`
  - `Acc → Vec/Mat`
- Acc→Vec supports `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`
- Bias: `Cols * sizeof(DstType)` aligned to 64B, max 4096 bytes
- Scaling: `Cols * sizeof(DstType)` aligned to 128B, max 4096 bytes

### Vec element-wise (TADD etc.) — A5 constraints

- **Data types**: int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, half, bfloat16_t, float
- Tile layout must be row-major (`TileData::isRowMajor`)
- Register width: 256 bytes (VL), so elements per repeat = `256 / sizeof(T)`

### TEXP — A5 constraints

- **Data types**: float, half
- Tile location: `TileType::Vec`
- **A5-only feature**: `ExpAlgorithm::HIGH_PRECISION` template parameter
  - Default: `ExpAlgorithm::DEFAULT` — faster, lower precision
  - A5 addition: `ExpAlgorithm::HIGH_PRECISION` — slower, higher precision (ignored on A2/A3)
- Usage: `TEXP<ExpAlgorithm::HIGH_PRECISION>(dst, src);`

---

## A5-§Instructions: PTO Instruction Quick Reference

| Category | Instructions | Memory | Core |
|----------|-------------|--------|------|
| DMA load | `TLOAD`, `TPREFETCH` | GM → UB (MTE2) / GM → L1 | AIV / AIC |
| DMA store | `TSTORE` (Vec/Acc only, **no Mat**) | UB → GM (MTE3) / L0C → GM | AIV / AIC |
| Vec element-wise | `TADD`, `TSUB`, `TMUL`, `TMULS`, `TADDS`, `TDIV`, `TEXP`, `TLOG`, `TRELU`, `TSQRT`, `TMOV`, `TMAX`, `TMIN`, `TPRELU` | UB ↔ UB (Vec) | AIV |
| Vec scalar ops | `TADDS`, `TMULS`, `TDIVS`, `TSUBS`, `TEXPANDS`, `TSELS`, `TMAXS`, `TMINS` | UB ↔ UB (Vec) | AIV |
| Vec broadcast | `TROWEXPAND`, `TCOLEXPAND`, `TCOLEXPANDADD`, `TROWEXPANDADD` | UB ↔ UB (Vec) | AIV |
| Vec dtype | `TCVT` | UB ↔ UB (Vec) | AIV |
| Vec fill | `TEXPANDS` (scalar fill), `TFILLPAD` (zero pad) | UB (Vec) | AIV |
| Vec reduction | `TROWSUM`, `TCOLSUM`, `TROWMAX`, `TCOLMAX`, `TROWMIN`, `TCOLMIN`, `TROWPROD` | UB ↔ UB (Vec) | AIV |
| Vec complex | `TGATHER`, `TSCATTER`, `TSORT32`, `TMRGSORT`, `TPARTADD`, `TQUANT` | UB (Vec) | AIV |
| Cube extract | `TEXTRACT`, `TEXTRACT_FP` | L1 → L0A/L0B (MTE1) | AIC |
| Cube reshape | `TRESHAPE` | L1 → L1 (view reinterpretation) | AIC |
| Cube move | `TMOV` (Mat→Left/Right/Bias/Scaling) | L1 → L0A/L0B/BT/FP | AIC |
| Cube matmul | `TMATMUL` | L0A,L0B → L0C (Cube) | AIC |
| Cube matmul accumulate | `TMATMUL_ACC` | L0A,L0B,L0C → L0C (Cube) | AIC |
| Cube matmul MX | `TMATMUL_MX` | L0A,L0B,scales → L0C | AIC |
| Matrix-vector | `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS`, `TGEMV_MX` | L1 → Acc | AIC |
| Tile insert/extract | `TINSERT`, `TINSERT_FP`, `TCONCAT` | UB (Vec) | AIV |
| Tile alloc/free | `TALLOC`, `TFREE` | FIFO management | AIC/AIV |
| Inter-NPU comm | `TPUT`, `TGET`, `TBROADCAST`, `TREDUCE`, `TTEST`, `TWAIT` | GM ↔ GM (remote) | AIC/AIV |
| Async comm | `TPUT_ASYNC`, `TGET_ASYNC`, `TNOTIFY` | GM ↔ GM (remote) | AIC/AIV |
| Sync | `set_flag`, `wait_flag`, `pipe_barrier`, `TSYNC` | Intra-core (all pipes) | Any |
| Cross-core | `set_cross_core_flag`, `wait_flag_dev` | AIC ↔ AIV (SSBuffer) | AIC/AIV |
| Inter-core FIFO | `TPUSH`, `TPOP`, `TFREE` | L0C↔UB, UB↔L1 (FIFO) | AIC/AIV |
