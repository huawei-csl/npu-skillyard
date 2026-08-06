# PTO Stage Generation Cookbook

This document collects reusable, compile-proven PTO code patterns extracted from
working hand-written stage kernels. Each pattern has been validated through
compilation and runtime execution on Ascend A2/A3 hardware.

Use these patterns directly. Do not rely on memory of other repositories or
invent new PTO scaffolding.

Section IDs use `COOK-§` prefix for cross-referencing from SKILL.md and REVIEWER.md.

Each section gives:
- what the pattern is for
- a reusable code portion
- when to use it
- when not to use it

---

## COOK-§0: Stage Banner Comment

Start generated kernels with a truthful file banner before the first include.

```cpp
// ============================================================================
// <stage_name>.cpp — <descriptive label> stage kernel
//
// Stage role:
//   <1-3 lines describing what this stage computes>
//
// Architecture / dataflow:
//   <vec_only | cube_only | cube_vec_pipeline | varlen_tail>
//   One logical work item handles one (batch, head, chunk) tile group.
//
// Key PTO ops used:
//   <comma-separated ops actually used in this file>
//
// Evidence gaps / conservative choices:
//   <only when needed; otherwise omit>
// ============================================================================
```

Use when:
- you want each kernel to be self-describing
- the file may be reviewed without opening the stage spec JSON

Do not use when:
- the banner would claim behavior the file does not implement

---

## COOK-§0.5: Approved PTO Type Surface Families

Use one of these exact type-surface families. Do not invent variations.

### Family A: Fixed 1D Vec Pattern

Safest pattern for contiguous Vec processing and tail-aware tiles.

```cpp
using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
using TileData =
    Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;

GlobalData xGlobal(x + offset);
TileData xTile(1, cur_cols);
TASSIGN(xGlobal, x + offset + x_offset);
TASSIGN(xTile, TILE_UB_ADDR);
```

Use when:
- the kernel walks one contiguous logical span at a time
- a fixed compile-time outer tile width with runtime valid columns is enough

Do not use when:
- a dynamic 2D GM view with runtime row and col extents is required

### Family B: Dynamic 2D GM View (Strided Loads)

For loading rectangular sub-regions from a wider 2D matrix.

```cpp
// GM view: rows stride RowWidth apart, ColWidth-wide window per load.
using GmShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
using GmStride = Stride<1, 1, 1, RowWidth, 1>;
using GmFloat = GlobalTensor<float, GmShape, GmStride>;

// Usage: load valid_rows × ColWidth columns starting at (row_offset, col_offset)
GmShape gs;
gs.shape[3] = valid_rows;
gs.shape[4] = ColWidth;
GmFloat gm_view(gm_base + row_offset * RowWidth + col_offset, gs);
UbND<float, MaxRows, ColWidth, DYNAMIC, DYNAMIC, PadValue::Zero>
    ub_load(valid_rows, ColWidth);
TASSIGN(ub_load, LOAD_UB_ADDR);
TLOAD(ub_load, gm_view);
```

Use when:
- the input matrix has wide rows that don't fit in UB as a single load
- you need to load a column slice of a [rows, wide_cols] matrix
- RowWidth is the compile-time stride between rows in GM

Do not use when:
- rows are narrow enough to load in one TLOAD
- you don't need the dynamic 2D view abstraction

### Family C: UbND Tile Alias (Device-Only)

Standard UB Vec tile alias, guarded for device-only compilation.

```cpp
#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;
#endif
```

This alias is identical across all proven hand-written kernels. Always place it
under `#ifdef __CCE_AICORE__` so the host compilation pass never sees PTO tile
template instantiations.

Use when:
- your kernel needs Vec tiles in UB with padding and tail support
- you want alignment-safe 512-byte DMA

### Forbidden type-surface inventions

- `VecShape`, `VecStride`, `VecGlobal`, `MakeGlobal`
- guessed helper constructors for `GlobalTensor`
- mixing partially qualified and partially invented APIs
- For Family B, keep exact unqualified `Shape<1,1,1,DYNAMIC,DYNAMIC>` and
  `Stride<1,1,1,DYNAMIC,1>` — do not rewrite as `pto::Shape<...,pto::DYNAMIC,...>`

---

## COOK-§1: Stable Host/Device Split

Default ABI shape for a generated stage kernel.

```cpp
#if defined(__CCE_AICORE__)
#include <pto/pto-inst.hpp>
#endif
#include "acl/acl.h"
#include <runtime/rt_ffts.h>

#if !defined(AICORE)
#define AICORE __aicore__
#endif

#if defined(__CCE_AICORE__)
using namespace pto;
#endif

template <typename T>
AICORE void stage_kernel(
    __gm__ T* in0,
    __gm__ T* out0,
    int64_t total_tiles,
    uint64_t ffts_addr
) {
  set_ffts_base_addr(ffts_addr);
  const int64_t core_idx = static_cast<int64_t>(get_block_idx());
  const int64_t block_num = static_cast<int64_t>(get_block_num());

  for (int64_t tile = core_idx; tile < total_tiles; tile += block_num) {
    (void)tile;
    // PTO tile-based body
  }
}

extern "C" __global__ AICORE void launch_stage(
    __gm__ uint8_t* in0,
    __gm__ uint8_t* out0,
    int64_t total_tiles,
    uint64_t ffts_addr
) {
  stage_kernel<float>(
      reinterpret_cast<__gm__ float*>(in0),
      reinterpret_cast<__gm__ float*>(out0),
      total_tiles,
      ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim,
    void* stream,
    uint8_t* in0,
    uint8_t* out0,
    int64_t total_tiles
) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_stage<<<block_dim, nullptr, stream>>>(
      in0, out0, total_tiles, ffts_addr);
}
```

Use always. This is the minimum compile-safe skeleton.

---

## COOK-§1.5: Vec-Only Stage Preamble

Required preamble lines for every Vec-only kernel:

```cpp
#if defined(__DAV_C220_VEC__)
  auto vid = get_subblockid();
  if (vid != 0) return;

  set_mask_norm();
  set_vector_mask(-1, -1);
```

These must appear at the top of the device compute function body, after
`set_ffts_base_addr`, and before any tile declarations or compute logic.

Why:
- `vid != 0` return: **do NOT do this by default.** UB is PRIVATE per sub-block
  (PLAT-§UB), so there is no sharing hazard to dodge, and returning throws away half
  the Vec throughput. Keep both vids working. Return early only when the stage
  genuinely has no work for vid 1 -- and NEVER in a stage with a cross-core FFTS
  handshake, where both AIVs must signal (C12, COOK-§8.6). → PLAT-§Subblocks
- `set_mask_norm()`: Reset Vec mask to normal mode (all lanes active).
- `set_vector_mask(-1, -1)`: Enable all SIMD lanes.

---

## COOK-§1.6: UB Memory Address Carving (Vec-Only)

Explicit compile-time UB memory map with a capacity guard. Required
when more than one live tile shares UB.

```cpp
// UB memory layout:
//   [INPUT_UB_ADDR    .. INPUT_UB_ADDR+BlockBytes)     = input buffer
//   [OUTPUT_UB_ADDR   .. OUTPUT_UB_ADDR+BlockBytes)    = output buffer
//   [ACC_UB_ADDR      .. ACC_UB_ADDR+RowBytes)         = row accumulator

constexpr int32_t BlockBytes = ChunkRows * CTC * static_cast<int32_t>(sizeof(float));
constexpr int32_t RowBytes = CTC * static_cast<int32_t>(sizeof(float));
constexpr int32_t INPUT_UB_ADDR = 0;
constexpr int32_t OUTPUT_UB_ADDR = BlockBytes;
constexpr int32_t ACC_UB_ADDR = BlockBytes * 2;
constexpr int32_t MAX_UB_ADDR = ACC_UB_ADDR + RowBytes;

static_assert(MAX_UB_ADDR <= 196608,
              "UB footprint exceeds A2/A3 capacity (192 KB)");

UbND<float, 1, CTC> acc_ub;
TASSIGN(acc_ub, ACC_UB_ADDR);
```

Rules:
- Include a `static_assert` for UB capacity (192 KB = 196608 bytes on A2/A3;
  256 KB = 262144 bytes on A5 — see PLAT-§UB)
- Document each buffer's purpose and range in a comment block
- Derive addresses from buffer sizes, not magic numbers
- `CTC = ((ColTile + 7) / 8) * 8` to ensure 32-byte alignment

Use when:
- more than one live tile exists in UB at the same time
- the stage has a designed memory plan

Do not use when:
- only one small tile is live at a time

---

## COOK-§1.65: TLOAD + Pad + Sync Pattern (GM → UB Load)

Proven sequence for loading a 2D region from GM into UB with tail padding.

```cpp
{
  GmShape gs;
  gs.shape[3] = valid_rows;
  gs.shape[4] = ColWidth;
  GmFloat g_gm(g_ptr + row_offset * RowWidth + col_offset, gs);
  UbND<float, ChunkRows, CTC, DYNAMIC, DYNAMIC, PadValue::Zero>
      g_load(valid_rows, ColWidth);
  TASSIGN(g_load, INPUT_UB_ADDR);
  TLOAD(g_load, g_gm);
  if (valid_rows != ChunkRows || ColWidth != CTC) {
    UbND<float, ChunkRows, CTC, ChunkRows, CTC, PadValue::Zero> g_pad;
    TASSIGN(g_pad, INPUT_UB_ADDR);
    TFILLPAD_INPLACE(g_pad, g_load);
  }
}
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
```

Key points:
- `PadValue::Zero` on the load tile ensures TLOAD fills valid region, pads rest
- `TFILLPAD_INPLACE` zero-fills the region outside the valid area
- `set_flag(PIPE_MTE2, PIPE_V, ...)` then `wait_flag(...)`: ensures DMA complete
  before Vec reads the loaded data
- The braces `{...}` scope the gm view objects so their destructors run cleanly
- **`EVENT_ID0` is safe HERE and only here: an ADJACENT set/wait pair.** The pattern
  above sets and waits on consecutive lines with no PTO call in between, so the token
  is consumed before anything can clobber it.
  **It is NOT safe for a token that must SURVIVE a PTO call.** `EVENT_ID0` is the
  library's internal scratch id -- used inside `TROWSUM`, `TROWMAX`, `TROWMIN`,
  `TROWEXPAND`, `TEXTRACT`, `TTRANS`, `TDEQUANT` and 7 more -- so a slot-free or
  data-ready token parked on `EVENT_ID0` across any of them is destroyed. That is the
  distinction COOK-§6.5 is about, and the two rules do not conflict once it is stated:
  adjacent pair -> `EVENT_ID0` fine; held across a call (any pipeline, ring or
  double-buffer) -> ids start at 1, `eid = 1 + slot + (vid ? 4 : 0)`.

---

## COOK-§1.66: Row-Tile Accumulation Scan (Vec-Only Prefix Path)

Column-tiled prefix sum over rows. Process one column segment at a time
when the full row width exceeds UB capacity.

```cpp
// Column tile width: choose a value that fits in UB and divides RowWidth.
constexpr int32_t ColTileTarget = 128;
constexpr int32_t ColTile = (RowWidth < ColTileTarget) ? RowWidth : ColTileTarget;
constexpr int32_t CTC = ((ColTile + 7) / 8) * 8;  // 32B alignment
static_assert(RowWidth % ColTile == 0,
              "RowWidth must be divisible by ColTile");
constexpr int32_t NumColTiles = RowWidth / ColTile;

for (int32_t ct = 0; ct < NumColTiles; ++ct) {
  int32_t col_off = ct * ColTile;

  // Load column slice (see COOK-§1.65)
  // ... TLOAD with Shape[valid_rows, ColTile] ...

  // Vec prefix: row 0 copies, rows 1..valid-1 accumulate
  UbND<float, 1, CTC> row_0;
  TASSIGN(row_0, INPUT_UB_ADDR);
  TMOV(acc_ub, row_0);
  pipe_barrier(PIPE_V);

  UbND<float, 1, CTC> out_row_0;
  TASSIGN(out_row_0, OUTPUT_UB_ADDR);
  TMOV(out_row_0, acc_ub);
  pipe_barrier(PIPE_V);

  for (int32_t i = 1; i < valid_rows; ++i) {
    UbND<float, 1, CTC> row_i;
    TASSIGN(row_i, INPUT_UB_ADDR + i * RowBytes);
    TADD(acc_ub, acc_ub, row_i);
    pipe_barrier(PIPE_V);

    UbND<float, 1, CTC> out_row_i;
    TASSIGN(out_row_i, OUTPUT_UB_ADDR + i * RowBytes);
    TMOV(out_row_i, acc_ub);
    pipe_barrier(PIPE_V);
  }

  // V → MTE2 sync: prevent next TLOAD from overwriting UB while Vec reads
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

  // Store result (see COOK-§1.67)
  // ... TSTORE ...
}
```

Requirements:
- Row 0 copies directly (no prior row to accumulate against)
- Rows 1..valid-1: `TADD(acc, acc, g_row_i)` then `TMOV(out_row_i, acc)`
- `pipe_barrier(PIPE_V)` after each Vec tile operation to ensure ordering
- `set_flag(PIPE_V, PIPE_MTE2, ...)` / `wait_flag(...)` before the next TLOAD
  to prevent MTE2 from overwriting the input buffer while Vec may still read it

Use when:
- the stage is a row-wise prefix sum over a wide 2D matrix
- the full row width does not fit in a single UB tile

Do not use when:
- the stage needs Cube math
- the row width is narrow enough to fit entirely in UB

---

## COOK-§1.67: TSTORE + Sync Pattern (UB → GM Store)

Proven sequence for storing a 2D region from UB to GM.

```cpp
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

{
  GmShape ss;
  ss.shape[3] = valid_rows;
  ss.shape[4] = ColWidth;
  GmFloat gs_gm(out_ptr + row_offset * RowWidth + col_offset, ss);
  UbND<float, ChunkRows, CTC, DYNAMIC, DYNAMIC>
      s_store(valid_rows, ColWidth);
  TASSIGN(s_store, OUTPUT_UB_ADDR);
  TSTORE(gs_gm, s_store);
}
set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
```

Key points:
- `set_flag(PIPE_V, PIPE_MTE3, ...)` signals MTE3 that data is ready
- `wait_flag(PIPE_V, PIPE_MTE3, ...)` ensures Vec has finished writing before DMA starts
- After TSTORE: `set_flag(PIPE_MTE3, PIPE_V, ...)` / `wait_flag(...)` ensures
  DMA complete before next iteration can reuse the UB buffer

---

## COOK-§1.68: Fixed-Length Work Distribution (Grid-Stride Loop)

Standard round-robin chunk distribution across AI cores.

```cpp
int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
int64_t total_chunks = num_seqs * chunks_per_seq;

for (int64_t gi = static_cast<int64_t>(cid);
     gi < total_chunks;
     gi += static_cast<int64_t>(block_num))
{
  int64_t seq_idx = gi / chunks_per_seq;
  int64_t local_chunk = gi % chunks_per_seq;
  int64_t bos = seq_idx * seq_len;
  int64_t chunk_start = bos + local_chunk * ChunkSize;
  int64_t remaining = seq_len - local_chunk * ChunkSize;
  int32_t valid = static_cast<int32_t>(
      remaining < ChunkSize ? remaining : ChunkSize);

  // Process one chunk of 'valid' rows...
}
```

Use when:
- the workload is grid-shaped (fixed chunk size, statically known seq_len)
- both cases (fixed-length and variable-length) must be supported

---

## COOK-§1.69: Variable-Length Work Distribution

When `cu_seqlens` is provided, chunks span variable-length sequences.

```cpp
int64_t gi = 0;
for (int64_t si = 0; si < num_seqs; ++si) {
  int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
  int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
  int64_t slen = eos - bos;
  int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

  for (int64_t c = 0; c < nc; ++c) {
    if (gi % static_cast<int64_t>(block_num) ==
        static_cast<int64_t>(cid))
    {
      int64_t chunk_start = bos + c * ChunkSize;
      int64_t remaining = slen - c * ChunkSize;
      int32_t valid = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      // Process one chunk...
    }
    gi++;
  }
}
```

Use only when the StageSpec requires variable-length sequence support.

---

## COOK-§2: Fixed Tile Types With Runtime Outer Loops

Prefer static tile shapes and move dynamicity outward.

```cpp
template <typename T, int Rows, int Cols>
using L1Mat = Tile<TileType::Mat, T, Rows, Cols,
                   BLayout::ColMajor, Rows, Cols,
                   SLayout::RowMajor, 512, PadValue::Zero>;

template <typename T, int Rows, int Cols>
using UbVec = Tile<TileType::Vec, T, Rows, Cols,
                   BLayout::RowMajor, Rows, Cols,
                   SLayout::NoneBox, 512, PadValue::Null>;

template <int TileRows, int TileCols>
AICORE void body(__gm__ half* src, __gm__ half* dst, int64_t total_work) {
  const int64_t core_idx = static_cast<int64_t>(get_block_idx());
  const int64_t block_num = static_cast<int64_t>(get_block_num());

  L1Mat<half, TileRows, TileCols> src_l1;
  UbVec<half, TileRows, TileCols> src_ub;
  UbVec<half, TileRows, TileCols> dst_ub;

  for (int64_t work = core_idx; work < total_work; work += block_num) {
    (void)work;
    // Handle one logical chunk / tile / row-block.
  }
}
```

Use when:
- the stage math tiles naturally
- only total count, tail length, or chunk count is dynamic

---

## COOK-§3: Manual Memory Map With `TASSIGN`

Use explicit address carving in manual mode. → PLAT-§Manual

```cpp
constexpr int32_t QL1Addr = 0;
constexpr int32_t KL1Addr = QL1Addr + 32768;
constexpr int32_t AccUbAddr = 0;
constexpr int32_t TmpUbAddr = AccUbAddr + 32768;

L1Mat<half, 128, 128> q_l1;
L1Mat<half, 128, 128> k_l1;
UbVec<float, 64, 128> acc_ub;
UbVec<float, 64, 128> tmp_ub;

TASSIGN(q_l1, QL1Addr);
TASSIGN(k_l1, KL1Addr);
TASSIGN(acc_ub, AccUbAddr);
TASSIGN(tmp_ub, TmpUbAddr);
```

Use when:
- compiling under `-DMEMORY_BASE`
- more than one live tile exists in UB/L1/L0

---

## COOK-§4: Compile-Time Memory Budget Guards

Emit budget checks when nontrivial buffers are live. → PLAT-§UB

```cpp
constexpr int32_t UBBytes =
    (64 * 128 + 64 * 128 + 64 * 128) * static_cast<int32_t>(sizeof(half));
constexpr int32_t L0CBytes =
    (128 * 128) * static_cast<int32_t>(sizeof(float));

static_assert(UBBytes <= 72 * 1024,
              "Tile sizes exceed the validated UB budget for this kernel.");
static_assert(L0CBytes <= 112 * 1024,
              "Tile sizes exceed the validated L0C budget for this kernel.");
```

Use when:
- the kernel has a designed memory plan
- the skill is choosing a concrete tiling scheme

---

## COOK-§5: Narrow Pipe Handoff Helpers

Named helpers for synchronization.

```cpp
template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) {
  set_flag(Src, Dst, static_cast<event_t>(id));
}

template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) {
  wait_flag(Src, Dst, static_cast<event_t>(id));
}
```

Use when:
- repeated MTE1/MTE2/MTE3/Vec handoffs exist
- the kernel has double buffering or staged compute

---

## COOK-§6: UB Ping-Pong For Pure Vec Kernels

Cleanest reusable pattern for vector-only stages.

```cpp
constexpr uint32_t BUFFER_NUM = 2;
constexpr unsigned X_PING = 0x00000;
constexpr unsigned X_PONG = 0x08100;
constexpr unsigned CAL_PING = 0x10000;
constexpr unsigned CAL_PONG = 0x18100;

// Ids 3/4, NOT 0/1. These tokens are HELD ACROSS the loop body, so they must avoid
// every id the library touches: EVENT_ID0 is its internal scratch (14 core
// instructions), and ID1/ID2 are used by the comm/ collectives. ID3-ID6 are unused
// on device. Costs nothing and removes a whole class of "mysterious hang".
set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);

for (uint32_t processed = 0, ping = 1; processed < elements_to_process;
     processed += tile_cols) {
  const int8_t buf = ping ? 0 : 1;
  // NOT `const event_t` -- the CCE builtin check rejects a cv-qualified event_t
  // with "the 3rd parameter must be a type 'event_t'". See C1x.
  event_t ev = ping ? static_cast<event_t>(EVENT_ID3)
                    : static_cast<event_t>(EVENT_ID4);

  TileData xTile(1, tile_cols);
  TileData calTile(1, tile_cols);
  TASSIGN(xTile, buf == 0 ? X_PING : X_PONG);
  TASSIGN(calTile, buf == 0 ? CAL_PING : CAL_PONG);

  wait_flag(PIPE_V, PIPE_MTE2, ev);
  TLOAD(xTile, xGlobal);
  pipe_barrier(PIPE_ALL);

  set_flag(PIPE_MTE2, PIPE_V, ev);
  wait_flag(PIPE_MTE2, PIPE_V, ev);
  wait_flag(PIPE_MTE3, PIPE_V, ev);

  // Vec compute chain — use PTO tile ops only:
  TMULS(calTile, xTile, (half)-1);
  pipe_barrier(PIPE_ALL);
  TEXP(calTile, calTile);
  pipe_barrier(PIPE_ALL);
  TADDS(calTile, calTile, (half)1);
  pipe_barrier(PIPE_ALL);

  set_flag(PIPE_V, PIPE_MTE3, ev);
  wait_flag(PIPE_V, PIPE_MTE3, ev);
  TSTORE(yGlobal, calTile);
  pipe_barrier(PIPE_ALL);

  set_flag(PIPE_MTE3, PIPE_V, ev);
  set_flag(PIPE_V, PIPE_MTE2, ev);
  ping = 1 - ping;
}
```

Important: `TEXP` is the only approved way to compute exponentials. Never use
raw scalar `exp()`, `expf()`, `std::exp()`, or `__builtin_expf()` in PTO kernels.

Use when:
- the stage is activation-like, pointwise, prefix-like, or rowwise Vec-only math

---

## COOK-§6.5: Deep Software Pipelines -- balance, AND a per-sub-block id partition

> **READ THIS FIRST -- it is the single most common reason a double-buffered kernel
> "mysteriously" hangs or corrupts, and it cost this project two abandoned optimizer
> rounds.**
>
> **`set_flag`/`wait_flag` event ids are a per-CORE resource, SHARED BY BOTH AIV
> SUB-BLOCKS.** If both sub-blocks run the same flag protocol with the same event
> ids, they consume each other's tokens. A `wait_flag` is then satisfied by the
> *other* sub-block's token instead of blocking, the load races ahead of the store,
> and you get a hang or silent corruption.
>
> Measured on 910B2, one memory-bound streaming kernel, same device, same session,
> with an independent control kernel passing before and after:
>
> | both sub-blocks active | correct runs |
> |---|---|
> | same event ids on both | **2 / 14** |
> | ids partitioned by sub-block (sub0: 0-3, sub1: 4-7) | **10 / 10** |
>
> The fix is one line -- offset every event id by the sub-block:
>
> ```cpp
> // sub-block 0 gets ids 0..3, sub-block 1 gets ids 4..7, on EVERY pipe pair
> // NOTE THE +1: EVENT_ID0 is reserved by the library. See "EVENT_ID0" below.
> AICORE inline int eid(int slot) { return 1 + slot + (get_subblockid() ? 4 : 0); }
> ...
> wait_flag(PIPE_MTE3, PIPE_MTE2, (event_t)eid(s));
> ```
>
> This budgets 4 ids per sub-block, so a pipeline of up to 4 slots fits. Deeper than
> that, or serialize the sub-blocks (`if (get_subblockid() != 0) return;`) and accept
> half the memory parallelism.
>
> ### Corroborated against the vendor's own runtime (CANN 9.1.0 AscendC)
>
> `asc/impl/basic_api/kernel_tpipe_impl.h` allocates flag ids like this:
>
> ```cpp
> template <HardEvent evt> TEventID TPipe::AllocEventID() {
>     auto ptr = this->g_tpipeImpl.eventPool_ + EventToIndex(evt);  // pool PER event class
>     auto lastId = sff0(ptr->eventOccupy);                          // first free id
>     ...
> }
> ```
>
> with `constexpr int32_t QUE_MAX_EVENT = 8;` for `__NPU_ARCH__ == 2201` (dav-c220), and
> `HardEvent` enumerating every `(src,dst)` pipe pair (`MTE2_V`, `V_MTE2`, `MTE3_MTE2`, ...)
> as a separate class. So the id space is **8 per pipe-pair class**, allocated dynamically
> from a per-class occupancy bitmap — which is exactly the model this section assumes when
> it hands sub-block 0 ids 0-3 and sub-block 1 ids 4-7 *on every pipe pair*. The rule above
> stands; this is independent confirmation of its shape, not a change to it.
>
> **But the depth guidance was off-consensus.** Counting `BUFFER_NUM` across the ~490 shipped
> AscendC kernels that declare one:
>
> | ring depth | 1 | **2** | 3 | 4 | 5+ |
> |---|---|---|---|---|---|
> | vendor kernels | 145 | **316** | 4 | 9 | 5 |
>
> **Depth 2 is the vendor norm and depth >=3 is under 4% of kernels.** `deep_norm_grad`
> itself ships at `BUFFER_NUM = 1`. Prefer depth 2; treat depth >=3 as an exotic choice that
> must earn its keep against a measurement, not as the default a "deep pipeline" section
> implies. Two optimizer rounds were spent here chasing depth 3.
>
> **Why this looked like something else.** It is a race, so it is probabilistic, and
> sampling one run per shape makes it look shape-selective -- which is exactly how
> `dequant_swiglu_quant`'s parity round came to report a "shape-dependent deadlock"
> and conclude the tokens are single-bit rather than counting. The tokens ARE
> counting (COOK-6.5's original claim, independently re-verified 16 deep on three
> pipe pairs). The missing rule was never about depth or about pipe pairs; it was
> about the two sub-blocks sharing one id space. A kernel that does
> `if (vid != 0) return;` never hits it, which is why toy probes passed while the
> real kernels failed.

**For the complete, validated double-buffered loop -- both WAR guards, the id
partition, and an exact drain -- copy COOK-6.7 rather than assembling one here.**

A >=3-deep prefetch needs a WAR guard so a load cannot overwrite a slot still being read. The
working form gives each slot `s` an `MTE2->V` flag (load ready) and an `MTE3->MTE2` flag (buffer
free, set after the store that last read the slot), and collapses `V->MTE3` onto a single id because
its set/wait are back-to-back.

```cpp
// slot s in {0,1,2} -> event id s ; V->MTE3 always on ID3
for (int i = 0; i < n_chunks + 2; ++i) {
  const int s_load = i % 3, s_comp = (i - 2) % 3;
  if (i < n_chunks) {
    if (i >= 3) wait_flag(PIPE_MTE3, PIPE_MTE2, (event_t)s_load); // buffer free
    TLOAD(buf[s_load], ...);
    set_flag(PIPE_MTE2, PIPE_V, (event_t)s_load);
  }
  if (i >= 2) {
    wait_flag(PIPE_MTE2, PIPE_V, (event_t)s_comp);
    /* ... Vec body on buf[s_comp] ... */
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(..., buf[s_comp]);
    set_flag(PIPE_MTE3, PIPE_MTE2, (event_t)s_comp);  // buffer free
  }
}
```

**A separated `wait_flag(PIPE_MTE3, PIPE_MTE2, id)` -- set in one iteration, waited in a later one
-- is fully supported and reliable on dav-c220.** So is `wait_flag(PIPE_V, PIPE_MTE2, id)` (use it
when the guarded buffer is only read by Vec; it frees the slot one pipe stage earlier). Verified by
standalone probe at depths 2/3/4, 2-4096 iterations, block_dim 1-48, one and two AIV sub-blocks,
1-16 KB tiles, and 600 launches across 30 processes per form: zero hangs, bit-identical output. The
vendor's own `pto-isa/demos/torch_jit/add/add_custom.cpp` ships the same separated form.

> **OPEN, reported but not yet root-caused (2026-08-04).** A `deep_norm_backward` run
> reported two deadlocks against this paragraph: (a) a collapsed counting-id prefetch
> ring safe at depth 2 that hung at depth 3, reproduced on two kernels, and (b) a hang
> on `wait_flag(PIPE_V, PIPE_MTE2, id)` specifically.
>
> The pipe pair itself is NOT the problem -- the library uses it, e.g.
> `pto/comm/a2a3/TReduce.hpp:79` does `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0)`. The
> leading hypothesis is **id contention rather than pipe-pair support**: the example
> above previously bootstrapped its held tokens on `EVENT_ID0`/`ID1`, the two most
> contended ids, and a deeper ring consumes more ids and so collides more readily.
> That is why the ids were moved to 3/4.
>
> Not yet probed, because a probe whose failure mode is a DEADLOCK wedges the device
> for ~30 minutes and only `npu0` is currently usable (see `isa_probes/README.md`).
> Treat the "zero hangs at depths 2/3/4" verification above as conditional on ids that
> avoid the library's, until this is settled on a dedicated device.

**The single failure mode is an unbalanced set/wait count on a `(srcPipe, dstPipe, event_id)`
triple.** Every `wait_flag` that executes must have a `set_flag` that executed before it -- from the
bootstrap, a prior iteration, or the same iteration. Deadlock is what an unbalanced count looks
like, and it is *deterministic*: 100% hang, first launch, every time. If a pipeline hangs, do not
hunt for a bad pipe pair or a bad event id -- count the sets and waits on each triple along every
path, including the prologue, the `if`-guarded issue slots, the tail iterations, and the drain. The
two classic bugs are (a) no bootstrap `set_flag` for the initial "buffer free" tokens, and (b) a
`set_flag` inside a conditional whose matching `wait_flag` is not under the same condition.

Two things that do **not** cause deadlock, contrary to a claim that appeared in an earlier version
of this entry: sharing one event id across several pipe pairs (ids are scoped per
`(srcPipe, dstPipe)`; four pairs on one id ran clean), and using `EVENT_ID4`-`EVENT_ID7` (guard ids
4,5,6,7 ran clean). Flags are counting semaphores at least 16 deep per triple, not one-bit flags, so
extra outstanding tokens leak rather than deadlock.

One real id hazard, distinct from the above: several PTO ops emit an internal `set_flag`/`wait_flag`
pair on a **hardcoded `EVENT_ID0`** (`pto::PtoSetWaitFlag`). Most use `PIPE_S` pairs and are
harmless, but `MScatter` uses `(MTE3, MTE2)`, `MGather` uses `(MTE2, V)` and `(MTE2, MTE3)`, and
`TTrans` uses `(MTE3, V)` and `(V, MTE3)`. If your kernel calls one of those, keep your pipeline
flags for that pair off `EVENT_ID0`.

**CORRECTED -- this is a BUG, not hygiene.** An earlier version of this paragraph said the
collision "did not deadlock or corrupt in a simple pipeline, so treat this as hygiene, not a
known bug". That conclusion came from a probe too simple to expose it. The pto-isa source
settles it: `EVENT_ID0` is used internally by **14 core A2A3 instructions** including every
row reduction and `TEXTRACT`, so a user token held on `EVENT_ID0` across any of them is
destroyed. It is silent, and it is not restricted to the pipe pairs listed above. See
*EVENT_ID0 IS RESERVED BY THE LIBRARY* below, and never use `EVENT_ID0` for a token that must
survive an instruction call.

Independent of sync: **validate a deep pipeline at a high iteration count, not the minimum** -- a
genuine buffer-reuse hazard can pass at 2 iterations and fail at >=4 -- and re-check determinism
with fresh GM allocations, not just repeated launches on the same buffers.

**If your seam is Cube<->Vec rather than intra-core, reach for the
`TPipe`/`TPUSH`/`TPOP` FIFO (COOK-6.6) before hand-rolling it -- it handles
space-wait, store and data-ready sync for you. Bring it up at your production
shape first: COOK-6.6 records shape/depth combinations that fault on hardware.**
This entry applies only to intra-core pipe pairs.

Scope: these findings are from Vec-only probes and say nothing about `SYNCALL<Mix>` cross-core
seams; they settle the intra-core pipe-pair question only.

Measured effect: on a masked-softmax kernel, 2-slot prefetch took the ratio to vendor from 1.02x to
0.92x and the 3-deep form to 0.75x. Combine with `TAXPY` (C27-escape) to keep the Vec body short
enough that the loads dominate.

---

### EVENT_ID0 IS RESERVED BY THE LIBRARY -- never hold a user token on it

Read from the pto-isa source, not inferred: **`EVENT_ID0` appears 250 times across
the library**, against 26 for `EVENT_ID1`, 5 for `ID2`, 1 for `ID3` and 16 for `ID7`.
It is the library's default internal scratch id, and it is used *inside instructions
your kernel calls*:

```
A2A3 instructions that issue set_flag/wait_flag on EVENT_ID0 internally:
  TROWSUM  TROWMAX  TROWMIN  TROWEXPAND  TROWREDUCEIDXOPS
  TEXTRACT  TINSERT  TCONCAT  TTRANS  TDEQUANT  TFILLPAD  TPUSH  TSYNC  SYNCALL
```

That list contains the row reductions used by **every** softmax / attention /
normalization kernel and `TEXTRACT`, used by **every** Cube kernel. `TRowExpand`
does it once per row *inside its loop*:

```cpp
// pto/npu/a2a3/TRowExpand.hpp
for (int i = 0; i < validRow; i++) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    ...
}
```

**So any outstanding user token on `EVENT_ID0` is destroyed the moment you call one
of these.**

**This does NOT ban `EVENT_ID0` outright, and COOK-§1.65 is not in conflict.** The
distinction is whether the token has to SURVIVE a call:

| shape | `EVENT_ID0`? |
|---|---|
| adjacent `set_flag` / `wait_flag` with no PTO call between them | **fine** -- consumed immediately (COOK-§1.65, EX-§2) |
| a token held across ANY PTO call -- ring slot, double-buffer, prefetch, cross-stage | **never** -- start ids at 1 | A slot-free token you set before a `TROWSUM` and wait on after it is
simply gone -- the `wait_flag` is satisfied by the library's token, the ring
advances early, and you get corruption or a hang depending on timing.

**This is why the original `eid(slot) = slot + vid*4` was wrong**: it puts slot 0 of
sub-block 0 exactly on `EVENT_ID0`. A moe_token_permute run reported precisely that
signature -- "every corrupt row was on slot 0 of sub-block 0" -- and it is explained
in full by the source above. The formula now starts at 1.

Safe-id summary, from the source:

| id | reserved by | safe for your ring? |
|---|---|---|
| `EVENT_ID0` | 14 core A2A3 instructions (above) | **NEVER** |
| `EVENT_ID1`, `ID2` | `comm/` collectives only (`TGATHER`, `TREDUCE`, `TPUT`, `TSCATTER`, `TBROADCAST`, `TGET`) | yes, unless the stage uses collectives |
| `EVENT_ID3`-`ID6` | nothing on device (`ID3` only in the CPU stub) | **yes -- prefer these** |
| `EVENT_ID7` | `TPOW` only | yes, unless the stage uses `TPOW` |

The earlier note that this collision was "hygiene, not a known bug" was wrong and has
been removed. It is a bug, the mechanism is in the source, and it is silent.

## COOK-§6.6: `TPUSH`/`TPOP` -- the supported Cube<->Vec staging FIFO

**Read the scope line first: this is for CROSS-CORE Cube<->Vec pipelines. It is NOT a
replacement for intra-core UB double buffering.**

> **What the implementation actually does** (read from `pto/npu/a2a3/GridTPush.hpp`
> and `GridTPop.hpp`, not inferred from behaviour). This matters because COOK-6.6's
> earlier "exact but faults once the ring iterates" was an observation with no
> mechanism, and the mechanism is right there in the source:
>
> * **It is a cross-RANK neighbour-SRAM ring, not a set_flag/wait_flag FIFO.** The
>   producer copies the tile into the *neighbour's* SRAM slot
>   (`copy_sram_to_neighbour_sram`) and then bumps a cross-rank counter with
>   `mtspr_neighbor_counter(Ready, ...)`. The consumer spins on that counter with
>   `wfe_neighbor_counter(...)` and, after copying out, bumps the producer's `Free`
>   counter. **No `EVENT_ID` is involved anywhere in the path.**
> * **The slot index wraps modulo `SlotCount`** (`idx % Pipe::SlotCount`), so the
>   ring genuinely reuses slots -- which is why a single tile passes and a
>   multi-tile run faults.
> * **`GRID_TPUSH_IMPL` and `GRID_TPOP_IMPL` DISCARD the failure signal.** Both are
>   one-line wrappers over the `TRY` form with `maxSpins = 0`:
>   ```cpp
>   (void)GRID_TRY_TPUSH_IMPL<Dir, Pipe, TileProd>(pipe, tile, 0);
>   ```
>   The `TRY` form returns `false` and sets a fault flag when the ready/free counter
>   does not arrive; the non-`TRY` form throws that away. So the plain `TPUSH`/`TPOP`
>   have **no backpressure and no error reporting** -- exactly matching the probed
>   "1 tile PASS, 4 tiles FAULT". If you use this FIFO, use the `TRY` forms and check
>   the return, or provide your own flow control.
> * **There is a mandatory publish fence** between the payload store and the ready
>   flag -- `pipe_barrier(PIPE_ALL); dsb(DSB_DDR);` -- and the source comment states
>   that without it the scalar-pipe flag write can become visible on the peer before
>   the MTE3 slot bytes commit, so the consumer's `TLOAD` reads pre-publish zeros.
>   Any hand-rolled cross-rank publish needs the same fence. `TPipe`'s direction enum is
`DIR_C2V` (1), `DIR_V2C` (2), `DIR_BOTH` (3), plus `_CTRL`/`_GM` variants
(`include/pto/common/fifo.hpp`). There is no Vec->Vec direction. If what you want is
MTE2 run-ahead over Vec inside one core, this entry does not apply -- see COOK-6.5.

When a kernel alternates Cube and Vec stages (attention, GEMM+activation chains),
the intended mechanism is **not** hand-rolled `set_flag`/`wait_flag` but a FIFO
declared as a `TPipe`, written with `TPUSH` and read with `TPOP`:

```cpp
// include/pto/npu/a2a3/TPush.hpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

`TPUSH(pipe, tile)` does three things for you (`docs/isa/TPUSH.md`):
1. waits for FIFO space when `Pipe::shouldWaitFree(pipe.prod.tileIndex)`;
2. stores the producer tile into the current slot;
3. records data-ready synchronization for the consumer.

That is the whole **cross-core** protocol: no FFTS flag ids to allocate, no
cross-core credit balance to reason about, no WAR guard to hand-write. It does
**not** cover the intra-core ordering around the call -- you still write the
ordinary `set_flag`/`wait_flag` pairs (`PIPE_M -> PIPE_FIX` before the push,
`PIPE_MTE3 -> PIPE_MTE2` around the pop), exactly as in a kernel with no FIFO.
There is also a GM-slot workflow: `TALLOC` a slot view, write it with ordinary
memory instructions, then `TPUSH(Pipe&, GlobalData&)` to commit.

**Vendor reference pattern.** pto-isa ships a Flash Attention pipeline recipe using
exactly this (`.claude/skills/pto-isa-flash-atten-a3-pipeline/` upstream): a
GM-staged FIFO with an 8-deep slot ring, three logical pipes (QK cube->vec,
PV cube->vec, P vec->cube), and a `QK_PRELOAD` depth knob so the producer runs
several slots ahead. Its shape:

```text
cube QK:  [QK0] [QK1] [QK2] [QK3] ...
vec  P :        [P0]  [P1]  [P2]  ...   (lags by QK_PRELOAD-1)
cube PV:              [PV0] [PV1] ...
vec  GU:                    [GU0] ...
```
with a distinct prologue (prime the FIFO), steady loop, and epilogue (drain) --
collapsing the prologue into the steady loop breaks the lag invariant.

**Why this entry exists.** `TPUSH`/`TPOP` have been present in the pinned tree the
whole time (`TPush.hpp`, `TPop.hpp`, `GridTPush.hpp`, `GridTPop.hpp`,
`grid_pipe_runtime.hpp`, with `docs/isa/TPUSH.md` and `docs/isa/TPOP.md`), yet this
cookbook mentioned them zero times, `platform_model.md` (A2/A3) zero times --
only `platform_model_a5.md` covered them -- and none of 76 generated kernels used
one. A generated flash-attention kernel consequently staged its Cube<->Vec seam by
hand and saturated at 48 TFLOP/s against a vendor operator reaching 128.

**Status: PROBED ON REAL HARDWARE (910B2, CANN 9.1.0, pto-isa `109c9f72`).** It
works, it is exact, and it is fragile in ways we could not pin down. Read all
three parts before using it.

*(1) It works and it is bit-accurate.* A C2V pipe (`TPipe<0, DIR_C2V, SlotBytes,
1>`, `TPUSH` of a `TileAcc` on Cube, `TPOP` of a `Tile<Vec>` on both AIV
sub-blocks, `TILE_UP_DOWN`) computing `A@B + bias` matched a CPU-float64
reference to **max relative error 1e-07** over 1..64 tiles at 128x128x128. Two
API facts worth having: `TPOP` **assigns the consumer tile itself** out of
`C2V_CONSUMER_BUF` with `LocalSlotNum`-way rotation, so do not `TASSIGN` the
popped tile; and the FIFO handles only the *cross-core* handshake -- you still
write the ordinary intra-core `set_flag`/`wait_flag` around `TPUSH`/`TPOP`
(`PIPE_M -> PIPE_FIX` before the push, `PIPE_MTE3 -> PIPE_MTE2` around the pop).
The earlier claim in this entry that there is "no set/wait balance to reason
about" was wrong and is corrected here.

*(2) Some (shape, depth) combinations fault, deterministically.* Each row below
was repeated on independent, freshly health-checked devices:

| tile M x K x N | slot | depth 1 | depth 2 |
|---|---|---|---|
| 128 x 128 x 128 | 64 KB | PASS 8/8 | FAULT 8/8 |
| 128 x 128 x 64  | 32 KB | PASS | PASS |
| 128 x 128 x 32  | 16 KB | FAULT 6/6 | PASS |
| 16 x 32 x 32    | 2 KB  | PASS 5/5 | PASS 5/5 |

The fault is an AIV aicore exception (`507015`, *"fftsplus aivector error"*,
decoded as *"VEC instruction to read/write UB is out of bounds"*) -- not a hang,
not a wrong answer. Ruled out as causes: our UB layout (80 KB of extra clearance
changes nothing), our compile flags (pto-isa's exact CCE option set changes
nothing), and the torch_npu/ctypes loader (a pure-ACL host reproduces it). **Not**
ruled out: a defect in our probe -- pto-isa's own `tpushpop_cv` conformance
kernel, run unmodified through the same loaders, passed 50/50.

*(2b) Second independent probe, at an attention production shape -- the failure tracks
RING ITERATION, not just depth.* A later run probed `TPipe<0, DIR_C2V, 65536, depth>`
with `TileAcc<float,128,128>` and `TILE_NO_SPLIT` before building a stage around it,
health-checking the device after every fault so nothing was attributed to poisoning:

| tiles pushed | depth 1 | depth 2 |
|---|---|---|
| 1 | PASS (rel err 5.4e-08) | PASS (5.4e-08) |
| 4 | aicore exception 507015 | aicore exception 507015 |

So a single push/pop is exact at both depths, and it faults as soon as the ring is
actually *iterated* -- i.e. once `tileIndex > 0` brings the `shouldWaitFree` /
free-notify path into play. Combined with the table above, treat "it worked at one
tile" as no evidence at all: **probe at your real trip count, not just your real
shape.** That run fell back to a hand-rolled point-to-point FFTS handshake, and its
ablation then showed the seam sync was only 69 us of 1904 us -- so on that kernel
TPUSH could not have been the win anyway. Measure where the time actually is before
spending effort on the seam.

*(3) Therefore: usable, but validate the exact production configuration.* Do not
assume a pipe that passes at one tile shape or FIFO depth passes at another --
that assumption is falsified above. Bring the FIFO up at the real shape first, on
real hardware, before building a stage around it. And note **no performance
number exists**: the comparison against a hand-staged seam needs the 64 KB-slot,
depth>=2 configuration, which is the one that faults, so there is no
TPUSH-vs-hand-rolled ratio to cite. Full probe, controls and raw counts:
`skillyard-runs/isa_probes/README.md`.

Two smaller findings from the same probe. `docs/isa/TPUSH.md` in the pinned tree
says `SyncPeriod = (SlotNum <= 2) ? SlotNum : SlotNum/2`; both the A2/A3 and A5
headers define `SyncPeriod = SlotNum`, so trust the header. And the npu-coding-mcp
serves a **newer** TPUSH page than the pinned checkout (extra overloads, and the
wrong `SyncPeriod` line already removed) -- when the two disagree on this
instruction, the MCP is ahead.

---

## COOK-§6.7: The complete double-buffered Vec loop (copy this)

The pattern below is the one that took `dequant_swiglu_quant` from **1.120x to
1.084x** of `npu_dequant_swiglu_quant` (canonical protocol, null control valid,
disjoint CIs), after two earlier rounds abandoned double buffering as "shape
dependently unsafe". Nothing about it is subtle *once written down*, and everything
about it is easy to get wrong from scratch -- so start from this, do not re-derive it.

**Retiring the per-item `pipe_barrier(PIPE_ALL)` retires TWO hazards, not one.**
That barrier is what a non-pipelined loop uses to cover everything at once. Replace
it and you owe *both* of these:

| hazard | who races whom | guard |
|---|---|---|
| **input WAR** | next item's MTE2 DMA overwrites a slot the current Vec work still reads | `V -> MTE2` token per slot |
| **output WAR** | next item's Vec writes to the output tiles while the previous item's MTE3 stores are still reading them | `MTE3 -> V` token |

The input one is obvious and everyone writes it. **The output one is not, and omitting
it produces a kernel that validates EXACTLY at small sizes and corrupts at production
size** -- because at small sizes each lane owns one item and the pipelined path is
never taken. That is precisely how it failed here: exact at T=8 and T=64, wrong from
T=512 up. See COOK-6.5 for why event ids are partitioned by sub-block.

```cpp
// Event ids are a per-core resource shared by both AIV sub-blocks (COOK-6.5), AND
// they are allocated PER PIPE-PAIR CLASS -- confirmed against CANN's own allocator,
// `TPipe::AllocEventID<HardEvent evt>`, which indexes `eventPool_ + EventToIndex(evt)`
// with QUE_MAX_EVENT = 8 on arch 2201. So each (src,dst) class has its own 8 ids.
//
// FIXED: this used to read `eb = get_subblockid() * 4`, which gives vid 0 the range
// 0..3 -- putting its first slot on EVENT_ID0, the id COOK-6.5 reserves. §6.5 even
// names `slot + vid*4` as the wrong formula, so the cookbook was documenting the bug
// in one section and committing it in another.
//
// Because the pools are per class, each class only has to partition ITS OWN slots
// across the two sub-blocks, so 2 slots x 2 sub-blocks fits in 1..4 with 0 free:
const int32_t eb       = 1 + static_cast<int32_t>(get_subblockid()) * 2;
const int32_t kSlotA   = eb + 0;   // slot 0 free / ready
const int32_t kSlotB   = eb + 1;   // slot 1 free / ready
// kStoreEv / kOutEv live on DIFFERENT pipe-pair classes (V->MTE3 and MTE3->V) from
// the slot tokens, so they may reuse the same NUMBERS without colliding. If you are
// unsure which class a flag lands on, keep the numbers distinct -- ids are cheap
// per class and a collision is a silent hang.
const int32_t kStoreEv = eb + 0;   // V -> MTE3 before the stores
const int32_t kOutEv   = eb + 1;   // MTE3 -> V : output tiles free again

// How many items THIS lane owns -- needed so prologue and drain are exact for
// every trip count, including 0 and 1.
int64_t my_items = 0;
for (int64_t t = lane; t < num_items; t += lanes) ++my_items;

if (my_items > 0) {
    set_flag(PIPE_V,    PIPE_MTE2, (event_t)kSlotB);   // slot 1 starts free
    set_flag(PIPE_MTE3, PIPE_V,    (event_t)kOutEv);   // outputs start free
    /* TLOAD item(lane) into slot 0 */
    set_flag(PIPE_MTE2, PIPE_V,    (event_t)kSlotA);
}

int64_t idx = -1;
for (int64_t it = lane; it < num_items; it += lanes) {
    ++idx;
    const int32_t slot   = (int32_t)(idx & 1);
    const int32_t cur_ev = slot ? kSlotB : kSlotA;
    const int32_t nxt_ev = slot ? kSlotA : kSlotB;

    // 1. issue the NEXT DMA first, so it overlaps this item's Vec work
    if (it + lanes < num_items) {
        wait_flag(PIPE_V, PIPE_MTE2, (event_t)nxt_ev);   // input WAR
        /* TLOAD item(it + lanes) into the other slot */
        set_flag(PIPE_MTE2, PIPE_V, (event_t)nxt_ev);
    }
    // 2. this item's data landed earlier; take its ready token
    wait_flag(PIPE_MTE2, PIPE_V,    (event_t)cur_ev);
    // 3. and wait for the PREVIOUS item's stores to release the output tiles.
    //    The DMA above is already in flight, so this does not serialise the load.
    wait_flag(PIPE_MTE3, PIPE_V,    (event_t)kOutEv);   // output WAR

    /* convert / copy the input OUT of the slot (e.g. TCVT into working tiles) */
    set_flag(PIPE_V, PIPE_MTE2, (event_t)cur_ev);       // slot refillable NOW

    /* ... the rest of the Vec body, writing the output tiles ... */

    set_flag(PIPE_V, PIPE_MTE3, (event_t)kStoreEv);
    wait_flag(PIPE_V, PIPE_MTE3, (event_t)kStoreEv);
    /* TSTORE every output */
    set_flag(PIPE_MTE3, PIPE_V, (event_t)kOutEv);       // outputs free once retired
}

// Exactly one free token per slot is outstanding for EVERY trip count, plus one
// output token. Drain them: do NOT leave tokens outstanding at kernel exit.
if (my_items > 0) {
    pipe_barrier(PIPE_ALL);
    wait_flag(PIPE_V,    PIPE_MTE2, (event_t)kSlotA);
    wait_flag(PIPE_V,    PIPE_MTE2, (event_t)kSlotB);
    wait_flag(PIPE_MTE3, PIPE_V,    (event_t)kOutEv);
}
```

**Release the input slot as early as it is genuinely dead** -- at step 3 above, right
after the conversion reads it, not at the end of the iteration. That is what gives the
next DMA a full Vec body to land in.

### Size the tile for the second slot AT DESIGN TIME, not when optimizing

Double buffering costs **one extra input slot in UB**. Decide that when you pick the
tile, because retrofitting it is expensive:

* Here the input tile was aliased over two working fp32 tiles, leaving 3.5 KB spare of
  the 90,624 B per-sub-block footprint. A second slot did not fit at 4 rows/item.
* Dropping to 2 rows/item made room but **cost 1.120x -> 1.327x on its own**, because
  the per-item DMA halved from 32 KB to 16 KB on a kernel that is ~95% memory
  movement. The prefetch then repaid that and more (1.224x over its own control), but
  the whole detour was avoidable.

So when the stage is memory-bound, budget `2 * input_tile` from the start and pick the
largest rows-per-item that still fits, rather than the largest that fits without a
prefetch. And note the row count is not the only lever -- freeing one live working
tile buys the same room without shrinking the DMA.

**Always measure the control.** If an optimization also changes a structural
parameter (rows per item, tile width, core count), build the variant that changes only
that parameter and time it too. Without variant B above, "prefetch made it 1.084x"
looks like a small win over 1.120x; with it, the prefetch is worth 1.224x and the
tiling change is a 1.19x regression that had to be paid for. Those are different
engineering conclusions.

---

## COOK-§7: L0 Ping-Pong For Cube GEMM Slices

When a `K` dimension is split into repeated 64-wide pieces.

```cpp
template <int M, int N, int K>
AICORE inline void MatmulL1(TileAcc<float, M, N, M, N>& dst,
                            L1Mat<half, M, K>& a_l1,
                            L1Mat<half, K, N>& b_l1,
                            bool init) {
  constexpr int KStep = 64;
  constexpr int Parts = K / KStep;
  constexpr uintptr_t AStepBytes = M * KStep * sizeof(half);
  constexpr uintptr_t BStepBytes = KStep * N * sizeof(half);

  TileLeft<half, M, KStep, M, KStep> a_l0[2];
  TileRight<half, KStep, N, KStep, N> b_l0[2];
  TASSIGN(a_l0[0], static_cast<uintptr_t>(0));
  TASSIGN(a_l0[1], AStepBytes);
  TASSIGN(b_l0[0], static_cast<uintptr_t>(0));
  TASSIGN(b_l0[1], BStepBytes);

  SetFlag<PIPE_M, PIPE_MTE1>(0);
  SetFlag<PIPE_M, PIPE_MTE1>(1);

  for (int part = 0; part < Parts; ++part) {
    const int buf = part & 1;
    WaitFlag<PIPE_M, PIPE_MTE1>(buf);

    TEXTRACT(a_l0[buf], a_l1, 0, part * KStep);
    TEXTRACT(b_l0[buf], b_l1, part * KStep, 0);

    SetFlag<PIPE_MTE1, PIPE_M>(buf);
    WaitFlag<PIPE_MTE1, PIPE_M>(buf);

    if (init && part == 0) {
      TMATMUL(dst, a_l0[buf], b_l0[buf]);
    } else {
      TMATMUL_ACC(dst, dst, a_l0[buf], b_l0[buf]);
    }

    SetFlag<PIPE_M, PIPE_MTE1>(buf);
  }

  WaitFlag<PIPE_M, PIPE_MTE1>(0);
  WaitFlag<PIPE_M, PIPE_MTE1>(1);
  pipe_barrier(PIPE_ALL);
}
```

Use when:
- the stage has a repeated Cube matmul inner loop
- `K` is large enough that one-shot extract/compute is not ideal

---

## COOK-§8: Two-Slot Cube/Vec Workspace Pipeline

When Cube produces a workspace that Vec consumes, with overlap across iterations.

Bootstrap rule — ALWAYS do this before any `wait_flag_dev()`:

```cpp
// Before the first iteration's wait on a workspace-free flag,
// emit the initial producer-side free-slot signals.
set_cross_flag<PIPE_MTE3>(WorkspaceFree0, 2);
set_cross_flag<PIPE_MTE3>(WorkspaceFree1, 2);
```

NEVER do this (no producer has set the flag before the wait):

```cpp
wait_flag_dev(ReadyFlag);  // WRONG: first-iteration wait with no prior producer
```

Protocol skeleton:

```cpp
constexpr int32_t StageCount = 2;
constexpr int32_t WorkspaceSlotElems = 128 * 128;
constexpr int32_t WorkspaceElems = StageCount * WorkspaceSlotElems;

// Bootstrap: signal both slots are free before first consumer wait.
set_cross_flag<PIPE_MTE3>(FREE_FLAG_0, 2);
set_cross_flag<PIPE_MTE3>(FREE_FLAG_1, 2);

for (int iter = 0; iter < num_iters; ++iter) {
  const int slot = iter & 1;
  const int free_flag = (slot == 0) ? FREE_FLAG_0 : FREE_FLAG_1;
  const int ready_flag = (slot == 0) ? READY_FLAG_0 : READY_FLAG_1;

  // Wait for slot to be free
  wait_flag_dev(free_flag);

  // Produce: write to workspace slot
  // ... Cube/Vec produce data into workspace[slot] ...

  // Signal: slot is ready for consumer
  set_cross_flag<PIPE_MTE3>(ready_flag, 2);
}
```

---

## COOK-§8.5: Proven Cube/Vec Pipeline Tile Types (Hard Layout Rules)

When the stage requires Cube+Vec cooperation, use EXACTLY these type templates.
These are the only proven layouts that compile on A2/A3 under `-DMEMORY_BASE`.
Do not modify BLayout or SLayout values. → PLAT-§Topology

```cpp
#ifdef __CCE_AICORE__

// L1 staging — Cube engine loads from GM into L1, then TEXTRACT to L0.
// Mat tile: BLayout::ColMajor, SLayout::RowMajor.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C,
                        pto::BLayout::ColMajor, RV, CV,
                        pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L1 transposed — for RHS operand that needs layout reversal.
// Mat tile: BLayout::RowMajor, SLayout::ColMajor.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN = pto::Tile<pto::TileType::Mat, T, R, C,
                          pto::BLayout::RowMajor, RV, CV,
                          pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

// L0A left operand: BLayout::RowMajor, SLayout::RowMajor.
template <typename T, int R, int C, int RV = R, int CV = C>
using TileLeftF = pto::Tile<pto::TileType::Left, T, R, C,
                            pto::BLayout::RowMajor, RV, CV,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L0B right operand: BLayout::RowMajor, SLayout::ColMajor.
template <typename T, int R, int C, int RV = R, int CV = C>
using TileRightF = pto::Tile<pto::TileType::Right, T, R, C,
                             pto::BLayout::RowMajor, RV, CV,
                             pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

// L0C accumulator: BLayout::ColMajor, SLayout::RowMajor.
// The 512 here differs from the library's own alias, which uses
//   using TileAcc = Tile<TileType::Acc, ..., TileConfig::fractalCSize>
// (pto_tile.hpp:1767) where `fractalCSize = 1024` and `fractalABSize = 512`
// (pto_tile.hpp:953-954). A run reported this as a bug -- "at 512 an fp32
// accumulator gets a 16x8 fractal instead of 16x16".
// PROBED, NOT REPRODUCED: `isa_probes/probe_accfrac.cpp` runs a real 128x128x128
// fp16 matmul built both ways; both give rel err 2.960e-04 against an fp64
// reference -- bit-for-bit the same answer. Both also compile (the static_assert
// accepts either constant). So 512 is kept, because it is the value every
// validated kernel here was built with, and switching a working constant on an
// unreproduced report is how regressions get introduced.
// If you find a shape where the two DIVERGE, that is a real finding -- record it.
template <typename T, int R, int C, int RV = R, int CV = C>
using TileAccF = pto::Tile<pto::TileType::Acc, T, R, C,
                           pto::BLayout::ColMajor, RV, CV,
                           pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// UB Vec row-major: BLayout::RowMajor, SLayout::NoneBox.
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C,
                       pto::BLayout::RowMajor, RV, CV,
                       pto::SLayout::NoneBox, 512, P>;

// UB Vec column-major: BLayout::ColMajor, SLayout::NoneBox.
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C,
                       pto::BLayout::ColMajor, RV, CV,
                       pto::SLayout::NoneBox, 512, P>;

#endif
```

**NEVER use these wrong layouts for Mat tiles. They will fail to compile:**
- `BLayout::RowMajor, SLayout::NoneBox` on Mat — wrong.
- `BLayout::RowMajor, SLayout::RowMajor` on Mat — wrong.
- No SLayout on Mat — wrong.

---

## COOK-§8.6: Cross-Core Sync Protocol (Cube ↔ Vec)

Cube and Vec are separate cores. They cannot access each other's UB/L1/L0.
All communication goes through **GM workspace buffers** + **FFTS cross-core flags**.
→ PLAT-§CrossCore

```cpp
// Bootstrap: signal free slots BEFORE first consumer wait.
template <pipe_t Pipe>
AICORE inline void set_cross_flag(int32_t flag, int32_t mode) {
  int config = 1 | (mode << 4) | (flag << 8);
  ffts_cross_core_sync(Pipe, config);
}

// Before entering main loop, bootstrap free flags:
set_cross_flag<PIPE_MTE3>(FLAG_FREE_0, 2);
set_cross_flag<PIPE_MTE3>(FLAG_FREE_1, 2);

// Optional: sync_all() global barrier at start/end of kernel.
sync_all();

// Producer side (Cube or Vec):
// 1. Wait for slot to be free:  wait_flag_dev(FREE_flag)
// 2. Produce data into workspace: TLOAD / compute / TSTORE
// 3. Drain this core:            pipe_barrier(PIPE_ALL)
// 4. Signal slot is ready FROM THE PIPE THAT COMMITTED THE STORE:
//      set_cross_flag<PIPE_FIX>(READY_flag, 2)   // after a Cube L0C->GM TSTORE
//      set_cross_flag<PIPE_MTE3>(READY_flag, 2)  // after a Vec/UB->GM TSTORE

// Consumer side (Vec or Cube):
// 1. Wait for data: wait_flag_dev(READY_flag)
// 2. pipe_barrier(PIPE_ALL); consume: TLOAD / compute
// 3. Signal slot is free: set_cross_flag<PIPE_MTE3>(FREE_flag, 2)

// After loop, optional global barrier:
sync_all();
```

**Why signal from the storing pipe (the load-bearing rule).** `ffts_cross_core_sync`
reaches its sync point only after all preceding ops *on the pipe it is issued from*
have completed and committed. Issuing the READY signal from the same pipe that did
the GM store is therefore what guarantees the consumer cannot observe the slot
ready before the data is actually written. A Cube result lands in GM via the
**FIX** pipe (L0C->GM `TSTORE`); a Vec/DMA result lands via **MTE3**. Signalling
READY from `PIPE_MTE3` after a Cube `PIPE_FIX` store is a real fault (it can fault
the core, e.g. aicore exc 507015, or feed the consumer stale data) and the
simulator does NOT catch it. Always match the signal pipe to the store pipe, with
a `pipe_barrier(PIPE_ALL)` drain immediately before the signal.

**Data-cache coherency (DCCI) -- do NOT bulk-flush the hand-off data; it is REDUNDANT
and will dominate wall time.** The bulk DMA path -- `TSTORE` (L0C->GM via FIX, or
UB->GM via MTE3) -> GM -> `TLOAD` (GM->L1/UB via MTE2) -- does NOT pass through the
scalar Data Cache that `dcci` manages. That cache sits on the SCALAR pipe (PIPE_S),
between GM and UB for scalar load/store only (CCE Sync Interfaces 6.6.5). So once the
READY signal is raised from the STORING pipe (the load-bearing rule above), the store
has committed and the consumer's MTE2 `TLOAD` observes fresh GM with NO flush. `dcci`
is REQUIRED only when you publish a value through a SCALAR SOFTWARE SIGNAL WORD (a
plain int read/written with scalar ld/st on PIPE_S) -- and with FFTS hardware flags you
never do (the PTO comm layer `dcci`s ONLY its int32 signal word, never bulk data).
megagdn's hand-tuned A2/A3 kernels do ZERO dcci on bulk hand-off data.

VALIDATED (chunk_o, dav-c220, HV=4 and HV=32): a bulk bidirectional `dcci` over every
[128,128] region (1024 cache lines each) on every hand-off was ~85% of in-kernel wall
time -- it made the handshake 4.5-6.5x SLOWER than a stream-serialized split launch.
Removing it was 6.6-7.3x faster AND still numerically correct and deterministic.

The NON-DETERMINISTIC, run-to-run-varying race is almost never a cache issue -- it is
IN-PLACE GM REGION REUSE: a producer overwrites a region another core still has live
across the hand-off (a static-layout aliasing bug that only SHOWS as a race because the
timing of the overwrite vs the cross-core read varies with grid occupancy; it surfaces
at high items/core, hides at ~1 item/core). Fix it with rule 3 below -- a DISTINCT GM
region per cross-core-live intermediate -- NEVER by flushing. Example: chunk_o stored
masked `Aqk` back into the same slot that still held the Cube-published `P1`; giving
`Aqk` its own region restored 30/30 determinism at HV=32 with no perf cost.

**The 3-rule cross-core hand-off recipe (correct AND fast):**
1. **Same-pipe FFTS handshake** -- signal READY from the pipe that committed the store
   (FIX for Cube/L0C, MTE3 for Vec/UB), after a `pipe_barrier(PIPE_ALL)` drain.
2. **No bulk `dcci`** on hand-off data (see above). Reserve `dcci` for a scalar signal
   word only.
3. **A distinct GM region per cross-core-live intermediate** -- never let a producer
   overwrite a region a consumer still needs across the hand-off. One [tile] slot per
   live value; bump the workspace stride rather than reusing a slot in place.

**Flag-ID hygiene.** FFTS `flagID` is in `[0,15]`. Per-slot READY and FREE flags
must be mutually disjoint, and disjoint from any global-barrier (`sync_all`) flag
IDs -- never reuse the same flagID on the same pipe/trigger without an intervening
wait. Partition the ID space (a band for slot READY/FREE, a separate band for
global barriers). **Flags 11-15 are RESERVED by the library all-core barrier**
`pto::SYNCALL<>` (`pto/common/type.hpp`: `SYNC_AIC_FLAG=11`, `SYNC_AIV_FLAG=12`,
`SYNC_AIC_AIV_FLAG=13`, `SYNC_AIV_ONLY_ALL=14`, `SYNC_FLAG_ID_MAX=16`) -- keep your
own cross-core slot flags in `[0,10]` so a hand-rolled handshake never collides with
a `SYNCALL` in the same kernel. For a plain all-core barrier prefer `SYNCALL<Mix>`
over a hand-rolled one (see SKILL §A6 and the 507015 note).

**Looped cross-core handshakes: flag-counter + both-vids discipline (validated).**
`ffts_cross_core_sync(pipe, 0x1|(mode<<4)|(flagID<<8))` and `wait_flag_dev(flagID)`
form a COUNTING SEMAPHORE per flagID (CCE Sync Interfaces): each signal INCREMENTS
the flagID counter, each `wait_flag_dev` DECREMENTS it and blocks while it is 0.
Counters PERSIST across iterations and saturate in [0,15] (overflow faults). A
one-off (niter=1) handshake tolerates a one-step imbalance; a LOOP does not -- an
imbalance deadlocks, usually on the FIRST iteration. Rules for a per-iteration
looped handshake (validated on dav-c220, single-buffered serial, niter up to 512):
1. **Balance every flagID each iteration** -- signals == waits, so the counter
   returns to its starting value. Never rely on a post-loop drain to fix an
   in-loop imbalance.
2. **Bootstrap the back-edge flag on its PRODUCER side, before the loop.** A loop
   adds a consumer->producer FREE flag; the consumer core must signal FREE once
   before the loop so the producer's first wait clears. A core's own
   `ffts_cross_core_sync` feeds the PEER's counter, never its own -- bootstrap on
   the side that produces the flag.
3. **mode 2 spans the whole Group (1 AIC + 2 AIV on dav-c220); directions are
   ASYMMETRIC:**
   - Cube->Vec is a BROADCAST: one Cube signal, each waiting AIV decrements its
     own copy -- a single waiting AIV is fine.
   - Vec->Cube is a REDUCE: EVERY participating AIV must signal, and the two AIV
     signals COMBINE into a single +1 on the AIC counter -- so the AIC waits
     exactly ONCE.
   Therefore do NOT `if (vid != 0) return;` before a mode-2 handshake (see C12):
   both AIV sub-blocks must execute every `ffts_cross_core_sync`/`wait_flag_dev`;
   gate only the DATA work to one vid when the buffer is single-owner. Silencing
   one AIV starves the Vec->Cube reduce and deadlocks immediately (even at niter=1).
4. **The AIC drains a Vec->Cube flag ONCE per iteration, not once-per-AIV**
   (waiting twice for the two-AIV reduce re-deadlocks).
5. **Order data before release:** `pipe_barrier(PIPE_ALL)` before each cross-core
   signal; signal READY from the pipe that committed the producing store
   (PIPE_FIX for L0C->GM), FREE from the pipe that committed the consuming store
   (PIPE_MTE3 for UB->GM).

A stream-serialized SPLIT launch (Cube kernel then Vec kernel, no cross-core flags)
remains a valid simpler alternative when an in-kernel handshake is not warranted.

**NEVER:**
- Use `pipe_barrier(PIPE_ALL)` as cross-core sync — it only syncs pipes within one core.
- Emit `wait_flag_dev(N)` at iteration 0 without a prior producer `set_cross_flag`.
- Signal READY from a pipe other than the one that committed the GM store (FIX for Cube, MTE3 for Vec).
- Use Vec-only `#if defined(__DAV_C220_VEC__)` when StageSpec requires `TMATMUL`.

---

## COOK-§8.6P: Fused multi-stage Cube/Vec kernel PERFORMANCE

These are general patterns for making a SINGLE-LAUNCH fused kernel (in-kernel loop over
an outer dim, multiple Cube+Vec stages) actually FASTER than the equivalent per-stage
split-launch chain. Validated on a 6-stage fused KDA kernel (dav-c220, HV=4 and HV=32),
but every rule below is algorithm-agnostic. Collapsing launch count 28->1 alone buys
NOTHING -- a stream-serialized chain already overlaps its sub-launches; the fused win has
to come from the items below.

**#0. FUSE ONLY WHEN A MEASURED WIN EXISTS -- otherwise ship the chain.** Fusion is not a
default. Launch-collapse buys nothing (above); and on A2/A3 a Cube<->Vec intermediate
CANNOT stay on-chip (GM-backed; the on-chip CV FIFO is A5-only), so the classic "keep
intermediates resident" win does NOT apply across that boundary here. Before fusing,
confirm ONE concrete lever: a same-core sub-chain that goes L1/UB-resident (#2); a
GM-heavy Cube op with a GM-light Vec partner to overlap (#9 pairing rule, proven by a #10
noop-floor probe); or a large intermediate that must be STREAMED to scale (materializing
a full [.,S,S]-type buffer overflows int32 offsets ~23k and OOMs -- only tiled/streaming
survives long context). And make each stage LEAN FIRST (#12): the gap to a hand-tuned
reference is usually a weak STAGE, not the composition -- fusing weak kernels just
serializes slow parts. Never put a grid barrier (`SYNCALL<Mix>`) on a per-tile Cube<->Vec
hand-off (#5); measure any fused build against a tuned/vendor reference, never only
against our own chain.

**1. The bottleneck is the RENDEZVOUS COUNT, not Cube compute -- diagnose it first.**
Each cross-core rendezvous costs ~2 `pipe_barrier(PIPE_ALL)` drains + an FFTS round-trip,
and they are SERIAL. Count rendezvous per loop iteration. Before optimizing, run a
**Cube-noop lower-bound diagnostic**: keep the full Vec prep + ALL handshakes but stub out
the Cube compute, and time it. If that floor is already at/over your target, Cube compute
is the minority and neither on-chip residency NOR Cube/Vec overlap can help -- attack the
rendezvous count and Vec wall-time instead. (KDA example: a Cube-noop fused kernel was
already slower than the chain; Cube was only ~18-23% of total.)

**2. Collapse Cube-only sub-chains to L1 residency (the biggest lever).** When several
consecutive Cube ops have NO interleaved Vec dependency -- iterative refinement, power
series, doubling, any repeated/chained GEMM -- never rendezvous between them. Keep their
operands L1-resident across all steps: the matmul result lands in L0C, write it back to L1
with `TMOV(Acc->Mat)` and feed it straight into the next GEMM, with NO GM store and NO Vec
handshake. This turns an N-step chain's N rendezvous into 1 (seed in, result out).
CONSTRAINT (A2/A3): `TMOV Acc->Mat` (`CheckTMovAccToMat`) REQUIRES a half/bf16 destination
-- fp32 L0C->L1 writeback is unsupported, so the resident chain runs in fp16 (check the
math tolerates it). (KDA example: a 7-step Neumann inverse `X+=X@Y; Y=Y@Y` went from 14
rendezvous to 1; `X+X@Y` done as the two-matmul accumulate `X@I + X@Y` into one L0C, no Vec
add. Total rendezvous/chunk 23->10, the single largest speedup.)

**3. Split row-parallel Vec prep across BOTH AIV sub-blocks (HalfC).** Any row-independent
Vec preprocessing over `[C, ...]` (elementwise scale/exp/cast/mask) halves its wall time if
vid0 owns rows `[0, C/2)` and vid1 owns `[C/2, C)`. TWO hard rules: (a) both vids must still
execute EVERY mode-2 cross-core signal/wait (gate only the DATA work by row range; never
`if (vid != 0) return;` around a rendezvous -- the Vec->Cube reduce needs both vids, C12).
(b) Each vid's scratch must NOT alias another vid's LIVE buffer -- give each vid disjoint
scratch or route to a provably-dead slot. Cross-vid aliasing is a nondeterministic race
(run-to-run), NOT a cache problem; fix it by layout, not by a flush.

**4. Move a serial scan onto the idle Cube core.** An associative within-tile scan written
as an N-iteration row-by-row Vec loop (N GM round-trips + N `pipe_barrier`, the S11
anti-pattern) is often expressible as ONE structured matmul and run on the otherwise-idle
Cube core. (KDA example: an inclusive prefix-sum cumsum became `gcs = L_ones @ g`, a single
lower-triangular-ones GEMM -- ~8-10% off total.) Prefer a one-shot Cube matmul over a long
Vec scan whenever the scan is a triangular/structured contraction.

**5. Heavy barriers MASK latent hazards.** A per-iteration all-core barrier
(`SYNCALL<Mix>`) or a bulk `dcci` incidentally serializes everything, hiding latent
intra-core WAR hazards and in-place GM region-reuse races. Removing them for speed EXPOSES
the real hazard as a nondeterministic, run-to-run failure. Fix it with correct ordering
(intra-core `set_flag`/`wait_flag` guards before a reload) or a distinct region (COOK-§8.6
rule 3) -- NEVER by re-adding the barrier/flush. Keep `SYNCALL<Mix>` for stage SEAMS only
(see SKILL C6), never as a per-iteration Cube<->Vec hand-off.

**6. Transpose via the matmul's ZN reshape, never a Vec TTRANS, in a fused kernel.** To
feed `A^T` into a GEMM, store `A` in its natural row-parallel layout and transpose INSIDE
the matmul via the left-operand `L1MatZN` + `TRESHAPE` tile reshape. A Vec `TTRANS` couples
rows (defeats a vid-split), needs a 3rd UB temp, and that temp can force a resident-state
tile to spill+restore through GM (2 extra round-trips/iteration). The in-matmul transpose
is free. (KDA example: krest^T fed to a Cube GEMM via `gemm_lt` ZN-transpose instead of a
Vec TTRANS -- removed the transpose temp + the resident-S spill, ~1.08x, and unblocked the
Stage-4 vid-split.)

**7. Run-ahead rendezvous pairing.** Two independent same-core ops whose operands are all
already staged and whose results are consumed later can share ONE run-ahead handshake:
signal all operands ready, let the peer core do both, drain all results once. Fewer serial
rendezvous than one-handshake-per-op. (Neutral when the kernel is Cube-bound, but free to do
and it helps the moment Vec prep is on the critical path.)

**8. fp16 operands + fp32 accumulator -- ONLY after verifying the operand range.** A GEMM
whose operands AND result fit the fp16 range should use fp16 operands with an fp32
accumulator (faster Cube path). But VERIFY the operand range against the data first: do NOT
fp16 an operand that can overflow -- exp-scaled values are the classic trap (e.g. a gate
cumsum reaching -28 makes `exp(-gcs) ~ 1e11`, far over fp16max ~6.5e4; the fp16 operand goes
Inf and Inf*0=NaN). Keep such operands fp32 even if it costs a Cube pass. A 0/1 or
bounded-range operand (a triangular-ones scan matrix, a normalized weight) is safe fp16.
Per-K log-shift balancing does NOT rescue an operand that is intrinsically out of range.

**9. True Cube/Vec cross-op overlap -- and the PAIRING RULE that decides if it pays.** The
deepest fused-kernel lever: make Cube and Vec run CONCURRENTLY instead of ping-ponging.
Shape = split one rendezvous into **signal-now / run-independent-op / wait-later**: (1)
signal op A's operands ready (from the committing pipe), (2) run an INDEPENDENT op B in the
gap, (3) only then wait on A's result. Give the overlapped edge its OWN cross-core flag pair
(disjoint from other rendezvous + the SYNCALL 11-15 band); both AIVs run both signal and
wait (mode-2 reduce). ISOLATION IS MANDATORY: enumerate every (A reads/writes) vs (B
reads/writes) GM region pair and confirm strictly disjoint -- two ops simultaneously in
flight must never share a region. A run-to-run nondeterministic failure means a missed
alias; fix by routing the colliding scratch to a DEAD slot / a dedicated region (NEVER dcci
or a barrier).
**THE PAIRING RULE (load-bearing -- validated both directions):** overlap pays ONLY for a
**GM-heavy Cube op || GM-light (UB-resident) Vec op**. GM-heavy || GM-heavy CONTENDS for
GM/L2 bandwidth and is net-NEGATIVE even when correct; a compute-light L1-resident Cube op
has no latency worth hiding. So before overlapping an edge, classify both sides' GM traffic;
only pair a bandwidth-bound Cube GEMM with a register/UB-bound Vec op. (KDA example:
kv-GEMM || Vec state-decay overlapped for a small robust win; overlapping the GM-heavy
Neumann doubling with the GM-heavy Stage-3 Vec prep was made correct+deterministic via region
isolation but REGRESSED 8% from contention -- a correct overlap that does not pay. If no
GM-light Vec partner exists for a GM-heavy GEMM, that edge stays serial -- this is the floor
that keeps a rendezvous-bound kernel above a hand-tuned one without a deeper restructure.)

**10. Probe the OTHER core's noop floor BEFORE building any overlap pipeline.** Before
investing in a cross-op or cross-item Cube||Vec software pipeline, measure the single-core
floor of the core you are NOT trying to hide. Build a `-D<OTHERCORE>_NOOP` variant that
keeps every cross-core handshake (`wait`/signal stay, so the schedule is intact) but
early-returns from that core's compute BEFORE its `TLOAD`/`TMATMUL`/`TSTORE`. That core is
now free and never stalls the other, so the run time IS the bottleneck core's intrinsic
floor. ANY double-buffered overlap schedule is bounded below by `max(coreA_floor,
coreB_floor)`. If that bound already exceeds your target, **pipelining cannot win** -- stop
and attack the bottleneck core's intrinsic per-item cost instead (fewer GM round-trips via
UB-resident prep ON THE EXPOSED CRITICAL PATH, fused elementwise passes -- but NOT by
narrowing `PIPE_ALL` drains; see #11). A cheap probe falsifies an overlap plan for a fraction of the build
cost; run it FIRST. (KDA example: at HV=32 seq1024 the Cube-noop Vec floor was ~1805us,
already above megagdn's ~1471us with Cube only ~17% of wall -> Vec-prep-bound, not
overlap-bound; the real lever was cutting the Vec floor's ~32 PIPE_ALL drains + ~73 GM
round-trips/chunk, not any pipeline.)

**11. Cutting a Vec/scalar-core floor: what actually helps (and what does NOT).** Once a
core is bound by sync drains + GM traffic (not ALU), the effective levers are narrow:
- **Do NOT narrow `pipe_barrier(PIPE_ALL)` into `set_flag`/`wait_flag` on a near-idle store
  path.** On A2/A3 `PIPE_ALL` is a near-free HW barrier when the pipes are already drained;
  an explicit-event flag has HIGHER fixed latency, so the "narrow it" instinct REGRESSES
  (~+70us measured on the Vec store path). Explicit narrow flags pay off only on the
  genuinely-busy core (e.g. overlapping a long Cube GEMM), never as a blanket replacement.
- **UB-resident prep helps ONLY on the EXPOSED critical path** (the serial gap between two
  rendezvous). A GM park that is already hidden under adjacent rendezvous latency costs
  nothing to keep, and forcing it resident can SERIALIZE worse. Diagnose per-block with an
  interleaved Cube-noop A/B before keeping a residency change; block-by-block park removal
  plateaus once the off-critical-path parks are all that's left.
- **A width-changing `TCVT` (e.g. fp16->fp32) with ALIASED src/dst self-corrupts** -- the
  growing element width overruns the source mid-op. Keep src and dst in DISJOINT UB regions.
- **The structural ceiling is the LAYOUT, not the local ops.** A tight per-chunk loop that
  parks every intermediate to GM (e.g. ~30 `PIPE_ALL` + ~70 GM round-trips + ~15
  rendezvous/chunk) cannot be filed down to a hand-tuned kernel that processes work items in
  BULK PER STAGE with ~9 lifetime-aliased named UB sub-buffers fully resident, parking only
  the few cross-core slots to GM (~2 `PIPE_ALL` + ~8 GM copies + ~4 hand-offs/STAGE). Local
  park-elimination plateaus; closing the rest needs adopting the bulk-per-stage
  full-UB-residency layout wholesale -- a high-risk rewrite, weigh it against the gap.

**12. Before ANY fusion work, check the floor = sum of per-stage COMPUTE -- the gap to a
hand-tuned kernel is often a single weak STAGE, not the fusion.** Fusion (any topology) only
removes launch/dispatch overhead and inter-stage GM round-trips; it makes no individual stage
faster. So the floor of any fused kernel >= sum of the per-stage compute times. MEASURE each
stage standalone and sum it FIRST. If that sum already exceeds your target, no fusion layout
can win -- and if one stage dominates the sum, that stage's ALGORITHM is the lever, not the
fusion. (KDA example: our 6 stages sum to ~2918us at HV=32 s1024 -- already above megagdn's
1552us WHOLE kernel -- because our inversion stage alone is 901us = 58% of megagdn's entire
runtime. We used 7-step Neumann doubling = 14 full [128,128] GEMMs/chunk; megagdn uses a
block-recursive fractal tri-inverse that is several-fold cheaper. No seam/barrier/residency
rearrangement recovers that; the fix is a better inversion KERNEL.)
**Two fusion topologies and WHEN each wins:** *stage-major* (megagdn -- each stage a bulk
pass over all items, ~6 stages sequential in one launch, SYNCALL only at seams, recurrence
loop inside the chunk_h/chunk_o stage with S resident) amortizes launch/seam overhead at
large work/seq but inherits each stage's full cost with ZERO cross-stage Cube/Vec overlap
(every seam is a hard barrier) -- it pays ONLY if the per-stage kernels are already near
optimal. *Chunk-major* (one outer loop runs all stages per chunk, S on-chip across the loop)
has lower fixed overhead -> wins at small seq/HV and when per-stage kernels are weak (it never
pays the bulk-stage-sum). Pick stage-major only after the per-stage sum says it can win.

**13. Triangular inverse: block-recursive fractal, NOT full Neumann doubling.** To invert a
unit-(lower/upper)-triangular `M = I + L` ([N,N], L nilpotent), do NOT run Neumann doubling on
the full N (`X+=X@Y; Y=Y@Y`, log2(N) steps x 2 DENSE [N,N] GEMMs = ~14 dense GEMMs at N=128).
Use the block-recursive fractal inverse (megagdn `runKernelTriInvRecUnroll`):
- **Phase A (diagonal fractal):** partition into F×F diagonal blocks (F = the cube's fractal
  size, 16 for fp16 on A2/A3). Invert ALL N/F diagonal blocks at once via Neumann doubling
  RESTRICTED to the block diagonal -- each step is one block-diagonal GEMM whose useful work
  is only the tiny F×F products (a few % of a dense GEMM). log2(F) steps.
- **Phase B (recursive assembly, F->2F->...->N):** combine size-b inverses into 2b via
  `inv([[A,0],[C,D]]) = [[A^-1,0],[-D^-1 C A^-1, D^-1]]`, vectorized across block pairs = 2
  block-sparse off-diagonal GEMMs per level, log2(N/F) levels.
Several-fold fewer cube-cycles than full Neumann, runs as a single persistent Cube kernel (no
Vec ping-pong, no per-step GM round-trip). fp16 operand / fp32 accumulate keeps the
(numerically sensitive) inverse accurate. VALIDATED: 5.9x faster than full Neumann at N=128
(895->152us), frob 1.4e-4. CAVEAT (the #10/#12 lesson again): a faster inverse only helps the
WHOLE kernel if the inverse is on the EXPOSED critical path -- in a chunk-major fused kernel
that overlaps the Cube inverse under the Vec floor, the inverse is already hidden, so a 5.9x
faster inverse moved the fused total by ~0. Speed the stage that is exposed, measured by a
noop probe (#10), not the stage that looks biggest in isolation.

**14. The per-chunk SLOPE is the production metric; lean named-UB residency is how you cut
it -- but it is bounded by the UB budget.** At large T (production), fused-kernel latency =
slope·num_chunks + intercept. Small-seq benchmarks measure mostly the intercept (fixed
overhead) and can show a generated kernel "winning" while it loses 3x at production T. ALWAYS
fit the slope across >=2 T values (e.g. T=1024 and T=4096) -- that is the number that matters.
A hand-tuned kernel's edge is almost entirely a leaner slope (megagdn ~90us/chunk vs a
generated ~287), coming from a lean Vec layout: a fixed map of NAMED, lifetime-aliased UB
sub-buffers (small row-split half-tiles, sequential-lifetime aliasing, cast-in-place) that
keeps every working value UB-resident across the chunk body and parks to GM ONLY the few
cross-core hand-off slots (~8-15 GM round-trips/chunk vs a naive ~70).
- **The residency-pays test (sharper than #11):** making a value UB-resident wins ONLY when
  it removes a REDUNDANT RELOAD/RECOMPUTE (a value read or recomputed N times). It LOSES when
  it trades a pipeline-overlapped DMA park for a serialized resident-ALU chain -- so a park
  that is already latency-hidden should stay a park. Verify with a tight paired A/B.
- **The UB-budget wall (the hard limit):** any full-[tile,tile] phase sitting between two prep
  stages occupies the UB those stages would share, COLLAPSING the residency of everything they
  pass through it. A lean slope requires EVERY phase be half-tile / row-split / streaming;
  one full-[128,128] "finish" block per stage is enough to pin the slope far above hand-tuned.
  Reaching the lean slope is therefore a WHOLE-KERNEL half-tile restructure, not a per-block
  residency tweak -- high-risk for cross-core determinism; scope it against the gap.
- **Measurement on a drift-prone device:** trust ONLY a tight within-process paired A/B
  (alternating A,B per rep, same process) -- a non-paired single-session median can be off by
  a large factor from device slow-drift (observed a false "950us faster" that was the baseline
  inflated, not the variant improved). Re-confirm every win paired.

**15. Row-split finish blocks + the chunk-level double-buffer that a RECURRENCE blocks.** To
row-split a [C,C] post-GEMM "finish" block (mask/scale/combine) across both AIV sub-blocks
(each vid owns C/2 OUTPUT rows in a [HalfC,C] half-tile): both vids hold all columns, only
output rows split; a strict-lower mask for global rows [rb,rb+HalfC) is `TTRI(diagonal=rb-1)`
on the local tile -- the diagonal offset absorbs the global row shift. Correct + deterministic.
BUT (the load-bearing negative result): row-splitting a finish block forces BOTH vids into
that block's cross-core rendezvous, which SERIALIZES the consuming Cube op (it must wait for
both vids' half-tile stores). In a SINGLE-BUFFERED kernel every such rendezvous is exposed and
the cost compounds with the outer loop count -> a large-T REGRESSION even though it lowered
the cube-noop Vec floor. The win only materializes if paired with CHUNK-LEVEL DOUBLE-BUFFERING
(slot = ci&1) so chunk N+1's Vec prep hides chunk N's Cube finish (how megagdn reaches ~73-90
us/chunk vs a generated ~250-290). THE CATCH: a kernel with a loop-carried RECURRENCE (S_{n+1}
depends on S_n, held in-place) CANNOT double-buffer the recurrent state, so the recurrence
stages stay single-buffered and their finish rendezvous stay exposed. Only the NON-recurrent
operands can be prefetched N+1-ahead. Net: a hand-tuned kernel's leaner slope comes from BOTH
a leaner Vec floor AND chunk double-buffering; for a recurrent algorithm the second is
partially blocked, so matching the hand-tuned slope requires reproducing its whole lean Vec
micro-architecture (the floor) AND double-buffering every non-recurrent operand -- a wholesale
clone, best-case ~parity. Quantify the per-chunk slope gap FIRST (#14) and decide if parity is
worth a clone before starting.

**16. Cargo-cult GM commits + per-item mask rebuilds are pure SLOPE -- the biggest, most
common generated-kernel inefficiency.** A generated stage kernel routinely (a) "commits" a
Vec result to GM via a `TSTORE`->`TLOAD` round-trip and (b) inserts `TMULS(x,x,1.0f)`
"push-to-pipeline" no-ops between DEPENDENT Vec ops -- believing both are needed for
buffer-stability/correctness (a C19/C31 cargo-cult). For a Vec->Vec dependency neither is
needed: a single `pipe_barrier(PIPE_V)` between the producer and consumer is sufficient and
gives BIT-IDENTICAL output. **Now confirmed by direct probe and generalized: the
`TMULS(x,x,1.0f)` push is unnecessary after a `TLOAD` too, provided the `MTE2 -> V`
handshake is present -- and if that handshake is MISSING the push makes the kernel
wrong on every run rather than some. See the corrected C27.**

**BUT there is one barrier you must NOT remove, and removing it is invisible at the
shape most kernels are validated on.** The end-of-item `pipe_barrier(PIPE_ALL)` is
NOT cargo-cult: it retires an **output WAR** as well as the input one -- the next
item's Vec writes land on the output tiles while the previous item's MTE3 stores are
still reading them. With **one item per lane there is no next item**, so the kernel is
correct at `items_per_lane <= 1` and silently wrong from 2 upward. A stage validated
only at its production shape can therefore pass while carrying the bug.

If you remove it, replace it -- do not just delete it. Carry an explicit
`MTE3 -> V` token on the output tiles (COOK-§6.7 has the complete loop and the
two-hazard table), and validate at `items_per_lane >= 3` per artifact rule 31, which
exists for exactly this failure mode: three, not two, so the ring must WRAP and
re-enter steady state rather than just run prologue-then-epilogue.

The GM round-trip is pure per-item slope. Reserve GM commits /
explicit flags for GENUINE CROSS-ENGINE boundaries only (Vec<->Cube via PIPE_FIX/PIPE_MTE3,
or Vec<->MTE). Second lever: HOIST any item-INDEPENDENT tensor (a strict-lower/causal mask,
a constant) OUT of the work-item loop and keep it UB-resident -- rebuilding a 0/1 `TTRI` mask
per item is pure slope (build it once, keep resident). VALIDATED: kkt stage 84->21 us/chunk
(3.98x, accuracy identical 3.59e-4, bit-exact) -- the GM-commit removal was the single biggest
lever, and this is the same class of inefficiency that makes a generated per-stage kernel
~3-5x slower than a hand-tuned one (the whole production gap is per-stage Vec compute, not
fusion). AUDIT EVERY generated stage for cargo-cult commits before blaming the algorithm.
Gotchas surfaced: `TROWEXPAND` requires a RowMajor **dst**; the **src may be ND or DN** (see `COOK-§6.9` -- the earlier "RowMajor src AND dst, a ColMajor column gives nan" wording here was WRONG);
`TTRI` is validated on fp32, not fp16 (build the mask in fp32, then `TCVT` to a resident fp16).

**17. A per-row / per-scan-step GM round-trip inside a within-chunk reduction or scan is the
single biggest slope inflation -- collapse it to a block-resident pass.** A generated
cumulative-scan / reduction stage often loads ONE row, scans it, stores it, per row -- N GM
round-trips for an N-row block. Instead load the WHOLE `[block, dim]` block to UB once, scan
IN-UB with a `pipe_barrier(PIPE_V)` between steps (the running accumulator is one resident
`[1,dim]` tile), and do ONE bulk store. VALIDATED: gate_cumsum 51.5 -> 6.2 us/chunk (8.4x),
bit-identical. **Critical companion GOTCHA:** even with the output parked in a separate UB
slot, you still need a V->MTE2 fence (`set_flag(PIPE_V, PIPE_MTE2)`) before the NEXT group's
`TLOAD` -- missing it is a seed/head-dependent race (frob 0.3-0.6) that only manifests when
`block_dim < total_groups` (most cores process >1 group). **Corollary (extends #16 to
transpose):** a `TTRANS` reading a freshly-produced UB tile needs NO "commit the buffer" GM
round-trip -- `pipe_barrier(PIPE_V)` + disjoint src/dst/tmp UB slots is bit-identical.
NOTE the de-inflation campaign's limit: stages that are GENUINELY load/store-bound (a split
prep->GEMM GM hand-off) or rendezvous-bound (a serial recurrence's per-chunk Cube<->Vec
handshake) are already near-minimal -- de-inflation does NOT touch them; closing THOSE needs
on-chip residency / true fusion (the structural lever), not Vec-op removal.

**18. Before building ANY fused kernel, measure Sigma(per-stage cube-noop Vec floor) -- it is
the hard floor no fusion or double-buffering topology can beat.** Build a cube-noop variant of
EACH stage (keep the handshakes, skip the Cube GEMMs) and time its Vec-prep + rendezvous. The
sum of those Vec floors is the absolute floor of any fused kernel built from those stages,
because double-buffering only hides Cube UNDER Vec -- it cannot remove Vec-prep. If
`Sigma vec_floor >= target`, the fusion build is FUTILE: stop and attack per-stage Vec-prep
leanness instead. This one measurement converts a would-be multi-day fusion build into a
one-shot check -- the highest-leverage de-risk in the playbook (the whole-stage-set
generalization of #10). (KDA: Sigma vec_floor = 134 us/chunk vs megagdn 77; wy 41.6 + chunk_h
48.7 alone exceed megagdn's ENTIRE slope, and total hideable Cube was only 33.6 -- so no
stage-major + double-buffer build could reach parity. The honest residual lever was rewriting
wy and chunk_h to a leaner Vec-prep LAYOUT, not any fusion.) Corollary: a stage whose Vec
floor ~= its real time is Vec-bound (fusion won't help it); a stage whose Vec floor ~= 0
(all-Cube, e.g. a fractal inverse) is fully hideable and free to fuse.

**19. Lean named-UB prep->GEMM layout: fold a contraction-axis scalar into the matmul OPERAND,
not into both inputs.** When a GEMM input is an elementwise reweight of a raw tensor
(`A @ (scale * B)`) and `scale` indexes the shared/contraction axis, fold the reweight into the
OTHER operand so the raw tensor reaches the Cube UNMODIFIED and direct from GM, halving the
prep->GEMM GM hand-offs. (KDA wy: U=A2@V, W=A2@K_eff with beta a per-column scale -> column-scale
the inverse `A2[r,c]=INV[r,c]*beta[c]` so V loads Cube-direct; hand-offs 8->4, Vec floor
42.3->12.4 us/chunk, 3.41x.) General lean layout: load all operands ONCE into named UB
sub-buffers, build products in UB with only `pipe_barrier(PIPE_V)` between Vec->Vec deps, and
park to GM ONLY the operands the cross-core Cube hand-off genuinely needs (the rest stay UB/L1
resident and feed the GEMM on-chip). This is a LAYOUT rewrite -- it cuts the floor of a stage
that is GM-transfer-bound (not cargo-cult-inflated; de-inflation #16 won't touch it).
GOTCHA: gathering a per-column scalar for `TCOLEXPANDMUL` via a `[1,C]` ND load with non-unit
inner stride silently gathers WRONG (frob ~0.5, invisible at hv=1, shows at hv>=2); the reliable
gather is TLOAD into col 0 of `[C,16]` -> `TTRANS [C,16]->[16,C]` (needs the 3rd `tmp` operand)
-> use row 0.

**20. UB-resident recurrent state -- pin loop-carried state in UB, but it does NOT remove the
recurrence's rendezvous floor.** For a serial recurrence (`S_{n+1} = f(S_n, x_n)`), hold the
state `S` in a NAMED UB tile and update it IN PLACE across the loop; keep only the single
irreducible cross-core store (the `S`->Cube transit for the next GEMM). Cuts the per-iteration
`S` GM round-trips (KDA chunk_h: 3->1, Vec floor 41.6->34.0). Budget it (a [128,128] fp32 `S`
is 64KB of 192KB UB) by pushing the transpose off the Vec core (Cube ZN `TRESHAPE`+`TEXTRACT`
-- a strided transposing L1 GM load produces ZEROS) and avoiding in-place fp16->fp32 widening
casts under tile pressure (multiply in fp16, widen into a disjoint tile). CRITICAL CAVEAT:
residency removes the GM-round-trip part of the floor but NOT the rendezvous serialization --
a recurrence is SINGLE-BUFFERED (loop-carried state can't double-buffer), so its floor is
bounded below by the per-iteration Cube<->Vec rendezvous, not by memory traffic. Closing THAT
needs chunk-level double-buffering of the NON-recurrent operands (prefetch iter n+1's
independent inputs while iter n's Cube runs) -- the only part the dependency permits. So the
recurrence stage is the irreducible long-tail of a fused kernel's slope.

**21. Compose lean stages into a fast fused kernel: LEAN-THEN-COMPOSE, not merge-then-tune.**
The whole-project synthesis. To match a hand-tuned fused kernel, do NOT start by merging stages
into one in-kernel FFTS-stitched body (that hits the rendezvous-serialization + UB-budget walls
#11/#15). Instead:
1. **Lean-ify each stage STANDALONE first** (#16-20: drop cargo-cult GM commits, block-resident
   scans, resident recurrent state, fold contraction-axis scalars into the matmul operand). The
   per-stage Vec floor -- not the fusion -- is the production slope (#12/#18).
2. **Make every stage share ONE layout (e.g. BSND)** so the seams need no repack -- run shared
   Cube sub-kernels (e.g. the fractal tri-inverse) in their NATIVE shared-layout mode.
3. **Compose by namespace-`#include` + chaining each stage's `launch_*` in ONE host
   `call_kernel`** -- STREAM ORDERING is the free seam barrier (no `SyncAll<Mix>` needed). The
   fused slope = the EXACT SUM of the lean per-stage slopes, zero fusion penalty, plus the
   host-dispatch-collapse win over a separate-launch chain. (KDA: slope 103.7 us/chunk, beats
   the chain ~1.34x and the in-kernel-FFTS chunk-major v8 2.5x; 0.72x megagdn at T=1024 -- we
   win small -- and only 1.08x at T=4096; correct, 30/30 deterministic.)
4. **Tune `block_dim` below the core-count cliff** (a sharp latency cliff sits just above the
   balanced point; KDA bd 48->46 = -15 us/chunk).
5. **Reserve a single-launch in-kernel FFTS MERGE only for a recurrent stage** where chunk-level
   double-buffering is the sole remaining lever -- and only when its rendezvous is hideable (for
   a serial loop-carried-state recurrence it is NOT, so that stage is the irreducible slope
   floor; KDA chunk_h = 43.5 of the 103.7, the residual to megagdn's 76.5).

**22. Recurrence double-buffering: run-ahead the NON-recurrent operand prep into the Cube gap;
the recurrent state stays single-buffered in-place.** A serial-state recurrence stage still has
partial overlap headroom: split its per-chunk Vec prep into (a) RECURRENT ops that need the
Cube result or the loop-carried state (`v_corr = u - w@S`, the `S`-update) and (b) NON-recurrent
ops that need only the raw chunk inputs (`k_rest = k*exp(g_total-g_cs)`). Hoist (b) to AFTER the
cross-core signal but BEFORE blocking on the Cube result, so Vec preps chunk n+1's (b) while
Cube finishes chunk n. No `ci&1` GM slot double-buffering is needed when (b) feeds the SAME
chunk's later GEMM (its hand-off slot isn't reused until the next chunk). The recurrent state
stays in-place single-buffered (#20). VALIDATED: KDA chunk_h real slope 45.5->39.4 us/chunk,
folded fused 103->97.6, reaching megagdn PARITY at T=4096 (1.017x), faster at T=1024. Zero
repairs (the trailing `pipe_barrier(PIPE_ALL)` already drains the run-ahead store).
**Companion diagnostic (inverse of the de-inflation signature):** if the cube-noop Vec floor
stays FLAT while the real slope drops, the win is genuine Cube/Vec OVERLAP (latency hiding),
not Vec-op reduction (#16); if both drop together, it was op-count. CAVEAT: the serial
v_corr + S-update remain exposed (~the residual above the Vec floor) -- that is the irreducible
recurrence floor; the further lever is a 2-vid HalfC split of the recurrence's Vec work (both
AIV subblocks each own half the state rows -- a stage running all work on vid==0 leaves half the
Vec cores idle), which is orthogonal/additive to this run-ahead.

**23. 2-vid HalfC recurrence split + cross-vid coherence via cheap `dsb`, NOT bulk `dcci`.**
Split a recurrence stage's per-iteration Vec work across BOTH AIV sub-blocks (each owns HalfC =
half the state rows): the S-decay, the non-recurrent operand build, v_corr, snapshot all run at
HalfC on both vids instead of full-C on vid0 only -- ~halves the per-vid Vec compute INSIDE the
rendezvous gaps (does NOT reduce the rendezvous count). VALIDATED: KDA chunk_h 39.4->26.7
us/chunk (-32%), bit-identical math. Two non-obvious requirements:
1. **UB is SHARED between the two AIV sub-blocks** -- partition it into DISJOINT per-vid windows
   (reusing addresses aliases and corrupts); budget tightly (e.g. 2x96KB).
2. **Cross-vid disjoint-half-buffer coherence is fixed by a CHEAP producer-side `dsb(DSB_DDR)`
   before the FFTS signal -- NOT per-cache-line `dcci`.** Full `dcci` ranges fix coherence too
   but cost +27-40 us/chunk, ERASING the entire split win; a single `dsb` barrier is ~free and
   sufficient. (This is the cheap coherence primitive the earlier no-bulk-dcci rule #16/§8.6
   was missing for the cross-VID case.)
DEPLOYMENT WALL (the decisive, corrected finding): a 2-vid recurrence built on a per-HEAD
fp32 GM workspace with CROSS-CORE disjoint-half writes has an IRREDUCIBLE cross-core
GM-coherence race -- it intermittently hard-aborts (all-zero/NaN, nondeterministic) and NO
`dcci`/`dsb`/bootstrap-drain variant reliably closes it (targeted half-dcci even regresses --
the two vids' adjacent-half writes race at the shared boundary cache line). The race is
OCCUPANCY-INDEPENDENT (not a full-core-count wall) and is exposed by re-allocating GM inputs
per repeat (a fixed-alloc harness hides it -- always fault-test with FRESH GM allocation).
The structural fix is how a hand-tuned kernel avoids it entirely: a per-CORE workspace
(`cid*WS_PER_CORE`) so Cube<->Vec sharing is INTRA-core only, NEVER cross-core GM, with fp16
hand-offs and a wave-loop flag-balance -- and ZERO dcci/dsb. So a 2-vid recurrence must be
designed per-core-workspace from the START; retrofitting cross-core coherence onto a per-head
layout is a dead end. (KDA: the 2-vid chunk_h hit 26.7 us/chunk math-correct but could not be
made HV=32-deterministic; deploying it = porting megagdn's per-core-workspace structure
wholesale. The deployable result stayed the per-stage-lean fused kernel at parity, not the
2-vid beat.)

---

## COOK-§8.7: One-Shot GEMM Pattern (Single K Block)

When the K dimension fits in one L0 tile (K ≤ 128), use this direct pattern.
This is the simplest Cube path and should be the first choice for small-K stages.

**CRITICAL — TEXTRACT feed compatibility rule:**

The TEXTRACT source tile layouts determine which L0 destination they feed:

```
Source L1 tile                     → TEXTRACT destination    → TMATMUL operand
────────────────────────────────────────────────────────────────────────────
L1Mat  (BLayout::ColMajor,        → TileLeft<half, M, K>    → Left operand
        SLayout::RowMajor)

L1Mat  (BLayout::ColMajor,        → TileRight<half, K, N>   → Right operand
        SLayout::RowMajor)                                     (NO transpose)

L1MatZN (BLayout::RowMajor,       → TileRight<half, K, N>   → Right operand
         SLayout::ColMajor)                                    (WITH transpose)
```

**TMOV vs TEXTRACT for L1→L0 data movement:**
Both `TMOV` and `TEXTRACT` can move data from L1 (MatTile) to L0 (LeftTile/RightTile).
In this workflow, **always use `TEXTRACT`** for L1→L0 transfers in Cube GEMM patterns.
`TEXTRACT` is the compile-proven surface under `-DMEMORY_BASE` and is the only
instruction used in the proven cookbook patterns above. `TMOV` for L1→L0 may appear
in other PTO documentation or auto-mode code, but is not part of the approved
surface here.

Every reference kernel follows this exact mapping for the NON-transposed feed.

> **CORRECTION -- "L1MatZN into TileLeft will fail with `static_assert`" is FALSE.**
> A transposing extract exists in the pinned tree: `TExtractToATranspose`
> (`pto/npu/a2a3/TExtract.hpp:28`, dispatched at :99, plus a `...Compact` variant at :235).
> So a transposed left operand does **not** require a separate dual-layout staging scheme.
> One campaign run built an entire `pT`/`dsT` dual-layout design around the supposed absence
> of this path before finding it. Check `TExtract.hpp` before designing around a transpose.
>
> The mapping below is still the right default; it is the *impossibility* claim that was
> wrong.

**The table above is HARDWARE-VERIFIED.** `isa_probes/probe_accfrac.cpp` stages both
operands through `L1Mat` (`SLayout::RowMajor`) and extracts the right one into a
`TileRight` (`SLayout::ColMajor`), then checks the product against an fp64 reference:
the answer is **`A @ B`** at rel err 2.960e-04, while `A @ B^T` is off by 1.399.
So `L1Mat -> TileRight` is the **NO-transpose** path, exactly as the table says.

A run reported the table as "backwards" on the theory that the library selects
`Transpose = (Dst::SFractal != Src::SFractal)`. **Not reproduced** -- the SFractals
do differ on that path (RowMajor vs ColMajor) and no transpose occurs. Do not
"fix" this table.

**What that run actually hit is the SNIPPET BELOW, which is wrong.** It performs
the *transposed* feed (`TRESHAPE` to `L1MatZN`, then `TEXTRACT`) unconditionally,
in a section whose table is about the untransposed case. Follow the snippet blindly
for a plain `C = A @ B` and you get `B^T`. Pick the branch that matches your maths:

```cpp
{
    TileLeft<half, M, K, M, K> _l0a;
    TileRight<half, K, N, K, N> _l0b;
    TASSIGN(_l0a, 0x0);
    TASSIGN(_l0b, 0x0);

    auto _we = EVENT_ID1;
    set_flag(PIPE_MTE2, PIPE_MTE1, _we);
    wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
    set_flag(PIPE_M, PIPE_MTE1, _we);
    wait_flag(PIPE_M, PIPE_MTE1, _we);

    TEXTRACT(_l0a, a_l1, 0, 0);

    // CHOOSE ONE -- these compute DIFFERENT products.
    //
    // (a) C = A @ B  -- the usual case. Feed the right operand straight from
    //     L1Mat. No TRESHAPE, no L1MatZN. Hardware-verified above.
    TEXTRACT(_l0b, b_l1, 0, 0);
    //
    // (b) C = A @ B^T -- only when your maths actually wants B transposed:
    //     L1MatZN<half, K, N> _bzn;
    //     TRESHAPE(_bzn, b_l1);
    //     TEXTRACT(_l0b, _bzn, 0, 0);

    set_flag(PIPE_MTE1, PIPE_M, _we);
    wait_flag(PIPE_MTE1, PIPE_M, _we);
    TMATMUL(c_l0, _l0a, _l0b);

    set_flag(PIPE_MTE1, PIPE_MTE2, _we);
    wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
    set_flag(PIPE_M, PIPE_FIX, _we);
    wait_flag(PIPE_M, PIPE_FIX, _we);
}
```

**NEVER**:
- Omit `TRESHAPE` and `TEXTRACT` directly from `L1Mat` for transposed operands.
- TLOAD data into `L1MatZN` — it's a view, not a storage tile.
- Use `SLayout::NoneBox` on Left, Right, or Acc tiles.

**sync flag protocol (one-shot) — exact sequence:**

| Phase | Flags | Meaning |
|---|---|---|
| 1 | `set(MTE2, MTE1)` → wait | Wait for TLOAD to fill L1 |
| 2 | `set(M, MTE1)` → wait | Wait for previous TMATMUL to release L0A/L0B |
| 3 | TEXTRACTs | L1 → L0A/L0B |
| 4 | `set(MTE1, M)` → wait | L0A/L0B data ready |
| 5 | TMATMUL | Compute |
| 6 | `set(MTE1, MTE2)` → wait | Release MTE1 for next L1 fill |
| 7 | `set(M, FIX)` → wait | Commit L0C result |

---

## COOK-§8.7B: L1 MACRO-BLOCKING -- reuse operands across output tiles

**Read this before shipping any dense contraction.** COOK-§8.7 and §8.8 show how to
compute ONE output tile. Followed literally for a full GEMM -- one output tile per
work item, operands re-fetched per tile -- they produce a kernel that moves **3.2x
more GM bytes than necessary**. That is not a hypothetical: it was the measured
starting point of a `quant_matmul` campaign, and closing it was worth **3.2x**, from
3.196 to 0.892 against the vendor.

The arithmetic is the whole argument. For an `[M,K] x [K,N]` GEMM tiled `MT x NT`:

```
one output tile per work item : GM reads = (M/MT)*(N/NT) * (MT*K + K*NT)
                                         = M*N*K*(1/NT + 1/MT)          <- every A
                                           tile re-read for every N tile, and v.v.
macro-block MB x NB tiles     : each A macro-row is read ONCE per macro-column
                                and each B macro-column ONCE per macro-row, so the
                                traffic falls by ~MB and ~NB respectively.
```

So: **make the work item a MACRO-BLOCK of output tiles, not a single tile.** Stage
the A macro-row and the B macro-column into L1 once, then sweep the `MB x NB` output
tiles out of L0 without touching GM again.

```cpp
// work item = one MB x NB block of output tiles
for (int32_t item = get_block_idx(); item < n_items; item += get_block_num()) {
  // stage ONCE per macro-block
  TLOAD(a_l1, a_macro_row);          // MB*MT x K
  TLOAD(b_l1, b_macro_col);          // K x NB*NT
  for (int32_t j = 0; j < NB; ++j) {         // N-MAJOR: see below
    // TEXTRACT indices are ELEMENT offsets, NOT tile indices -- see the note below.
    TEXTRACT(l0b, b_l1, 0, j * NT);
    for (int32_t i = 0; i < MB; ++i) {
      TEXTRACT(l0a, a_l1, i * MT, 0);
      TMATMUL(acc, l0a, l0b);
      TSTORE(gm_tile(i, j), acc);
    }
  }
}
```

> **`TEXTRACT`'s last two arguments are ELEMENT offsets, not tile indices.** This snippet
> previously passed the loop counters `i` and `j` directly, which extracts from element row
> `i` instead of element row `i * MT` -- **silently wrong**, with no error. Two independent
> campaign runs reported it. The implementation is unambiguous: `a2a3/TExtract.hpp` computes
> `indexRow * srcColNum * sizeof(SrcType) >> SHIFT_BLOCK_BYTE`, i.e. it scales `indexRow` by
> the source row width, and asserts 16-element alignment. Multiply by the tile extent.

**Iterate N-MAJOR, and do not assume it is a wash.** Measured on a 14-shape on-device
sweep: N-major (`2x8`) beat M-major (`8x2`) by **1.38x at IDENTICAL byte counts** --
both move 400 KB per 16 tiles. The difference is burst shape, not traffic, and it is
not derivable on paper. **Sweep the block geometry; do not reason about it.**

**A single L0C accumulator serializes the Cube.** With one accumulator the MMAD pipe
must wait for the FIXPIPE writeback of tile `t` before starting `t+1`; double-buffer
L0C so the two overlap. This was worth crossing vendor parity in the campaign above.

**Sizing.** L1 must hold `MB*MT*K + K*NB*NT` elements, so the macro-block is capped
by L1 capacity, not by preference -- derive it in your C7 budget. If it does not fit,
slice K (COOK-§8.8) *inside* the macro-block rather than shrinking the block to one
tile, which puts you straight back to the 3.2x.

**Emit an `OPTIMIZER-TARGET` marker** if you ship a dense contraction that re-fetches
an operand once per output tile. The campaign that found this shipped its baseline
with no marker at all, silently asserting no strong form applied -- an assertion that
was false and cost 3.2x.

## COOK-§8.8: K-Sliced GEMM Pattern

When K > 128, split into 128-element blocks with TMATMUL (first block) +
TMATMUL_ACC (remaining blocks).

```cpp
constexpr uint32_t kL0Size = 128;
const uint32_t kL0split = (K + kL0Size - 1) / kL0Size;

auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

for (uint32_t kL0Idx = 0; kL0Idx < kL0split; ++kL0Idx) {
    const bool initflag = clear && (kL0Idx == 0);
    const bool is_tail_block = (kL0Idx == kL0split - 1);

    if (is_tail_block) {
        // Use K_tail-sized tiles for the last partial block
        TileMatL0A<T, M, K_tail, M, K_tail> l0a;
        TileMatL0B<T, K_tail, N, K_tail, N> l0b;
        TASSIGN(l0a, 0x0);
        TASSIGN(l0b, 0x0);
        set_flag(PIPE_M, PIPE_MTE1, war_event_id);
        wait_flag(PIPE_M, PIPE_MTE1, war_event_id);
        TEXTRACT(l0a, A, 0, kL0Idx * kL0Size);
        // For transposed B: TRESHAPE to L1MatZN, then TEXTRACT
        TEXTRACT(l0b, B, kL0Idx * kL0Size, 0);  // or via _bzn
        set_flag(PIPE_MTE1, PIPE_M, war_event_id);
        wait_flag(PIPE_MTE1, PIPE_M, war_event_id);
        if (initflag) TMATMUL(C, l0a, l0b);
        else TMATMUL_ACC(C, C, l0a, l0b);
    } else {
        TileMatL0A<T, M, kL0Size, M, kL0Size> l0a;
        TileMatL0B<T, kL0Size, N, kL0Size, N> l0b;
        TASSIGN(l0a, 0x0);
        TASSIGN(l0b, 0x0);
        set_flag(PIPE_FIX, PIPE_M, war_event_id);
        wait_flag(PIPE_FIX, PIPE_M, war_event_id);
        set_flag(PIPE_M, PIPE_MTE1, war_event_id);
        wait_flag(PIPE_M, PIPE_MTE1, war_event_id);
        TEXTRACT(l0a, A, 0, kL0Idx * kL0Size);
        TEXTRACT(l0b, B, kL0Idx * kL0Size, 0);
        set_flag(PIPE_MTE1, PIPE_M, war_event_id);
        wait_flag(PIPE_MTE1, PIPE_M, war_event_id);
        if (initflag) TMATMUL(C, l0a, l0b);
        else TMATMUL_ACC(C, C, l0a, l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
        wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    }
}
set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
set_flag(PIPE_M, PIPE_FIX, war_event_id);
wait_flag(PIPE_M, PIPE_FIX, war_event_id);
```

Key: `K_tail = (K % 128 == 0) ? 128 : (K % 128)`. The tail block uses
different-sized L0A/L0B tiles and does NOT use `PIPE_FIX→M` sync.

---

## COOK-§8.9: L0C → GM Store Pattern

After TMATMUL completes, store the accumulator result to GM workspace.

**DANGER -- read C33 before using the `DYNAMIC` form below.** A boxed tile's valid
extent is honoured only within its LAST 16-row fractal: `TileAcc<float, M, N,
DYNAMIC, DYNAMIC>(valid_M, ...)` is correct **iff `ceil(valid_M/16) ==
ceil(M/16)`**, and silently wrong otherwise (probed: at `M=128`, only
`valid_M = 113..128` are correct; `valid_M = 16/32/64` are all WRONG, so this is
not a multiple-of-16 rule). Declare the tile at the size you actually use.

```cpp
{
    TileAcc<float, M, N, DYNAMIC, DYNAMIC> _l0(valid_M, valid_N);
    TASSIGN(_l0, 0);
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = valid_M; _gs.shape[4] = valid_N;
    GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, N, 1>> _gm(
        workspace_ptr + slot * workspace_slot_bytes, _gs);
    TSTORE(_gm, _l0);
}
```

Use when:
- Cube produces a result that Vec must consume (gating, masking, normalization)
- OR Cube result is the final output

**Never**: Read L0C directly from Vec — they are separate physical cores.
Always stage through GM. → PLAT-§Illegal

### COOK-§8.9B: populating the Bias table for `TMATMUL_BIAS` (the only route)

Folding the bias into the Cube accumulator removes a whole Vec stage, a GM
workspace and a cross-core handshake. The path to get a bias INTO the bias table is
documented nowhere upstream, and the two natural guesses are both wrong. All four
legs below were compile-checked against `bisheng -xcce --cce-aicore-arch=dav-c220`:

| what you might try | result |
|---|---|
| `TLOAD(bias_tile, gm)` -- load GM straight to Bias | **FAILS** `static_assert`: a TLOAD destination must be Vec or Mat |
| `BiasTile<half, ...>` -- keep the bias in fp16 | **FAILS** `static_assert(std::is_same<half,float>)`: the table is fp32, always |
| GM -> L1 `Mat<half>` -> `TMOV` -> `Bias<float>` | **COMPILES** -- and the `TMOV` performs the fp16->fp32 conversion in hardware |
| same, with the boxed `L1Mat` (ColMajor/RowMajor) source | **COMPILES** too |

```cpp
// GM(half) -> L1 Mat -> TMOV -> BiasTable(float). The TMOV converts.
L1MatND<half, 1, N> b_l1;   TASSIGN(b_l1, B_L1_ADDR);
TLOAD(b_l1, g_bias);                       // half -> half, no conversion
BiasT<float, 1, N> bt;      TASSIGN(bt, BT_ADDR);
TMOV(bt, b_l1);                            // half -> float, IN HARDWARE
TMATMUL_BIAS(acc, l0a, l0b, bt);
```

**Why this matters more than it looks.** The hardware conversion is what keeps an
fp16-bias GEMM `cube_only`. Without it the obvious readings are "the bias must
already be fp32" -- forcing a Vec pre-pass, an FFTS handshake and a GM workspace --
or "`TMATMUL_BIAS` is unusable". Two independent `grouped_matmul` runs hit this;
the one that found the conversion deleted an entire stage with it.

**The bias table is small and it silently caps your output tile.** It holds one
fp32 row, so `N <= 256` on A2/A3. That is a real constraint on the column extent of
the output tile, and it is not reported as a capacity error -- derive it in your C7
budget rather than discovering it.

Note `TMOV` here is the exception to COOK-§8.7's blanket "always `TEXTRACT`, never
`TMOV` for L1->L0": that rule is about the A/B operand feed. For `Mat -> Bias`,
`TMOV` is the ONLY path.

### COOK-§8.9Q: int32 accumulator -> fp16 with a dequant scale, FUSED into the store

For a quantized matmul (int8 x int8 -> int32, then `* scale` -> fp16) do **not**
write int32 to GM and rescale it on Vec. The fixpipe applies the scale during the
L0C->GM writeback, selected by the source/destination type pair:

```cpp
// acc is TileAcc<int32_t,...>, the GlobalTensor is <half>. The int32->fp16 type
// pair selects QuantMode_t::DEQF16, which applies the scale AND does NZ->ND in
// one pass. No int32 intermediate ever reaches GM: no second stage, no Vec work,
// no cross-core handshake, no int32-sized workspace.
TSTORE(gm_half, acc_int32, quant_pre);
```

**The `quant_pre` register encoding is documented nowhere** -- not in the ISA docs,
not in the pto-isa headers, not in the CANN AscendC headers, which expose only the
field name `deqScalar`. It is:

```
quant_pre = (1ULL << 46) | (uint64_t)(fp32_bits(scale) & 0xFFFFE000)
```

That is: bit 46 set, and the fp32 mantissa keeps only its **top 10 explicit bits**
(bits 22..13; bit 12 is always cleared). The scale register is effectively a
**19-bit float** (sign + 8 exponent + 10 mantissa).

Evidence -- three independent derivations agree:
1. Verified 800/800 against `torch_npu.npu_trans_quant_param` over random scales
   spanning 1e-3..1e3, both signs. (`npu_trans_quant_param` is a documented public
   *parameter-packing* API, so using it is not a provenance breach -- no kernel
   source is read.)
2. Independently re-derived by a later run that had no access to the first.
3. `gen_ab_quant.py` asserts both packings and the vendor's own agree bit-for-bit
   on a snapped scale, and the resulting kernel output is **bit-identical to a
   CPU-fp64 reference** at every size from M=1024 to M=8192.

**Consequence for your tolerance, and it is a trap.** The applied scale differs
from the requested fp32 scale by up to `2^-10 = 9.77e-4` relative. That is LARGER
than fp16 rounding, so a contract that says "the only error is one fp16 rounding"
is wrong. Two defensible choices, and you must say which you took:
* **truncate** (`& 0xFFFFE000`) -- matches the vendor exactly, so outputs can be
  compared bit-for-bit against `npu_quant_matmul`;
* **round to nearest-even** -- roughly halves the scale error, but then you are no
  longer bit-comparable with the vendor and must validate against fp64 instead.

Validate against a reference built with the **effective** (post-truncation) scale,
never the requested one, or you will chase a 1e-3 "bug" that is the hardware's
documented behaviour.

---

## COOK-§8.10: GM Workspace Layout

Cube and Vec exchange data through GM workspace buffers.

```cpp
// Per-core: one contiguous region per core
constexpr int32_t WS_Q   = 0;
constexpr int32_t WS_K   = WS_Q   + C * K_DIM;
constexpr int32_t WS_V   = WS_K   + C * K_DIM;
constexpr int32_t WS_PER_CORE = WS_V + C * V_DIM;

// Addressing: cid = get_block_idx()
__gm__ half* ws_ptr = workspace_handle + static_cast<int64_t>(cid) * WS_PER_CORE;
```

OR double-buffered per-core:

```cpp
int32_t slot_bytes = M * N * static_cast<int32_t>(sizeof(float));
__gm__ half* ws_slot = workspace_handle +
    (static_cast<int64_t>(cid) * 2 + slot) * slot_bytes;
```

Workspace data type: always `half` (fp16), even when Cube accumulates in float32.
TSTORE converts float → half automatically.

---

## COOK-§8.11: Data Type Casting (TCVT)

float ↔ half conversion must use TCVT, never scalar casts in device code.

```cpp
// float → half
TCVT(half_dst, float_src, pto::RoundMode::CAST_NONE);
pipe_barrier(PIPE_V);

// half → float
TCVT(float_dst, half_src, pto::RoundMode::CAST_NONE);
pipe_barrier(PIPE_V);
```

Always use `pto::RoundMode::CAST_NONE`. Always follow with `pipe_barrier(PIPE_V)`.

---

## COOK-§8.12: Complete Vec→Cube→Vec Dataflow

When Vec pre-computes coefficients that Cube uses, then Vec post-processes Cube output:

```
Vec phase:
  1. TLOAD inputs (GM → UB)
  2. Vec compute (TEXP, TMUL, TADD, TCVT, TSUB, etc.)
  3. pipe_barrier(PIPE_ALL)
  4. TSTORE pre-computed workspace (UB → GM)
  5. pipe_barrier(PIPE_ALL)
  6. set_cross_flag<V→C>  signal "workspace ready"
  
Cube phase:
  1. wait_flag_dev(V→C)   wait for workspace
  2. TLOAD pre-computed data (GM → L1)
  3. TEXTRACT + TMATMUL  (L1 → L0 → compute)
  4. TSTORE result (L0C → GM workspace)
  5. set_cross_flag<C→V>  signal "result ready"

Vec phase (post-process):
  1. wait_flag_dev(C→V)   wait for result
  2. TLOAD result (GM → UB)
  3. Vec post-process (TEXP, TADD, TMUL, TSTORE)
  4. TSTORE final output (UB → GM)
```

Flags: use cross-core flag IDs that don't collide with sync_all flags
(see COOK-§8.6). Double-buffer with `slot = ci & 1` for overlap.
Bootstrap free-slot signals before first `wait_flag_dev`.

See EX-§3 in `examples.md` for a complete working example.

---

## COOK-§8.13: Blocked / Log-Depth Triangular Inverse -> Cube

A dense unit-(lower-)triangular inverse `(I + strict_lower(M))^-1` LOOKS sequential
(forward substitution is O(BT) row-by-row), but once blocked it is dominated by
DENSE MATMULS -- so a LARGE one belongs on Cube, not a row-sequential Vec loop.
Two exact realizations for the strictly-lower / nilpotent case (`N = strict_lower`,
`N^BT = 0`; small `||N||`, e.g. the L2-normalized regime):

- **Neumann doubling (simplest when BT fits one Cube tile).** With `P = -N`,
  `inv = product_{s>=0} (I + P^(2^s))`:
  ```
  X = I;  P = -N
  repeat ceil(log2(BT)) times:      // BT=64 -> 6 steps
    X = X + X @ P                   // Cube TMATMUL (X@P), then Vec TADD (the I-term)
    P = P @ P                       // Cube TMATMUL
  ```
  Collapses the O(BT) row dependency to ~log2(BT) dense BTxBT matmuls.
- **Blocked recursion (LAPACK TRTRI).** Partition into blocks; invert small
  diagonal blocks (Neumann / forward-sub base case), then sweep off-diagonal
  blocks `inv[i,j] = -inv(L_ii) @ (sum_{j<=k<i} L_ik @ inv[k,j])` -- each
  off-diagonal update is a Cube TMATMUL. Dependency O(BT/blk) blocks.

**Dataflow (stream-serialized, no in-kernel handshake):** Vec prep -> Cube raw
GEMM (build M) -> Vec seed (`P0=-N`, `X0=I`) -> { Cube `X@P`/`P@P`, Vec `X +=` }
x log2(BT) -> Vec post-scale. Round-trip operands through GM between matmuls (C19)
and stream-serialize the steps on one stream -- no per-step Vec<->Cube cross-core
flags, so no C6 deadlock risk. Reuse a dead post-GEMM workspace region for the
P/X/TMP tiles to keep workspace bytes constant.

**ISA caveat (A2A3 / dav-c220):** `TEXTRACT` into a `TileAcc` (streaming a
GM-resident C0 into the `TMATMUL_ACC` accumulator) is **A5-only**. On A2A3 the
unit-diagonal `I +` term CANNOT ride on `TMATMUL_ACC` -- add it with a Vec `TADD`
and keep the dense products on plain `TMATMUL` (which takes the `(float,float,
float)` triple natively, no fp16 cast).

**Validated:** ~10x stage speedup vs the row-sequential Vec solve at BT=64/K=128
on real NPU (a dominant 222 ms stage -> ~22 ms at NT=512). See the size gate in S3.

---

## COOK-§8.14: Double-buffered Cube/Vec chunk pipeline (overlap)

**When to use.** A per-chunk loop where Cube produces a tile each chunk and Vec
post-processes it, and you want Cube of chunk t+1 to run concurrently with Vec of
chunk t (GEMM-bound chunked stages). NOTE: this is a CONSTRUCTION -- a single-
buffered per-chunk handshake (COOK-§8.6) is simpler and, for a GEMM-bound
recurrence, the stream-serialized host sub-launch loop is already overlapped (so
in-kernel fusion alone does not win -- see C6 performance caveat). Only build this
when you have measured that overlapping Cube_{t+1} with Vec_post_t is the lever.

**Applicability boundary (the load-bearing rule).** Pipeline overlap of ANY depth
(2-slot double-buffer or deep GM FIFO) only helps when the HEAVY producer (Cube)
is RECURRENCE-FREE -- its inputs depend only on per-iteration data, never on the
consumer's output from the prior iteration. Then Cube runs ahead and the FIFO
absorbs consumer jitter (this is exactly how flash-attention overlaps: its heavy
QK/PV Cube producer is recurrence-free; only the LIGHT Vec running-O/max/sum
update is on the serial edge). If instead the heavy Cube op READS the loop-carried
state (e.g. a recurrence chunk that starts with `W @ S_t`), the serial edge
`S_t -> Cube(W@S_t) -> Vec(...) -> Vec(S_{t+1}) -> Cube(W@S_{t+1})` means the
expensive matmul cannot start until the consumer finishes the prior chunk -- there
is no independent producer to fill a FIFO, so overlap of any depth does NOT help
(measured: a single-launch in-kernel loop regressed ~4%, and a deeper FIFO cannot
recover it). Do NOT pipeline such a recurrence; for a state-reading heavy producer
the levers are state residency (keep S in L1/L0C to drop the per-chunk GM
round-trip) and per-kernel micro-efficiency (fp16, tiling), not overlap.

**Mechanism (independent chunks).** Two GM scratch slots, `slot = chunk & 1`. Four
FFTS counting-semaphore flags: `READY[0..1]` (Cube->Vec) and `FREE[0..1]`
(Vec->Cube). FFTS flags are persistent counting semaphores (signal=+1, wait=-1);
on A2/A3 a mode-2 Cube-side `wait_flag_dev` consumes both AIV sub-blocks as ONE
decrement, so both vids must signal each FREE once per iteration (C12/COOK-§8.6).
- **Prologue:** before the loop, the consumer (Vec) signals BOTH FREE flags once
  (`set_cross_flag<PIPE_MTE3>`), so the producer may fill both slots before any
  drain -- this one-iteration head start IS the overlap.
- **Steady state:** Cube `wait FREE[slot]; produce -> TSTORE(slot); signal
  READY[slot]` (from PIPE_FIX). Vec `wait READY[slot]; TLOAD(slot); post-process;
  signal FREE[slot]` (from PIPE_MTE3, both vids). Because both FREEs were
  pre-signalled, Cube proceeds to slot (t+1)&1 while Vec drains slot t&1.
- **Epilogue:** the last Vec iteration drains the last filled slot; the two
  bootstrap FREE tokens are left unconsumed. Guard the final iteration's flag
  emissions so total signals == total waits per flag (`if (chunk+1 < n)`), then
  close with an all-core barrier on RESERVED flag IDs distinct from the data-flow
  IDs before any flag reuse.

**Loop-carried recurrent state (S) -- keep it off the critical path.** Put S on a
SEPARATE dependency edge from the ping-pong scratch (its own buffer + flag; never
route S through the ping-pong flags, or every chunk re-serializes). Vec is the
sole owner of the fp32 master S, RESIDENT IN UB (decay+add in-place via `TMUL` +
`TADD`). Keep the Cube-side S operand RESIDENT IN L1 so chunk t+1's `W@S`/`Q@S`
does not reload S from GM -- only the small KV delta crosses cores each chunk. For
maximum overlap, DOUBLE-BUFFER S (S_cur/S_nxt, swapped per chunk) so chunk t's
decay+add overlaps chunk t+1's S-independent GEMMs.

**The irreducible serial edge.** `KV_t (Cube) -> S_next (Vec) -> W@S_{t+1} (Cube)`
cannot be removed -- a recurrence is never fully parallel. The win is overlapping
everything ELSE: schedule the S-independent GEMMs (`Q@K^T`, `Q@S` on the entering
state, `Aqk@V`) and the bulk Vec work to fill the window while S_next forms. If
`w@S` dominates the chunk's Cube time, the overlap window shrinks toward the
single-buffered cost -- measure before committing.

**Failure modes.** UB peak (192 KB) with two fp32 S buffers + scratch; L0C peak
(128 KB) with concurrent accumulators; tail-chunk flag imbalance (deadlock);
routing S through the ping-pong flags (re-serializes every chunk).

## COOK-§8.15: Attention / online-softmax kernels -- the recipe

Everything here is measured on one generated flash-attention kernel taken from 2.72x the
vendor to ~1.15x at S=2048. It is written as a recipe because almost none of it is
specific to attention *as an algorithm* -- it applies to any stage that alternates Cube
and Vec over a swept axis with a running reduction. Read COOK-6.5 and COOK-6.7 first;
this entry assumes them.

### The one that costs the most: do NOT materialize the score row in GM

An online-softmax kernel exists precisely so the `[BQ, S]` score matrix never has to. If
your GM workspace formula contains the sweep axis, you have written the algorithm's
arithmetic without its memory behaviour, and it will fall off a cache cliff outside the
shape you tuned at. The kernel above had

```
per_core = 2 slots x ( BQ*S*2  +  BQ*S*2  +  BQ*D*4 )
                       ^scores    ^probs     ^accumulator
                       O(S)       O(S)       O(1)
```

and its footprint went 51 MB at S=2048 to 771 MB at S=32768. Ratio to vendor by S:
1.165 / 1.107 / (void) / **1.888** / **2.537** at 2048/4096/8192/16384/32768. The vendor
scaled as S^2 predicts; this scaled ~7x per doubling past 8192. **Write the workspace
formula down and check which terms carry the sweep axis, before you write the loop.**
See PLAT-§L2 for the capacity budget.

**Know what does and does not remove the O(S) term -- this was got wrong once already.**
Holding the score slab in UB so the second Vec pass never re-reads it from GM is worth
about **1.04x** (ceiling 1.184x), and it is NOT the structural fix: the Cube still writes
the score block and re-reads it as P, and **a Cube<->Vec operand must transit GM on
A2/A3**, so `[BQ, S]` stays resident in the workspace either way. Measured live footprint
after residency is `2 * (BQ*S*2 + BQ*D*4)` per core -- still linear in S.

Only a **true online formulation** is O(1): carry running `m`, `l` and a *rescaled* `O`
accumulator across K/V tiles and never materialize `[BQ, S]` at all.

**But do NOT assume the online form is the faster choice -- it was built and measured, and
on this hardware it LOST at every size up to S=32768.** Both kernels exist; canonical,
arity-matched, null controls valid:

| S | score-materializing + footprint-capped `block_dim` | true online |
|---|---|---|
| 2048 | **1.12** | 3.52 |
| 8192 | **1.07** | 3.28 |
| 16384 | **1.25** | 2.98 |
| 32768 | **2.15** | 2.85 |

The online kernel does exactly what it promises structurally: its workspace is **33.0 MB
at every S from 128 to 32768** (no term contains S), and its ratio *falls* with S -- 3.58
at 4096 down to 2.85 at 32768 -- where the materializing kernel's *rises*. The curves are
converging and would cross somewhere past 32768. Inside the measured range, materializing
and managing the footprint wins, by 3x at S=2048.

**Why: the online form trades memory for SERIALIZATION.** Its rescale recurrence
(`O_run = O_run*alpha + O_slab`) is a dependency chain across slabs, and a marginal-cost
probe on it found duplicating a score `TLOAD` cost 2.3%, a whole `TEXP` 2.0%, a `TSTORE`
0.16% -- nothing on the data path. Its Vec floor was **pure per-chunk sync count**, and
phase C (P@V) stayed fully exposed because phase D depends on it. A materializing kernel
has slack to pipeline; the recurrence does not.

**So choose by S range, and measure both if it matters.** Up to ~32K on A2/A3, materialize
and cap `block_dim` by footprint (PLAT-§L2). Choose the online form when S is large enough
that the footprint cannot be capped, or when the machine has less cache headroom. The
online form is also **2x more accurate here** (2.85e-04 vs 5.5e-04 Frobenius, landing
exactly on the vendor's own fp64 error) because it never round-trips the score block
through fp16 GM -- which may decide it for you independently of speed.

### Diagnose before optimizing -- three probes, in this order

1. **Engine-nulled ablation.** Build variants that null out one engine and time them. If
   `cube_only + vec_only - sync ~= full`, there is ZERO overlap and overlap is your
   biggest single lever. Measured: 731 + 1252 - 69 = 1914 against a full 1904 us.
2. **Marginal-cost probe.** Duplicate ONE op on ONE pipe, keeping all sync intact, and see
   what it costs. This kills plausible-but-wrong diagnoses in one measurement. Measured on
   a kernel sitting at "27% of the fp16 cube roofline": duplicating **every matmul** cost
   **4.8%**, while duplicating Vec loads cost 32.2% and Vec stores 24.9%. It was never
   MAC-underfilled -- both engines were stalled on memory movement, and every L0-tile /
   K-blocking / fractal-underfill hypothesis was dead. **A roofline percentage is not a
   diagnosis.**
3. **block_dim sweep.** If REDUCING `block_dim` makes it faster, you are footprint-bound,
   not compute-bound (see the optimizer skill's diagnostic, and PLAT-§L2).

### The ladder that worked, with what each step bought

Applied in this order to the kernel above; every number is a paired same-process A/B.

| step | gain |
|---|---|
| COOK-6.7 double-buffered Vec loop, event ids partitioned per sub-block | 1.28x |
| hoist per-chunk reductions: accumulate elementwise max/sum TILES, do ONE `TROWMAX`/`TROWSUM` at the end | 1.11x |
| 2-slot software pipeline inside each Cube phase (GM->L1 prefetch, L0B and L0C double buffered) | 1.18x |
| 2-slot GM workspace so Cube runs item i+1 while Vec runs item i | 1.24x |
| remove a redundant Vec->Cube "workspace free" back-edge | 1.090x |
| tile-block the score/prob workspace | 1.076x |
| write P over the scores in place (halves the workspace) | 1.010x |
| make both Vec passes read contiguous chunks | 1.119x, 1.168x |

**Rejected, each with the measurement -- these are as valuable as the wins:**
* 3-slot GM workspace: **1.49x REGRESSION**. Working set 53 -> 80 MB, loses L2.
* fp16 score workspace (halves the largest traffic leg): **no gain**, 1903.7 -> 1891.3 us.
  Falsified the bandwidth hypothesis outright.
* removing a V->V `pipe_barrier`: 1.3% faster and the OUTPUT CHANGED -> a latent ordering
  dependency, not a win.
* pinning K/V or the score chunk to a hot address: both SLOWER, which proved K/V traffic
  was already free -- and that is why a two-q-block merge worth a paper 13.6% traffic
  saving was correctly never built.
* `TPUSH`/`TPOP` for the Cube<->Vec seam: faults once the FIFO ring iterates (COOK-6.6).
  The ablation vindicated the fallback anyway -- the seam was 69 us of 1904.

### Two bugs this class of kernel produces, both from overlap

Both were introduced by the overlap steps above and both validated clean at small sizes:

* **A missing WAR guard on the score tile** made the result wrong by exactly `scale^2` --
  a pass-2 `TLOAD` landed before pass 1's `TMULS` on the same tile, applying the scale
  twice. Localize by dumping each intermediate against the fp64 reference; the phase that
  is still exact bounds the search.
* **The running softmax statistics are PER-WORK-ITEM state.** Once Cube runs item i+1
  while Vec runs item i, `m` and `l` are read one item behind. Give them **one slot per
  workspace slot**. This failed at every size, which is the lucky case -- a subtler
  version fails only at production.

And the coverage rule that catches them: rule 31 in the artifact generator. A kernel can
pass every generated test without its pipelined path ever running -- at `block_dim=20`
with 2 rows/item, T=8 and T=64 have fewer items than lanes, so the prefetch branch was
dead code. Require `items_per_lane >= 3` somewhere in the sweep.

### Benchmarking an attention kernel against a vendor operator

`npu_fusion_attention` returns **seven** values and writes **three** tensors: the
attention output plus `softmax_max` and `softmax_sum`, fp32 `[B, N, S, 8]` with the value
replicated across all 8 lanes. At S=2048 it writes 20.000 MB where a 1-output kernel
writes 16.000 -- **25% more output traffic on a memory-bound stage.** Emitting the two
missing statistics moved the honest ratio 1.110 -> 1.149 and turned "beats the vendor
below S=2048" into "parity at S=256, ~5% behind at S=512-1024". They are also what a
backward pass consumes, and an online-softmax kernel already has both, so emitting them
is a broadcast and a store, not new math. See rule 29(h) in the artifact generator; also
(i) allocation symmetry and (j) same-process comparison.

### Two more traps, both found the hard way

**A tile binding selected per outer-loop iteration and consumed in a LATER region does not
survive.** Only a compile-time-constant address, or a `TASSIGN` immediately followed by its
use, is reliable. A resident-slab implementation that picked its slab tile per iteration
and used it further down produced a stale read that looked like an addressing bug for
several rounds. If you need per-iteration slabs, accumulate into disjoint row ranges of ONE
statically-addressed tile and reduce once.

**Reusing resident UB slots across an outer-loop boundary needs a WAR guard even when the
inner loop does not.** Within a slab each chunk owns its own slot, so no guard is needed and
none gets written; at the slab boundary the slots are reused, and MTE2 refills them for slab
g+1 while slab g's second Vec pass is still draining on V and MTE3. The symptom is one slab
reading stale data while every slab is exact *when run alone* -- which is the tell: if
per-item results are correct in isolation and wrong together, it is interference, not
addressing. A directed slab-boundary guard costs far less than `pipe_barrier(PIPE_ALL)`
(measured: the barrier version was 1.293x SLOWER than baseline, the directed guard 0.977x).

---

## COOK-§9: L1 Prefetching For Next-State Tiles

Use a second L1 tile only when there is a clear next-state tile.

```cpp
constexpr int32_t HL1Addr = 65536;
constexpr int32_t HNextL1Addr = HL1Addr + 65536;

L1Mat<half, 128, 128> h_l1;
L1Mat<half, 128, 128> h_next_l1;
TASSIGN(h_l1, HL1Addr);
TASSIGN(h_next_l1, HNextL1Addr);
```

---

## COOK-§10: Layout Adaptation Via Broadcast Ops

Replace scalar extraction with broadcast-friendly tensor forms.

```cpp
UbND<float, 1, HalfChunk> g_r_ub;
UbND<float, 1, ChunkSize> g_c_ub;
UbND<float, HalfChunk, ChunkSize> g_r_2d_ub;
UbND<float, HalfChunk, ChunkSize> g_c_2d_ub;
UbND<float, HalfChunk, ChunkSize> coeff_ub;

TROWEXPAND(g_r_2d_ub, reinterpret_cast<UbDN<float, HalfChunk, 1>&>(g_r_ub));
TCOLEXPAND(g_c_2d_ub, g_c_ub);
TSUB(coeff_ub, g_r_2d_ub, g_c_2d_ub);
TMINS(coeff_ub, coeff_ub, 0.0f);
TEXP(coeff_ub, coeff_ub);
TMUL(coeff_ub, coeff_ub, msk_ub);
```

---

## COOK-§10.5: Wide-Axis Reduction and Vec Matvec (reduced axis > 64 lanes)

For a tile Vec matvec `y[i] = sum_j S[i,j] * k[j]` (a GEMV / one matvec step of a
recurrent scan) where the **reduced axis is wider than the 64-fp32 Vec lane block**
(e.g. dk = 128), TWO things decide correctness: the reduction DIRECTION and the 64-lane
SPLIT. Both are **verified on real NPU (dav-c220, CANN 9.1.0)** -- the ISA source alone
is misleading here (see the "why not one TROWSUM" note).

**(A) Direction -- reduce the wide axis with `TROWSUM` (per-row output), not `TCOLSUM`.**
- `TROWSUM(dst, src, tmp)` reduces each row over its COLUMNS -> one **per-row scalar**
  (narrow output). This is the right shape for a matvec: output `y[i]` is one value per
  output row.
- `TCOLSUM(dst, src)` reduces each column over its ROWS -> a `[1, W]` per-column row. It
  masks `set_vector_mask(0, W)` and issues a single `vadd` (`rptTimes = 0`), so its
  OUTPUT width truncates to the first **64 fp32 lanes**: for `W > 64` the tail is
  silently wrong. This is the trap behind "only the first 64 outputs are correct."

> **PREMISE (B) BELOW IS FALSE -- see C15.** `TROWSUM` does NOT truncate at 64 lanes;
> it is exact at 128-wide when its scratch is sized to the SOURCE (`tmp = src/2` exact,
> `src/4` silently wrong). The split described below is therefore unnecessary for
> correctness. It is retained only because it is a legitimate *performance* pattern in
> some shapes -- COOK-10.5's own hardware test measured the manual fold FASTER than a
> DYNAMIC-shaped full-width `TROWSUM`, which takes a generic path issuing a
> `pipe_barrier(PIPE_V)` per 64-element repeat. Split for SPEED if you measure it;
> never for correctness. Part (A) on direction is unaffected and still holds.

**(B) Split -- even `TROWSUM` reduces only the first 64 lanes of a >64-wide row on this
build.** The `TRowReduceInstr` repeat-tiling path in the ISA source is NOT what runs for
a RowMajor dst on dav-c220/CANN 9.1.0 (confirmed by isolation test: a single `TROWSUM`
over a 128-wide row summed only lanes 0..63). So **split the reduced axis into <=64-lane
blocks and `TADD` the partials** (this is the C15 pattern, and it is REQUIRED, not
optional):

```cpp
// S : [dv, dk]  (output dim = rows, reduced dim = cols) -- keep this orientation
// k : [1, dk] (the vector).  dk = 128 -> two 64-lane halves.
UbND<float, DV, 64> S_lo, S_hi;       // the two dk-halves of S (or views into S[:, :64] / S[:,64:])
UbND<float, 1, 64>  k_lo, k_hi;
UbND<float, DV, 64> kexp, prod;
UbND<float, DV, 8>  y_lo, y_hi, y, tmp;

TCOLEXPAND(kexp, k_lo); TMUL(prod, S_lo, kexp); TROWSUM(y_lo, prod, tmp);  // lanes 0..63
TCOLEXPAND(kexp, k_hi); TMUL(prod, S_hi, kexp); TROWSUM(y_hi, prod, tmp);  // lanes 64..127
TADD(y, y_lo, y_hi);                  // full-width per-row scalar in col 0
// y[i] = GetValue on row i, col 0
```

For the exact fp32 shapes `[64,128]`, `[32,256]`, `[16,512]`, `[8,1024]` with a ColMajor
`[R,1]` dst, `TROWSUM` auto-dispatches a correct shape-specialized fast path (no manual
split needed) -- but do NOT rely on this for other shapes/dst-layouts; verify on the gate.

**(C) Rank-1 update** `S += outer(y_delta, k)` stays in the SAME `[dv, dk]` orientation
(no transpose), so reduce direction and update direction agree end-to-end:

```cpp
UbND<float, DV, DK> y_exp, k_exp, outer;
TROWEXPAND(y_exp, reinterpret_cast<UbDN<float, DV, 1>&>(y_delta)); // per-row scalar across cols
TCOLEXPAND(k_exp, k_row);                                          // k down rows
TMUL(outer, y_exp, k_exp); TADD(S, S, outer);                     // S += outer
```

**(D) Placing a GM-contiguous vector into col 0** (needed to feed a per-row scalar, e.g.
`v` in a delta-rule step): a `[64,64]` `TTRANS` reliably delivers an output ROW but NOT
an output column (it keeps only the diagonal 16x16 fractal), and a `DIM_4=1` strided GM
load PACKS contiguously rather than scattering to col 0. The reliable per-row column
producer is **broadcast (`TCOLEXPAND`) x identity-mask (`TTRI`) then `TROWSUM`**.

Cross-refs: this **amends S9** (SKILL.md) for the wide-free-axis case (its `TCOLSUM`
orientation truncates when the free/output axis > 64) and **reinforces C15** (the 64-lane
limit applies to `TROWSUM` too on this build, so the block-split is mandatory). See also
COOK-§10 (broadcast ops). ALWAYS verify on the real-NPU gate across the FULL width
(not a sub-64 sim shape) and explicitly check the tail rows -- the truncation is silent.

---

## COOK-§11: Dynamic Tail Handling

Keep the fast path static and isolate only the tail logic.

```cpp
struct VarlenTileInfo {
  uint32_t gm_offset;
  uint32_t valid_size;
};

AICORE inline VarlenTileInfo get_tile_info(uint32_t tile_id,
                                           uint32_t tile_size,
                                           __gm__ int32_t* cu_seqlens) {
  return {0, tile_size};
}

// Fast path: full tile_size rows.
// Tail path: only the final partial tile narrows valid_size.
```

**WARNING: the helper above is a STUB.** It takes `cu_seqlens` and ignores it,
always returning `{0, tile_size}`. It is a shape placeholder, not a recipe. For an
actual runtime-determined schedule use COOK-§11.5.

#### COOK-§10.8: CROSS-WORK-ITEM reduction (a column sum across rows)

A reduction along the axis that work items are SPLIT over cannot be closed inside one
core's tile loop. This is the shape of every `dgamma`/`dbeta` in a norm backward, of
per-expert accumulators in MoE, and of any `sum(axis=0)` over a row-parallel stage.
Nothing else in this cookbook covers it.

Two mechanisms, both real on A2/A3:

**(a) Per-core partial + a second pass.** Each core reduces its own rows into a
`[1, D]` partial at `workspace + core_id * D`, then a second kernel (or a second phase
after a barrier) reduces the `block_dim` partials. Deterministic, needs no pre-zeroed
output, and -- the property that matters -- every lane writes ONE CONTIGUOUS `[1, D]`
run, so it never touches C1's 32-byte scalar-scatter hazard. Cost is a
`block_dim * D * 4` workspace and one extra launch: an INTERCEPT, not a slope.

**(b) `TSTORE` with atomic add.** `pto/npu/a2a3/TStore.hpp:126` takes
`AtomicType currentAtomicType = AtomicType::AtomicNone`, so
`AtomicType::AtomicAdd` is reachable on A2/A3 -- not an A5-only feature, which is the
only place the platform model previously mentioned it. One pass, no workspace, but the
destination must be pre-zeroed and the result is NOT bit-deterministic across runs
(accumulation order varies), so it fails a determinism gate.

**Choose (a) when the stage must be deterministic or bit-exact**, which is the default
for a validation gate here. Choose (b) when the workspace or the extra launch is the
measured bottleneck and non-determinism is acceptable -- and say so in the report.

Do NOT emulate this with interleaved scalar `__gm__` stores. Scalar stores are 32-byte
cache-line granular: an interleaved cross-lane scatter loses **half the array at
block_dim=1** (C1, probed). Contiguous per-lane runs are the only safe scalar form,
and MTE `TSTORE` is better still.

## COOK-§11.5: Schedule resolved ON DEVICE from a runtime boundary tensor

Use when the work partition is **data**, not shape: grouped matmul with a
`group_list`, varlen attention with `cu_seqlens`, MoE with runtime expert counts.
This was previously unimplementable here because C1 wrongly forbade the scalar
`__gm__` read it needs; C1 is now corrected and probed
(`isa_probes/probe_gmscalar.cpp`, including the read on the **Cube** core).

The three alternatives and why they lose:
* *host readback* -- adds a device->host sync per launch that the vendor does not pay;
* *pad to the worst case* -- 3 equal groups instead of 256/1024/512 costs 3x the FLOPs;
* *one group per core* -- 3 groups over 24 cores leaves 21 cores idle, and the
  largest group is 4x the smallest, so the imbalance is the runtime.

The pattern:

```cpp
// 1. EVERY core independently re-derives the SAME tile enumeration with an O(G)
//    scalar prefix walk. No host sync, no inter-core communication, no barrier:
//    the enumeration is a pure function of group_list, so all cores agree.
volatile __gm__ int64_t *gl = (volatile __gm__ int64_t *)group_list;
int32_t n_tiles = 0, prev = 0;
for (int32_t g = 0; g < G; ++g) {
  dcci((__gm__ void *)(gl + g), SINGLE_CACHE_LINE);   // host wrote it: invalidate
  int32_t end = (int32_t)gl[g];
  // Defensive: a malformed list must not generate an out-of-bounds tile.
  if (end < prev) end = prev;
  if (end > M)    end = M;
  n_tiles += (end - prev + MT - 1) / MT;              // MT = rows per tile
  prev = end;
}

// 2. Claim tiles by a grid stride. The WORK ITEM IS A TILE, NOT A GROUP -- that is
//    what turns 3 uneven groups into n_tiles balanced items across all cores.
for (int32_t t = get_block_idx(); t < n_tiles; t += get_block_num()) {
  // 3. Map t -> (group, row range) with a second O(G) walk.
  //    Generate the tile grid PER GROUP so no tile ever straddles a boundary:
  //    only the GM load/store extents are runtime-variable, and every Cube tile
  //    keeps a STATIC shape (Cube fractal tiles require static column extents).
  ...
}
// 4. Rows not covered by any group must be explicitly ZEROED, not left undefined.
```

Non-negotiables, each of which was a real failure mode:
* **Clamp the boundaries** non-decreasing and `<= M` before use. A malformed or
  non-monotonic `group_list` must produce zeros, not an OOB access.
* **Validate the degenerate cases**: zero-row first/middle/last group, two empty
  groups, all-empty, single-row groups, unaligned boundaries, a boundary past M,
  a short last boundary, non-monotonic input.
* **The optimal `block_dim` is data-dependent** -- the tile count is a function of
  runtime data, so no single value is right for all inputs. Measure the crossover
  and report it; do not quote the flattering point as if it generalised.

---

## COOK-§12: Wrapper-Side Padding And Block-Dim Selection

```python
def _round_up(v: int, tile: int) -> int:
    return ((v + tile - 1) // tile) * tile

def _choose_block_dim(m: int, n: int, max_block_dim: int) -> int:
    m_loop = m // 128
    n_loop = n // 256
    core_loop = m_loop * n_loop
    if core_loop <= 0:
        return 1
    return max(1, min(core_loop, max_block_dim))
```

---

## COOK-§13: Hard Reject List

Never emit these as the main solution:

- `BLayout::RowMajor, SLayout::NoneBox` on Mat tiles — the #1 cause of
  Cube compilation failures. Mat tiles MUST use `BLayout::ColMajor,
  SLayout::RowMajor` or (for ZN) `BLayout::RowMajor, SLayout::ColMajor`.
  `NoneBox` is for Vec tiles ONLY.
- `TEXTRACT(L1Mat) → TileRight` for transposed B — must route through
  `TRESHAPE(L1MatZN, L1Mat)` first. See COOK-§8.7 feed chain.
- `exp()`, `expf()`, `std::exp()`, `__builtin_expf()` — use `TEXP` on PTO tiles
- `wait_flag_dev(N)` without a prior `set_flag`/`set_cross_flag` producer
- guessing logical dimensions from one observed validation case
- inventing custom scalar helpers such as `exp_scalar(...)` for main stage math
- performing dominant BTxK / BTxV / BTxBT computation as GM pointer loops
- fake direct Cube↔Vec sharing without GM + FFTS handoff
- blanket `pipe_barrier(PIPE_ALL)` after every operation
- whole-kernel dynamic machinery when only the tail is dynamic
- prose before the first `#include`
- quoted C++ blobs containing literal `\n` escapes
- using `get_subblockid()` while both vids stay active on the same static UB address map

---

> If the stage alternates Cube and Vec over a swept axis with a running reduction
> (attention, online softmax, chunked scan), go to **COOK-8.15** first -- it is the
> whole recipe, including the three diagnostic probes to run BEFORE optimizing and the
> workspace rule that decides whether the kernel survives outside its tuned shape.

## COOK-§14: Pattern Selection Heuristic

**Primary signal: StageSpec.instruction_families. stage_family is semantic guidance only.**

```
IF instruction_families contains TMATMUL, TMATMUL_ACC, or TTRI:
  → cube_vec_pipeline or cube_only
  → Use COOK-§8.5-§8.12 for Cube type templates, one-shot/K-sliced GEMM,
    L0C store, transposed TRESHAPE, GM workspace layout, TCVT casts,
    and the complete Vec→Cube→Vec dataflow protocol.
  → stage_family tells you WHAT the contraction means (seed/closure/gram/
    correction), not HOW to compute it. Never use Vec-only reduction loops
    for contraction stages.

IF reference_source contains einsum, @, torch.matmul, torch.triu, or torch.tril:
  → same as above — these are matrix contraction patterns

ELSE (pure Vec ops only: TLOAD, TADD, TMULS, TMOV, TSTORE, TEXP, without
     any of the Cube signals above):
  → vec_only
  → Use COOK-§1, §1.5, §1.6, §1.65, §1.66, §1.67, §2, §6
```

---

> Attention / online-softmax stages: **COOK-8.15**.

## COOK-§15: Stage Archetypes

Before writing code, classify the stage into exactly one primary archetype.

### Archetype A: `vec_only`

Use for elementwise transforms, rowwise/colwise broadcast updates,
activation-like kernels, and prefix-sum stages.

Prefer `TLOAD → Vec ops → TSTORE` with explicit `set_flag`/`wait_flag` pairs.
For prefix-sum accumulation, use the column-tiled accumulation scan pattern (COOK-§1.66).

### Archetype B: `cube_only`

Use for dense matrix products, tile contractions, and block updates
dominated by `TMATMUL`.

### Archetype C: `cube_vec_pipeline`

Use for stages with GEMM output that must be gated/masked/normalized by Vec.

Requires explicit GM workspace + FFTS flag protocol with bootstrap (COOK-§8).

### Archetype D: `varlen_tail`

Use for packed sequences and ragged final chunks.

### Archetype E: `skeleton_only`

Use only for underspecified stages where the legal memory path is known but
compute lowering is not trustworthy. Never use as the final answer for a
semantically specified stage.

---

## COOK-§16: Anti-Patterns To Reject

(See SKILL.md Forbidden Patterns table for the consolidated list.)

---

## COOK-§17: Minimum Safe Default

If the stage is underspecified:

1. COOK-§1 host/device split
2. COOK-§2 outer runtime work loop
3. COOK-§3 explicit `TASSIGN` with budget guard (COOK-§4)
4. COOK-§5 narrow flag helpers
5. a small PTO-op-centric compute body

Do not replace that with commentary or scalar fallback code.

---

## COOK-§18: Operator Decomposition and Instruction Selection

When the stage math maps to a known operator pattern, use this table to
select the instruction sequence. Verify each instruction with MCP
(`get_cpp_intrinsic`) before emitting.

| Operator type | Decomposition | Typical instruction chain |
|---------------|---------------|---------------------------|
| Activation (pointwise) | load → compute → store | TLOAD → TEXP/TRELU/TLRELU → TSTORE |
| Reduction (axis) | load → reduce → store | TLOAD → TROWSUM/TCOLSUM/TROWMAX → TSTORE |
| Element-wise binary | load2 → op → store | TLOAD ×2 → TADD/TSUB/TMUL/TDIV/TMAX/TMIN → TSTORE |

**`TDIV` is RowMajor-only.** Compile-checked: a `BLayout::ColMajor` operand fails the
`static_assert` outright, while the RowMajor form compiles. This collides with
`TROWEXPANDMUL` Mode 1, which *requires* a ColMajor `[N,1]` operand -- so a
"compute `127/amax` then broadcast-multiply" chain cannot be written in one layout.
Compute the reciprocal in the ND/RowMajor domain, then move the single valid column
into ColMajor for the expand. A run lost a repair attempt to this.
| Element-wise scalar | load → scalar-op → store | TLOAD → TADDS/TSUBS/TMULS/TDIVS → TSTORE |
| Broadcast+op | load → fused-broadcast → store | TLOAD → TROWEXPANDADD/SUB/MUL/DIV → TSTORE |
| Matrix multiply (Cube) | load → extract → matmul → store | TLOAD → TEXTRACT → TMATMUL → TSTORE(L0C) |
| Fused multiply-add | load → fused-acc → store | TLOAD → TAXPY → TSTORE |
| Type conversion | load → convert → store | TLOAD → TCVT → TSTORE |
| Conditional select | load2 → compare → select → store | TLOAD ×2 → TCMP → TSEL/TSELS → TSTORE |
| Softmax | load → exp → sum → divide → store | TLOAD → TEXP → TCOLSUM → TCOLEXPANDDIV → TSTORE |
| LayerNorm-style | load → normalize → scale+shift | TLOAD → TSUBS → TDIVS → TMULS → TADDS → TSTORE |
| Math functions | load → math → store | TLOAD → TLOG/TSQRT/TRSQRT/TPOW/TRECIP/TABS/TNEG → TSTORE |
| Ternary fused | load2 → fused → store | TLOAD → TADDC/TSUBC/TADDSC/TSUBSC → TSTORE |

### Instruction selection principles

1. **Prefer fused instructions** (A5) — reduce intermediate tiles and data movement
2. **Prefer broadcast+op** over expand + separate op when available
3. **Use scalar ops** (`TADDS`, `TMULS`) when one operand is a compile-time constant
4. **Use axis reduction** (`TROWSUM`, `TCOLSUM`) instead of scalar accumulation loops
5. **Verify with MCP** — call `get_cpp_intrinsic` for any instruction not in the
   cookbook patterns above

---

## COOK-§19: Event-Based Sync API (Preferred Pattern)

The Event API provides automatic dependency tracking and is the **preferred**
sync pattern for simple linear compute chains. Fall back to manual
`set_flag`/`wait_flag` (COOK-§5, §6) for complex pipelines with branching
or overlapping stages.

### Basic Event pattern (linear chain)

```cpp
Event<Op::TLOAD, Op::TADD> event0;
Event<Op::TADD, Op::TEXP> event1;
Event<Op::TEXP, Op::TSTORE_VEC> event2;

event0 = TLOAD(srcTile, srcGlobal);            // fires event0 on completion
event1 = TADD(dstTile, src0Tile, src1Tile, event0);  // waits event0, fires event1
event2 = TEXP(outTile, dstTile, event1);       // waits event1, fires event2
TSTORE(dstGlobal, outTile, event2);            // waits event2
```

### When to use Event sync

- Simple linear chains: `TLOAD → op1 → op2 → ... → TSTORE`
- Single input, single output flows
- Activation functions, element-wise transforms

### When to use manual flag sync instead

- **Double-buffering / ping-pong** — need distinct event IDs per slot (COOK-§6)
- **Cube+Vec pipeline** — cross-core sync requires `set_cross_flag`/`wait_flag_dev` (COOK-§8.6)
- **Overlapping MTE2/Vec/MTE3** — need fine-grained pipe handoff (COOK-§6, §1.65-§1.67)
- **K-sliced GEMM** — complex MTE1/M/TEXTRACT interlock (COOK-§7, §8.8)

### Event sync rules

- Each `Event<Op::Src, Op::Dst>` consumes one event ID internally
- Events chain: `event1 = OP(..., event0)` means "wait for event0, then fire event1"
- `TSTORE` accepts an Event as its last argument for auto-wait
- Do not mix Event and manual `set_flag`/`wait_flag` on the same dependency edge
- Under `-DMEMORY_BASE` (manual mode), `pipe_barrier(PIPE_ALL)` is still required
  after TLOAD/TSTORE even when using Events

---

## COOK-§20: Tile Dimension Selection Guide

Choose tile dimensions based on data type and compute path. These are
starting points — adjust based on UB/L1 budget constraints.

### Vec tile dimensions (UB, per core)

| Data type | Recommended dimensions | Rationale |
|-----------|----------------------|-----------|
| `float` (4B) | 64×64, 32×128, 16×256 | 16KB per tile, 12 tiles fit in 192KB UB |
| `half` (2B) | 64×128, 32×256, 16×512 | 16KB per tile, 12 tiles fit in 192KB UB |
| `int32` (4B) | 64×64, 32×128 | Same as float |
| `int16` (2B) | 64×128, 32×256 | Same as half |
| `int8` (1B) | 64×256, 32×512 | 16KB per tile |

**Column alignment**: For `BLayout::RowMajor` tiles, `cols` must be
32-byte aligned. For fp32: cols % 8 == 0. For fp16: cols % 16 == 0. → PLAT-§Align

### Cube tile dimensions (L1/L0, per core)

| Buffer | Capacity (A2/A3) | Max tile size |
|--------|-------------------|---------------|
| L1 | 512 KB | 128×128 fp16 = 32KB, up to 16 tiles |
| L0A | 64 KB | 128×64 fp16 = 16KB, or 64×128 |
| L0B | 64 KB | Same as L0A |
| L0C | 128 KB (A2/A3), 256 KB (A5) | 128×128 fp32 = 64KB, up to 2 tiles |

**TMATMUL constraints**: M, N, K dimensions should be multiples of 16
for fp16 inputs. Tail handling needed for non-multiple dimensions.

### Budget calculation

Before choosing tile sizes, verify they fit:

```
Vec tiles:  sum(tile_bytes × live_count) ≤ 196608 (A2/A3) or 262144 (A5)
Cube L0:    L0A_bytes + L0B_bytes ≤ 131072 (128KB combined)
Cube L0C:   acc_bytes ≤ 131072 (A2/A3) or 262144 (A5)
```

Always emit `static_assert` guards for computed budgets. → COOK-§4


## COOK-§6.8: `pipe_barrier(PIPE_ALL)` is over-used -- but the obvious replacement is UNSAFE

> **RETRACTION (this section previously recommended the direct `MTE3->MTE2` guard; do not
> use it).** Two independent runs contradicted it:
>
> * **The safety claim failed.** A `masked_softmax` kernel validated **49/49 at
>   `block_dim=24`** and was **broken at `block_dim=1`**. The in-place compute-and-store slot
>   needs a *separated* token; the direct `MTE3->MTE2` form does **not** hold beyond one item
>   per lane -- **even with Vec work in the ring**, which is exactly the case the earlier
>   amendment declared safe. (`moe_token_permute` had already shown it failing on AIV
>   sub-block 1 in a ring with no Vec work.)
> * **The performance claim did not reproduce.** The original 1.028x re-measured as
>   **1.003x** on another kernel -- noise.
>
> **So the corrected guidance is conservative: use the `COOK-§6.7` pipe-role split for slot
> recycling, and do NOT hand-roll direct DMA-to-DMA tokens.** The census observation below
> stands and is still worth knowing, but at a measured 0.3-2.8% the upside does not justify
> hand-rolling a guard whose failure mode is *silent corruption that appears only at
> production item counts*. Change a barrier only with a measurement AND a `block_dim` sweep.

### The census observation (still valid)

`pipe_barrier(PIPE_ALL)` drains **every** pipe, so nothing of the next work item can start
until the current one has fully retired. The vendor almost never does this: a census of the
shipped AscendC kernels finds `PIPE_ALL` in **0.3%** of files (19 of ~5900) against 1198
using a scoped `PipeBarrier<pipe>` -- while **95% of our generated kernels use it**
(105 of 111, 417 occurrences). See `references/vendor_idiom_census.md`.

**Measured on a real stage** (`dequant_swiglu_requant`, two end-of-item `PIPE_ALL` barriers
replaced), validation PASS, same device, serialized runs:

| | time | GB/s |
|---|---|---|
| `pipe_barrier(PIPE_ALL)` | 46.06 us | 204.9 |
| scoped `MTE3->MTE2` + `MTE3->V` | **44.81 us** | 210.6 |

**1.028x.** Real, safe, and modest. Do not expect more than low single digits on a
memory-bound stage whose barriers sit at the end of an item; expect more where a `PIPE_ALL`
sits *inside* a hot loop, and less where the stage is already hidden behind another engine.

**The replacement, not the deletion.** Name the actual hazard and guard exactly it. At an
end-of-item seam the hazards against the outstanding `TSTORE` are both WAR:

```cpp
// MTE3 -> MTE2 : the next item's load must not overwrite UB the store is still reading
// MTE3 -> V    : the next item's first Vec op must not overwrite that tile   (D6)
set_flag(PIPE_MTE3, PIPE_MTE2, (event_t)evid(0, vid));
wait_flag(PIPE_MTE3, PIPE_MTE2, (event_t)evid(0, vid));
set_flag(PIPE_MTE3, PIPE_V,    (event_t)evid(0, vid));
wait_flag(PIPE_MTE3, PIPE_V,   (event_t)evid(0, vid));
```

Each `(src,dst)` pair is its own `HardEvent` class with its own id pool, so both may reuse
`evid(0, vid)` even when `MTE2->V`, `V->MTE2` and `V->MTE3` flags are already live in the
same kernel (`COOK-§6.5`).

> **AMENDMENT -- the direct `MTE3->MTE2` form above is only safe when the ring HAS Vec work.**
> In a ring with **no Vec stage** (a pure DMA copy/permute: load -> store, nothing in
> between), the direct DMA-to-DMA flag pairs do not hold the WAR guard on **AIV sub-block 1**.
> Probed on `dav-c220`, T=16, 8 tokens/lane, each token filled with its own index:
>
> | | result |
> |---|---|
> | vid 0 | clean **100/100** across 10 different event-id bases |
> | vid 1 | corrupt in **6 of those 10** -- the store read a slot the next load had refilled |
> | onset | exactly the **3rd** item per lane -- the first ring wraparound |
> | disjoint id ranges per pipe pair | did **NOT** fix it, so it is not an id collision |
> | re-mediating both edges through `PIPE_V` | clean **70/70**, and every validation since |
>
> This is the `COOK-§6.7` signature: correct at small sizes, wrong at production, because the
> pipelined path is never taken at small sizes. **For a ring with no Vec work, route both
> edges through `PIPE_V`** (`MTE3->V->MTE2` and `MTE2->V->MTE3`) -- it costs nothing because
> V is idle, and it is the same four pipe pairs the proven double-buffer recipe uses. The
> mechanism is not established; only the reproduction and the fix.
>
> The measured 1.028x for the scoped form above stands -- that stage has a Vec chain between
> load and store, which is the case the direct form is safe for.

**Never simply delete the barrier.** Removing it has failed validation before
(`dequant_swiglu_requant` attempt 1), and a per-item `PIPE_ALL` was silently protecting
output tiles against a WAR hazard no rule mentioned (`deep_norm_backward` D6). Replace,
then re-validate.


## COOK-§6.9: TROWEXPAND src may be ColMajor -- the dst is the one that must be RowMajor

**Corrects an earlier claim in §8.6P #16** ("requires RowMajor src AND dst; a ColMajor column
gives nan"). That is wrong about the src, and it steers a generator away from the natural
form of a row-broadcast. Read straight off the A2/A3 backend
(`pto/npu/a2a3/TRowExpand.hpp`), whose asserts are **asymmetric**:

```cpp
static_assert(TileDataSrc::SFractal == SLayout::NoneBox,
              "Fix: TROWEXPAND Src layout must be ND or DN!");        // src: EITHER layout
static_assert(TileDataDst::isRowMajor && TileDataDst::SFractal == SLayout::NoneBox,
              "Fix: TROWEXPAND dst layout must be ND!");              // dst: RowMajor ONLY
```

and whose own comment names both supported forms:

```
[1, M] -> [M, elemPerBlock], src is row major.
[M, 1] -> [M, elemPerBlock], src is column major.
```

So a **ColMajor `[M,1]` column is a first-class source**, not a bug. Which form you get is a
compile-time branch on `TileDataSrc::isRowMajor`.

**The fast path has extra conditions** -- miss any and you silently fall back to the generic
(slower) `TRowExpand` rather than the `TRowExpandBrcb` broadcast:
* dtype is b16 or b32 only;
* fully static shapes (`Rows == ValidRow` and `Cols == ValidCol` on BOTH tiles);
* `dst.Cols == elemPerBlock` (32 bytes / sizeof(T));
* `sizeof(T) * M` a multiple of 32 B, i.e. **M a multiple of 8** for b32.

**Method note, worth more than the rule.** This was reported by a pipeline run, rejected by
me on the strength of the MCP page -- which says *"ND fractal (`isRowMajor` and
`SLayout::NoneBox`) for both `src` and `dst`"* -- and then confirmed from the implementation,
which does **not** require `isRowMajor` on the src. **The MCP constraint text is wrong here.**
When a run's PROBED claim conflicts with a doc, the implementation is the tiebreaker, not the
doc. Both had to be checked; only one was authoritative.


## COOK-§6.10: NEVER remove `pipe_barrier(PIPE_V)` between dependent Vec tile ops

Three independent campaign runs have now tried this as an optimization. All three found it
unsafe, and the failure mode gets harder to detect each time:

| run | what removing it did |
|---|---|
| `dequant_swiglu_requant` | 1.011x faster and **wrong on all 96 runs** |
| `masked_softmax` / `top_k_top_p` | wrong **and** ~10% slower -- it is not even a speed win |
| `gelu` | 1.038x faster, **passed the entire validation suite AND an exhaustive sweep of the whole finite fp16 domain**, and is silently wrong at tile widths <= 768 |

The `gelu` case is the one to remember. It was correct at every width the suite happened to
use, and only surfaced because a *later, unrelated* optimization started producing narrower
tiles. An exhaustive value-domain check did not catch it, because the bug lives along the
**tile width** axis, not the value axis.

**Treat the barrier as load-bearing and do not spend an attempt on removing it.** If a
profile says it costs real time, change the tiling so fewer are needed -- do not delete them.

Two compounding hazards from the same run: two *individually safe* changes (aliasing the
output tile onto a scratch tile, and removing a barrier) combined into an intra-Vec WAR, so
**validate combinations, not just each change against the baseline**; and a mandatory barrier
belongs OUTSIDE any `#ifdef` that a variant can disable.

This is the sixth instance of the project's recurring blind spot: **a validation sweep that
never varies a dimension cannot see a bug that lives along it** -- here, tile width.


## COOK-§8.15 CORRECTION: the duplicate-the-op marginal-cost probe gives FALSE NEGATIVES

The marginal-cost probe -- duplicate an operation, measure the delta, conclude what it costs
-- returned **"every op is free" (0.6-3.1%)** on a kernel where the ops were emphatically not
free. It was wrong **twice, for two different reasons**:

* **Duplicated Vec ops are compiler-elidable.** The second copy has no observable effect, so
  it is removed and you measure nothing.
* **A duplicated `TLOAD` targeting the next block acts as a PREFETCH.** It does not add cost;
  it *hides* cost, and can make the probe read faster than the baseline.

Both failures point the same way -- toward "this resource is free" -- which is the most
expensive wrong answer available, because it retires the correct lever.

**Use an engine-nulled ablation instead: REMOVE the work and keep every flag, barrier and
descriptor.** Deleting is not elidable and cannot prefetch. It got the same kernel right
immediately, and it is the same construction as the noop-floor probe already used elsewhere
in this cookbook.

If you must duplicate rather than remove, make the duplicate observable (write to a distinct
live destination) and point any duplicated load at data the kernel will not touch again.
