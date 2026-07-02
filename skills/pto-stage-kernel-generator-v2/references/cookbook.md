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
- `vid != 0` return: UB is shared by both Vec sub-blocks. Without explicit
  address partitioning between vids, return on nonzero vid. → PLAT-§Subblocks
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
- Always use `EVENT_ID0` as the default event index unless double-buffering

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

set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

for (uint32_t processed = 0, ping = 1; processed < elements_to_process;
     processed += tile_cols) {
  const int8_t buf = ping ? 0 : 1;
  const event_t ev = ping ? static_cast<event_t>(EVENT_ID0)
                          : static_cast<event_t>(EVENT_ID1);

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
gives BIT-IDENTICAL output. The GM round-trip is pure per-item slope. Reserve GM commits /
explicit flags for GENUINE CROSS-ENGINE boundaries only (Vec<->Cube via PIPE_FIX/PIPE_MTE3,
or Vec<->MTE). Second lever: HOIST any item-INDEPENDENT tensor (a strict-lower/causal mask,
a constant) OUT of the work-item loop and keep it UB-resident -- rebuilding a 0/1 `TTRI` mask
per item is pure slope (build it once, keep resident). VALIDATED: kkt stage 84->21 us/chunk
(3.98x, accuracy identical 3.59e-4, bit-exact) -- the GM-commit removal was the single biggest
lever, and this is the same class of inefficiency that makes a generated per-stage kernel
~3-5x slower than a hand-tuned one (the whole production gap is per-stage Vec compute, not
fusion). AUDIT EVERY generated stage for cargo-cult commits before blaming the algorithm.
Gotchas surfaced: `TROWEXPAND` requires RowMajor src AND dst (a ColMajor column gives nan);
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

Every reference kernel follows this exact mapping. Never swap the destination —
L1MatZN into TileLeft or L1Mat into TileRight for transposed will fail with
`static_assert`.

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
    // Transposed right operand: TRESHAPE to L1MatZN, then TEXTRACT
    L1MatZN<half, K, N> _bzn;
    TRESHAPE(_bzn, b_l1);
    TEXTRACT(_l0b, _bzn, 0, 0);

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
