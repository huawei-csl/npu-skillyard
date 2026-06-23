# Examples

This document contains annotated failure patterns and complete archetype
examples. Use these as reference during generation and review.

---

## EX-§1: Annotated Failure Patterns

Each example shows a **wrong** pattern, why it fails, and the **correct** replacement.

### EX-§1.1: Direct GM Scalar Access (CRITICAL — NPU Crash)

```cpp
// ❌ WRONG: Scalar indexing of GM pointer — crashes the NPU
__gm__ float* out_ptr = output + offset;
for (int i = 0; i < n; ++i) {
    out_ptr[i] = computed_value[i];  // NPU Alarm state!
}
```

**Why**: The Ascend AI Core cannot address GM directly. Scalar `ptr[idx]`
on a `__gm__` pointer triggers an Alarm requiring hardware reset.

```cpp
// ✅ CORRECT: Use GlobalTensor + TSTORE
Shape<1,1,1,1,ELEMENTS_PER_TILE> shape;
Stride<1,1,1,1,1> stride;
GlobalTensor<float, decltype(shape), decltype(stride)> gm_out(out_ptr + offset);
Tile<TileType::Vec, float, 1, ELEMENTS_PER_TILE> ub_out(1, n);
TASSIGN(ub_out, OUTPUT_UB_ADDR);
// ... compute into ub_out ...
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(gm_out, ub_out);
pipe_barrier(PIPE_ALL);
```

### EX-§1.2: Scalar Exponential (STANDARD — Wrong Results)

```cpp
// ❌ WRONG: Custom scalar exp helper
AICORE inline float exp_scalar(float x) {
    return 1.0f + x + x*x*0.5f + x*x*x*0.166667f;  // polynomial approx
}
// Used as: out[i] = exp_scalar(gate[i]);
```

**Why**: Custom scalar approximations bypass the PTO tile execution engine,
produce incorrect results, and cannot be validated by the numerical checker.

```cpp
// ✅ CORRECT: TEXP on PTO tiles
TEXP(gate_tile, gate_tile);
pipe_barrier(PIPE_V);
```

### EX-§1.3: Missing FFTS Bootstrap (CRITICAL — Deadlock)

```cpp
// ❌ WRONG: wait_flag_dev with no prior producer
for (int iter = 0; iter < num_iters; ++iter) {
    wait_flag_dev(READY_FLAG);   // DEADLOCK on iter=0: no one set this flag!
    // ... consume workspace ...
    set_cross_flag<PIPE_MTE3>(FREE_FLAG, 2);
}
```

**Why**: On the first iteration, no producer has signaled the flag. The
consumer waits forever.

```cpp
// ✅ CORRECT: Bootstrap free-slot signals before the loop
set_cross_flag<PIPE_MTE3>(FREE_FLAG_0, 2);
set_cross_flag<PIPE_MTE3>(FREE_FLAG_1, 2);

for (int iter = 0; iter < num_iters; ++iter) {
    const int slot = iter & 1;
    const int free_flag = (slot == 0) ? FREE_FLAG_0 : FREE_FLAG_1;
    const int ready_flag = (slot == 0) ? READY_FLAG_0 : READY_FLAG_1;

    wait_flag_dev(free_flag);
    // ... produce into workspace[slot] ...
    set_cross_flag<PIPE_MTE3>(ready_flag, 2);
}
```

### EX-§1.4: Wrong Mat Tile Layout (CRITICAL — Compile Failure)

```cpp
// ❌ WRONG: NoneBox SLayout on Mat tile
using BadMat = pto::Tile<pto::TileType::Mat, half, 128, 128,
                         pto::BLayout::RowMajor, 128, 128,
                         pto::SLayout::NoneBox, 512>;
// static_assert failure: Mat tiles require a concrete SLayout
```

**Why**: `SLayout::NoneBox` is for Vec tiles only. Mat tiles require
`SLayout::RowMajor` or `SLayout::ColMajor`.

```cpp
// ✅ CORRECT: L1Mat uses ColMajor/RowMajor
using L1Mat = pto::Tile<pto::TileType::Mat, half, 128, 128,
                        pto::BLayout::ColMajor, 128, 128,
                        pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// ✅ CORRECT: L1MatZN (transposed) uses RowMajor/ColMajor
using L1MatZN = pto::Tile<pto::TileType::Mat, half, 128, 128,
                          pto::BLayout::RowMajor, 128, 128,
                          pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;
```

### EX-§1.5: Scalar Loop Matrix Multiply (STANDARD — Forbidden)

```cpp
// ❌ WRONG: Scalar for-loop matmul as dominant computation
for (int i = 0; i < bt; ++i) {
    for (int j = 0; j < k; ++j) {
        float sum = 0.0f;
        for (int p = 0; p < bt; ++p) {
            sum += A_ptr[i * bt + p] * B_ptr[p * k + j];
        }
        C_ptr[i * k + j] = sum;
    }
}
```

**Why**: Dominant matrix multiplications must use the Cube engine via
`TMATMUL`. Scalar loops bypass the hardware accelerator entirely.

```cpp
// ✅ CORRECT: Cube path with TEXTRACT + TMATMUL
// Load A and B into L1 tiles, extract to L0, multiply:
TEXTRACT(a_l0, a_l1, 0, 0);
TEXTRACT(b_l0, b_l1, 0, 0);
set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
TMATMUL(c_l0, a_l0, b_l0);
// Store result: TSTORE from L0C → GM (via TCVT if float→half needed)
```

### EX-§1.6: Recurrent Copy-Through (STANDARD — Wrong Semantics)

```cpp
// ❌ WRONG: o = u copy-through instead of real recurrent contraction
for (int v = 0; v < V; ++v) {
    o_row[v] = u_row[v];  // No recurrent state interaction!
}
```

**Why**: Recurrent stages must compute `(q_i * exp(g_i)) @ S` and `Aqk @ v_i`
and update state `S`. Copy-through produces outputs that never interact with
recurrent state.

```cpp
// ✅ CORRECT: Real recurrent contraction on Cube
// q_exp_g (BT x K) loaded into L1, S (K x V) loaded into L1
// TMATMUL to produce output = (q*exp(g)) @ S
// Aqk (BT x BT) loaded into L1, v_i (BT x V) loaded into L1
// TMATMUL to produce Aqk @ v_i
// State update: S_new = decay * S + k_i^T @ v_i
```

### EX-§1.7: UB Budget Overflow (CRITICAL — Silent Corruption)

```cpp
// ❌ WRONG: No static_assert, UB silently overflows
constexpr int32_t BUF_A = 0;
constexpr int32_t BUF_B = 65536;    // 64KB
constexpr int32_t BUF_C = 131072;   // 128KB
constexpr int32_t BUF_D = 196608;   // 192KB — exactly at boundary!
// But BUF_D is 64KB tile, so actual max address = 196608 + 65536 = 262144
// Silent UB overflow → data corruption or device trap
```

```cpp
// ✅ CORRECT: Derive addresses and guard with static_assert
constexpr int32_t TileBytes = Rows * Cols * sizeof(float);
constexpr int32_t BUF_A = 0;
constexpr int32_t BUF_B = BUF_A + TileBytes;
constexpr int32_t BUF_C = BUF_B + TileBytes;
constexpr int32_t BUF_D = BUF_C + TileBytes;
constexpr int32_t MAX_UB_ADDR = BUF_D + TileBytes;
static_assert(MAX_UB_ADDR <= 196608,
              "UB footprint exceeds A2/A3 capacity (192 KB)");
```

---

## EX-§2: Full Vec-Only Example (gate_cumsum)

Complete, compile-proven Vec-only kernel. Archetype: `vec_only`.

Stage: cumulative sum of gate features along the intra-chunk time axis.
One work item = one (batch, head, chunk). Grid-stride loop over total chunks.

```cpp
// ============================================================================
// gate_cumsum.cpp — gate prefix stage kernel
//
// Stage role:
//   Compute cumulative sum of per-dimension log-space decay gates along the
//   intra-chunk time axis. For each (batch, head, nt) chunk, accumulates
//   rows of [BT, K] along BT and writes g_prefix to GM.
//
// Architecture / dataflow:
//   vec_only
//   One logical work item handles one (batch, head, nt) chunk. Each chunk
//   iterates over BT rows, tiling K into fixed-width Vec segments for
//   load-accumulate-store.
//
// Key PTO ops used:
//   TLOAD, TADD, TSTORE, TEXPANDS, set_flag, wait_flag, pipe_barrier
// ============================================================================

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

constexpr int32_t CHUNK_SIZE = 64;
constexpr int32_t ELEMENTS_PER_TILE = 1024;

// UB address map
constexpr int32_t SRC_UB_ADDR = 0x0000;
constexpr int32_t ACC_UB_ADDR = 0x1000;

static_assert(
    static_cast<int64_t>(ACC_UB_ADDR) + ELEMENTS_PER_TILE * 4 <= 196608,
    "UB footprint exceeds A2/A3 capacity (192KB)");

template <typename T>
AICORE void stage_kernel(
    __gm__ T* g_chunked,
    __gm__ T* g_prefix,
    int64_t total_chunks,
    int64_t k,
    uint64_t ffts_addr) {

  set_ffts_base_addr(ffts_addr);

#if defined(__DAV_C220_VEC__)
  auto vid = get_subblockid();
  if (vid != 0) return;
  set_mask_norm();
  set_vector_mask(-1, -1);

  const int64_t core_idx = static_cast<int64_t>(get_block_idx());
  const int64_t block_num = static_cast<int64_t>(get_block_num());

  // Family A: fixed 1D Vec surface (COOK-§0.5)
  using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;
  using TileData =
      Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;

  const int64_t row_stride = k;
  const int64_t chunk_stride = static_cast<int64_t>(CHUNK_SIZE) * k;

  for (int64_t chunk_id = core_idx; chunk_id < total_chunks; chunk_id += block_num) {
    __gm__ T* chunk_in = g_chunked + chunk_id * chunk_stride;
    __gm__ T* chunk_out = g_prefix + chunk_id * chunk_stride;

    // Tile the K dimension into fixed-width Vec segments
    for (int64_t k_off = 0; k_off < k; k_off += ELEMENTS_PER_TILE) {
      const int32_t cur_cols = static_cast<int32_t>(
          (k - k_off) < ELEMENTS_PER_TILE
              ? (k - k_off)
              : static_cast<int64_t>(ELEMENTS_PER_TILE));

      TileData src_tile(1, cur_cols);
      TileData acc_tile(1, cur_cols);
      TASSIGN(src_tile, SRC_UB_ADDR);
      TASSIGN(acc_tile, ACC_UB_ADDR);

      TEXPANDS(acc_tile, static_cast<T>(0));
      pipe_barrier(PIPE_V);

      // Prefix scan over BT rows
      for (int32_t t = 0; t < CHUNK_SIZE; ++t) {
        __gm__ T* row_in = chunk_in + static_cast<int64_t>(t) * row_stride + k_off;
        __gm__ T* row_out = chunk_out + static_cast<int64_t>(t) * row_stride + k_off;

        GlobalData src_global(row_in);
        GlobalData dst_global(row_out);
        TASSIGN(src_global, row_in);
        TASSIGN(dst_global, row_out);

        TLOAD(src_tile, src_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TADD(acc_tile, acc_tile, src_tile);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(dst_global, acc_tile);
        pipe_barrier(PIPE_ALL);
      }
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_gate_cumsum(
    __gm__ uint8_t* g_chunked,
    __gm__ uint8_t* g_prefix,
    int64_t total_chunks,
    int64_t k,
    uint64_t ffts_addr) {
  stage_kernel<float>(
      reinterpret_cast<__gm__ float*>(g_chunked),
      reinterpret_cast<__gm__ float*>(g_prefix),
      total_chunks, k, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim,
    void* stream,
    uint8_t* g_chunked,
    uint8_t* g_prefix,
    int64_t total_chunks,
    int64_t k) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_gate_cumsum<<<block_dim, nullptr, stream>>>(
      g_chunked, g_prefix, total_chunks, k, ffts_addr);
}
```

---

## EX-§3: Full Cube+Vec Pipeline Example (contraction stage)

Complete Cube+Vec pipeline kernel. Archetype: `cube_vec_pipeline`.

Stage: matrix contraction `C = A @ B` where Cube computes the GEMM and Vec
post-processes the result (e.g., scaling, masking). Demonstrates:
- L1 tile loading and TEXTRACT feed chain (COOK-§8.7)
- Cross-core FFTS bootstrap and sync (COOK-§8.6)
- One-shot GEMM pattern (COOK-§8.7)
- L0C → GM store (COOK-§8.9)
- Vec post-processing via TLOAD from GM workspace (COOK-§8.12)

```cpp
// ============================================================================
// contraction_example.cpp — Cube+Vec pipeline example
//
// Stage role:
//   Computes C = A @ B using Cube TMATMUL, then Vec applies element-wise
//   post-processing (scaling) on the result.
//
// Architecture / dataflow:
//   cube_vec_pipeline
//   Cube core: TLOAD A,B into L1 → TEXTRACT to L0 → TMATMUL → TSTORE L0C to GM
//   Vec core: wait for result → TLOAD from GM → TSCALE → TSTORE final output
//   Cross-core sync via FFTS with bootstrap.
//
// Key PTO ops used:
//   TLOAD, TSTORE, TEXTRACT, TMATMUL, TRESHAPE, TMULS, TCVT,
//   set_flag, wait_flag, set_cross_flag, wait_flag_dev, pipe_barrier
// ============================================================================

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

// Problem dimensions (compile-time tile sizes, runtime outer dims)
constexpr int M_TILE = 64;
constexpr int N_TILE = 128;
constexpr int K_TILE = 64;

// ---- Device-only type aliases (COOK-§8.5) ----
#ifdef __CCE_AICORE__

// L1 staging: ColMajor buffer, RowMajor storage
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C,
                        pto::BLayout::ColMajor, RV, CV,
                        pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L1 transposed: RowMajor buffer, ColMajor storage
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN = pto::Tile<pto::TileType::Mat, T, R, C,
                          pto::BLayout::RowMajor, RV, CV,
                          pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

// L0A left operand
template <typename T, int R, int C, int RV = R, int CV = C>
using TileLeftF = pto::Tile<pto::TileType::Left, T, R, C,
                            pto::BLayout::RowMajor, RV, CV,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L0B right operand
template <typename T, int R, int C, int RV = R, int CV = C>
using TileRightF = pto::Tile<pto::TileType::Right, T, R, C,
                             pto::BLayout::RowMajor, RV, CV,
                             pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

// L0C accumulator
template <typename T, int R, int C, int RV = R, int CV = C>
using TileAccF = pto::Tile<pto::TileType::Acc, T, R, C,
                           pto::BLayout::ColMajor, RV, CV,
                           pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// UB Vec tiles
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C,
                       pto::BLayout::RowMajor, RV, CV,
                       pto::SLayout::NoneBox, 512, P>;

#endif

// L1 addresses (Cube core, 512KB L1)
constexpr int32_t A_L1_ADDR = 0;
constexpr int32_t B_L1_ADDR = A_L1_ADDR + M_TILE * K_TILE * sizeof(half);

// L0 addresses (Cube core, 64KB each)
constexpr int32_t A_L0_ADDR = 0;
constexpr int32_t B_L0_ADDR = M_TILE * K_TILE * sizeof(half);

// L0C address (128KB on A2/A3)
constexpr int32_t C_L0C_ADDR = 0;

// UB addresses (Vec core, 192KB)
constexpr int32_t RESULT_UB_ADDR = 0;
constexpr int32_t SCALED_UB_ADDR = RESULT_UB_ADDR + M_TILE * N_TILE * sizeof(float);
constexpr int32_t MAX_UB_ADDR = SCALED_UB_ADDR + M_TILE * N_TILE * sizeof(float);
static_assert(MAX_UB_ADDR <= 196608,
              "UB footprint exceeds A2/A3 capacity (192 KB)");

// Cross-core flags
constexpr int32_t FLAG_FREE_0 = 0;
constexpr int32_t FLAG_FREE_1 = 1;
constexpr int32_t FLAG_READY_0 = 2;
constexpr int32_t FLAG_READY_1 = 3;

// GM workspace layout per core
constexpr int32_t WS_RESULT_ELEMS = M_TILE * N_TILE;
constexpr int32_t WS_PER_CORE = WS_RESULT_ELEMS * sizeof(half);

// Helper: cross-core flag signaling
template <pipe_t Pipe>
AICORE inline void set_cross_flag(int32_t flag, int32_t mode) {
  int config = 1 | (mode << 4) | (flag << 8);
  ffts_cross_core_sync(Pipe, config);
}

template <typename T>
AICORE void stage_kernel(
    __gm__ half* a_gm,
    __gm__ half* b_gm,
    __gm__ half* c_gm,
    __gm__ half* workspace,
    int64_t total_work,
    int64_t scale_factor_fixed,
    uint64_t ffts_addr)
{
  set_ffts_base_addr(ffts_addr);
  const int64_t cid = static_cast<int64_t>(get_block_idx());
  const int64_t block_num = static_cast<int64_t>(get_block_num());

  __gm__ half* ws_ptr = workspace + cid * WS_PER_CORE;

#if defined(__DAV_C220_CUBE__)
  // ============ CUBE PHASE ============

  // Bootstrap: signal both workspace slots are free (COOK-§8, §8.6)
  set_cross_flag<PIPE_MTE3>(FLAG_FREE_0, 2);
  set_cross_flag<PIPE_MTE3>(FLAG_FREE_1, 2);

  for (int64_t wi = cid; wi < total_work; wi += block_num) {
    const int slot = static_cast<int>(wi & 1);
    const int free_flag = (slot == 0) ? FLAG_FREE_0 : FLAG_FREE_1;
    const int ready_flag = (slot == 0) ? FLAG_READY_0 : FLAG_READY_1;

    __gm__ half* a_ptr = a_gm + wi * M_TILE * K_TILE;
    __gm__ half* b_ptr = b_gm + wi * K_TILE * N_TILE;

    // Wait for workspace slot to be free
    wait_flag_dev(free_flag);

    // Load A into L1 (GM → L1 via MTE1)
    {
      L1Mat<half, M_TILE, K_TILE> a_l1;
      TASSIGN(a_l1, A_L1_ADDR);
      Shape<1,1,1,M_TILE,K_TILE> as;
      Stride<1,1,1,K_TILE,1> ast;
      GlobalTensor<half, decltype(as), decltype(ast)> a_global(a_ptr);
      TLOAD(a_l1, a_global);
    }

    // Load B into L1
    L1Mat<half, K_TILE, N_TILE> b_l1;
    TASSIGN(b_l1, B_L1_ADDR);
    {
      Shape<1,1,1,K_TILE,N_TILE> bs;
      Stride<1,1,1,N_TILE,1> bst;
      GlobalTensor<half, decltype(bs), decltype(bst)> b_global(b_ptr);
      TLOAD(b_l1, b_global);
    }

    // Wait for L1 fills to complete
    auto we = EVENT_ID1;
    set_flag(PIPE_MTE2, PIPE_MTE1, we);
    wait_flag(PIPE_MTE2, PIPE_MTE1, we);
    set_flag(PIPE_M, PIPE_MTE1, we);
    wait_flag(PIPE_M, PIPE_MTE1, we);

    // TEXTRACT: L1 → L0A/L0B (COOK-§8.7 feed chain)
    {
      TileLeftF<half, M_TILE, K_TILE> a_l0;
      TASSIGN(a_l0, A_L0_ADDR);
      TEXTRACT(a_l0, a_l1, 0, 0);

      // B is not transposed → L1Mat feeds TileRight directly
      TileRightF<half, K_TILE, N_TILE> b_l0;
      TASSIGN(b_l0, B_L0_ADDR);
      TEXTRACT(b_l0, b_l1, 0, 0);

      set_flag(PIPE_MTE1, PIPE_M, we);
      wait_flag(PIPE_MTE1, PIPE_M, we);

      // TMATMUL: L0A × L0B → L0C
      TileAccF<float, M_TILE, N_TILE> c_l0c;
      TASSIGN(c_l0c, C_L0C_ADDR);
      TMATMUL(c_l0c, a_l0, b_l0);

      set_flag(PIPE_MTE1, PIPE_MTE2, we);
      wait_flag(PIPE_MTE1, PIPE_MTE2, we);
      set_flag(PIPE_M, PIPE_FIX, we);
      wait_flag(PIPE_M, PIPE_FIX, we);

      // TSTORE: L0C → GM workspace (COOK-§8.9)
      {
        Shape<1,1,1,DYNAMIC,DYNAMIC> gs;
        gs.shape[3] = M_TILE;
        gs.shape[4] = N_TILE;
        GlobalTensor<half, decltype(gs), Stride<1,1,1,N_TILE,1>> ws_gm(
            ws_ptr + slot * WS_RESULT_ELEMS, gs);
        TSTORE(ws_gm, c_l0c);
      }
    }

    // Signal: workspace slot is ready for Vec consumer.
    // CRITICAL: issue the READY signal FROM THE PIPE THAT COMMITTED THE GM
    // STORE -- here the store was an L0C->GM TSTORE on the Cube FIX pipe, so the
    // signal must come from PIPE_FIX (NOT PIPE_MTE3). That is what orders the
    // write before the consumer's wait. Drain the core first. (See COOK-§8.6.)
    pipe_barrier(PIPE_ALL);
    set_cross_flag<PIPE_FIX>(ready_flag, 2);
  }

#elif defined(__DAV_C220_VEC__)
  // ============ VEC PHASE ============

  auto vid = get_subblockid();
  if (vid != 0) return;
  set_mask_norm();
  set_vector_mask(-1, -1);

  for (int64_t wi = cid; wi < total_work; wi += block_num) {
    const int slot = static_cast<int>(wi & 1);
    const int free_flag = (slot == 0) ? FLAG_FREE_0 : FLAG_FREE_1;
    const int ready_flag = (slot == 0) ? FLAG_READY_0 : FLAG_READY_1;

    __gm__ half* c_ptr = c_gm + wi * M_TILE * N_TILE;

    // Wait for Cube to produce the result
    wait_flag_dev(ready_flag);

    // TLOAD: GM workspace → UB (COOK-§8.12)
    UbND<float, M_TILE, N_TILE> result_ub;
    TASSIGN(result_ub, RESULT_UB_ADDR);
    {
      Shape<1,1,1,M_TILE,N_TILE> gs;
      Stride<1,1,1,N_TILE,1> gst;
      GlobalTensor<half, decltype(gs), decltype(gst)> ws_gm(
          ws_ptr + slot * WS_RESULT_ELEMS, gs);
      // Note: TLOAD converts half→float automatically if tile is float
      TLOAD(result_ub, ws_gm);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Vec post-processing: scale the result
    UbND<float, M_TILE, N_TILE> scaled_ub;
    TASSIGN(scaled_ub, SCALED_UB_ADDR);
    TMULS(scaled_ub, result_ub, static_cast<half>(scale_factor_fixed));
    pipe_barrier(PIPE_V);

    // TCVT: float → half for output (COOK-§8.11)
    UbND<half, M_TILE, N_TILE> out_ub;
    TASSIGN(out_ub, RESULT_UB_ADDR);  // reuse RESULT slot after Vec done
    TCVT(out_ub, scaled_ub, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);

    // TSTORE: UB → GM final output
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    {
      Shape<1,1,1,M_TILE,N_TILE> gs;
      Stride<1,1,1,N_TILE,1> gst;
      GlobalTensor<half, decltype(gs), decltype(gst)> c_global(c_ptr, gs);
      TSTORE(c_global, out_ub);
    }
    pipe_barrier(PIPE_ALL);

    // Signal: workspace slot is free for next Cube iteration
    set_cross_flag<PIPE_MTE3>(free_flag, 2);
  }
#endif
}

extern "C" __global__ AICORE void launch_stage(
    __gm__ uint8_t* a_gm,
    __gm__ uint8_t* b_gm,
    __gm__ uint8_t* c_gm,
    __gm__ uint8_t* workspace,
    int64_t total_work,
    int64_t scale_factor_fixed,
    uint64_t ffts_addr)
{
  stage_kernel<float>(
      reinterpret_cast<__gm__ half*>(a_gm),
      reinterpret_cast<__gm__ half*>(b_gm),
      reinterpret_cast<__gm__ half*>(c_gm),
      reinterpret_cast<__gm__ half*>(workspace),
      total_work, scale_factor_fixed, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim,
    void* stream,
    uint8_t* a_gm,
    uint8_t* b_gm,
    uint8_t* c_gm,
    uint8_t* workspace,
    int64_t total_work,
    int64_t scale_factor_fixed)
{
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  uint32_t actual_blocks = (block_dim > 0) ? block_dim : 1;
  launch_stage<<<actual_blocks, nullptr, stream>>>(
      a_gm, b_gm, c_gm, workspace, total_work, scale_factor_fixed, ffts_addr);
}
```

---

## EX-§4: Compile Error → Fix Mapping

Common Bisheng compiler errors and their fixes.

| Error message pattern | Root cause | Fix reference |
|----------------------|------------|---------------|
| `no matching function for call to 'TLOAD'` | Type mismatch between GlobalTensor and Tile (layout or dtype) | Check COOK-§0.5 family; ensure ND↔ND or DN↔DN layout match |
| `static assertion failed: invalid SLayout for Mat tile` | Used `SLayout::NoneBox` on Mat | Use `SLayout::RowMajor` (L1Mat) or `SLayout::ColMajor` (L1MatZN) — COOK-§8.5 |
| `static assertion failed: TEXTRACT feed mismatch` | Wrong destination tile for TEXTRACT source layout | Follow COOK-§8.7 feed chain: L1Mat→TileLeft or TileRight, L1MatZN→TileRight only |
| `use of undeclared identifier 'Tile'` | PTO types visible to host compiler | Wrap all PTO types under `#ifdef __CCE_AICORE__` — SKILL C2 |
| `no matching function for call to 'TMATMUL'` | L0A/L0B tile layout mismatch | Check Left=RowMajor/RowMajor, Right=RowMajor/ColMajor — COOK-§8.5 |
| `call to '__gm__' address space function from non-device function` | `__gm__` used outside `AICORE` scope | Move GM pointer usage inside device function — SKILL C3 |
| `undefined reference to 'ffts_cross_core_sync'` | Missing `<runtime/rt_ffts.h>` include | Add `#include <runtime/rt_ffts.h>` — COOK-§8.6 |
| `static assertion failed: UB footprint exceeds` | UB address map exceeds 192KB | Reduce tile sizes or live buffer count — PLAT-§UB, COOK-§4 |
| `no matching function for call to 'TCVT'` | Missing RoundMode argument | Add `pto::RoundMode::CAST_NONE` as third arg — COOK-§8.11 |
| `redefinition of 'launch_stage'` | Multiple `launch_*` definitions | Keep exactly one `extern "C" __global__ AICORE void launch_*` — SKILL C3 |
| `no viable conversion from 'float' to 'half'` | Implicit dtype conversion in tile ops | Use `TCVT` for explicit conversion — COOK-§8.11 |
| `'pipe_barrier' was not declared` | Missing `using namespace pto;` or device guard issue | Ensure `using namespace pto;` under `__CCE_AICORE__` — SKILL C2 |

---

## EX-§5: Softmax Pattern (Vec-Only)

Softmax is a common normalization pattern: `softmax(x) = exp(x) / sum(exp(x))`.

**Instruction chain**: TLOAD → TEXP → TCOLSUM → TCOLEXPANDDIV → TSTORE

```cpp
// ============================================================================
// softmax.cpp — Softmax normalization stage kernel
//
// Stage role:
//   Computes softmax over the column dimension: exp(x) / sum(exp(x))
//   For each row, normalizes across all columns.
//
// Architecture / dataflow:
//   vec_only
//   One work item = one (batch, seq, head) tuple. Processes [1, cols] per item.
//   Uses fused TCOLEXPANDDIV to combine broadcast and division.
//
// Key PTO ops used:
//   TLOAD, TEXP, TCOLSUM, TCOLEXPANDDIV, TSTORE, pipe_barrier
// ============================================================================

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

// Tile dimensions: process one row of length COLS at a time
constexpr int COLS = 128;  // runtime-symbolic in real stage, fixed here for example
constexpr int ELEMS_PER_TILE = COLS;

// UB address map
constexpr int32_t INPUT_UB_ADDR = 0;
constexpr int32_t EXP_UB_ADDR = INPUT_UB_ADDR + COLS * sizeof(float);
constexpr int32_t SUM_UB_ADDR = EXP_UB_ADDR + COLS * sizeof(float);
constexpr int32_t OUT_UB_ADDR = SUM_UB_ADDR + 1 * sizeof(float);  // scalar sum
constexpr int32_t MAX_UB_ADDR = OUT_UB_ADDR + COLS * sizeof(float);
static_assert(MAX_UB_ADDR <= 196608,
              "UB footprint exceeds A2/A3 capacity (192 KB)");

#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C,
                       pto::BLayout::RowMajor, RV, CV,
                       pto::SLayout::NoneBox, 512, P>;
#endif

template <typename T>
AICORE void stage_kernel(
    __gm__ T* input,
    __gm__ T* output,
    int64_t total_rows,
    int64_t cols,
    uint64_t ffts_addr)
{
  set_ffts_base_addr(ffts_addr);

#if defined(__DAV_C220_VEC__)
  auto vid = get_subblockid();
  if (vid != 0) return;
  set_mask_norm();
  set_vector_mask(-1, -1);

  const int64_t core_idx = static_cast<int64_t>(get_block_idx());
  const int64_t block_num = static_cast<int64_t>(get_block_num());

  // Tile declarations
  UbND<float, 1, COLS> x_tile(1, cols);
  UbND<float, 1, COLS> exp_tile(1, cols);
  UbND<float, 1, 1> sum_tile(1, 1);
  UbND<float, 1, COLS> out_tile(1, cols);

  TASSIGN(x_tile, INPUT_UB_ADDR);
  TASSIGN(exp_tile, EXP_UB_ADDR);
  TASSIGN(sum_tile, SUM_UB_ADDR);
  TASSIGN(out_tile, OUT_UB_ADDR);

  for (int64_t row = core_idx; row < total_rows; row += block_num) {
    __gm__ T* in_ptr = input + row * cols;
    __gm__ T* out_ptr = output + row * cols;

    // Load input row
    Shape<1,1,1,1,COLS> gs;
    Stride<1,1,1,1,1> gst;
    GlobalTensor<float, decltype(gs), decltype(gst)> in_gm(in_ptr);
    TLOAD(x_tile, in_gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Compute exp(x) for all columns
    TEXP(exp_tile, x_tile);
    pipe_barrier(PIPE_V);

    // Sum across columns: sum_tile = sum(exp_tile)
    TCOLSUM(sum_tile, exp_tile);
    pipe_barrier(PIPE_V);

    // Divide: out_tile[i] = exp_tile[i] / sum_tile (broadcast division)
    TCOLEXPANDDIV(out_tile, exp_tile, sum_tile);
    pipe_barrier(PIPE_V);

    // Store result
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GlobalTensor<float, decltype(gs), decltype(gst)> out_gm(out_ptr);
    TSTORE(out_gm, out_tile);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
#endif
}

extern "C" __global__ AICORE void launch_stage(
    __gm__ uint8_t* input,
    __gm__ uint8_t* output,
    int64_t total_rows,
    int64_t cols,
    uint64_t ffts_addr)
{
  stage_kernel<float>(
      reinterpret_cast<__gm__ float*>(input),
      reinterpret_cast<__gm__ float*>(output),
      total_rows, cols, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim,
    void* stream,
    uint8_t* input,
    uint8_t* output,
    int64_t total_rows,
    uint64_t ffts_addr)
{
  int64_t cols = /* derived from stage spec */;
  uint32_t ffts_len = 0;
  uint64_t ffts_addr_local = 0;
  rtGetC2cCtrlAddr(&ffts_addr_local, &ffts_len);
  launch_stage<<<block_dim, nullptr, stream>>>(
      input, output, total_rows, cols, ffts_addr_local);
}
```

**Key points:**
- Uses `TCOLSUM` to reduce across columns (axis reduction, not scalar loop)
- Uses fused `TCOLEXPANDDIV` instead of separate `TCOLEXPAND` + `TDIV`
- All operations on tiles, no scalar extraction loops
- Verify `TCOLSUM` and `TCOLEXPANDDIV` signatures with MCP before use

---

## EX-§6: Activation Function Patterns

### ReLU (simple pointwise)

**Instruction chain**: TLOAD → TRELU → TSTORE

```cpp
// ReLU: y = max(0, x)
UbND<float, 1, COLS> x_tile(1, cols);
UbND<float, 1, COLS> y_tile(1, cols);
TASSIGN(x_tile, X_UB_ADDR);
TASSIGN(y_tile, Y_UB_ADDR);

TLOAD(x_tile, x_gm);
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

TRELU(y_tile, x_tile);  // pointwise max(0, x)
pipe_barrier(PIPE_V);

set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(y_gm, y_tile);
```

### GELU (Gaussian Error Linear Unit)

GELU has no direct PTO instruction. Approximate using:
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

**Instruction chain**: TLOAD → TMUL → TMULS → TADD → TMULS → TEXP → TADD → TMULS → TMUL → TSTORE

```cpp
// GELU approximation using tanh via exponential
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) ≈ 2*sigmoid(2x) - 1

UbND<float, 1, COLS> x_tile, x3_tile, inner_tile, tanh_tile, out_tile;
// ... allocate and TASSIGN ...

TLOAD(x_tile, x_gm);
// ... sync ...

// x³ = x * x * x
TMUL(x3_tile, x_tile, x_tile);  // x²
TMUL(x3_tile, x3_tile, x_tile); // x³
pipe_barrier(PIPE_V);

// x + 0.044715 * x³
TMULS(inner_tile, x3_tile, 0.044715f);
TADD(inner_tile, x_tile, inner_tile);
pipe_barrier(PIPE_V);

// sqrt(2/π) * (x + 0.044715 * x³)
TMULS(inner_tile, inner_tile, 0.7978845608f);  // sqrt(2/π)
pipe_barrier(PIPE_V);

// tanh approximation: tanh(x) ≈ 2*sigmoid(2x) - 1
// sigmoid(2x) = 1 / (1 + exp(-2x))
TMULS(inner_tile, inner_tile, 2.0f);  // 2x
TMULS(inner_tile, inner_tile, -1.0f); // -2x
TEXP(tanh_tile, inner_tile);           // exp(-2x)
TADDS(tanh_tile, tanh_tile, 1.0f);    // 1 + exp(-2x)
// ... need TRECIP for 1/(1+exp(-2x)), then scale and shift ...

// 0.5 * x * (1 + tanh(...))
TADDS(tanh_tile, tanh_tile, 1.0f);    // 1 + tanh
TMUL(out_tile, x_tile, tanh_tile);
TMULS(out_tile, out_tile, 0.5f);
pipe_barrier(PIPE_V);

TSTORE(out_gm, out_tile);
```

**Key points:**
- GELU requires decomposition into multiple PTO ops
- Consider using `TRECIP` (reciprocal) if available (verify with MCP)
- Alternative: use polynomial approximation if stage spec allows
- Always verify intermediate tile values don't overflow UB budget

### Leaky ReLU (conditional activation)

**Instruction chain**: TLOAD → TCMP → TSEL → TSTORE (or TLRELU if available)

```cpp
// Leaky ReLU: y = x if x > 0, else 0.01 * x
// Check if TLRELU instruction is available via MCP

UbND<float, 1, COLS> x_tile, mask_tile, y_tile;
UbND<float, 1, COLS> leak_tile;
// ... allocate and TASSIGN ...

TLOAD(x_tile, x_gm);
// ... sync ...

// Option 1: Use TLRELU if available
TLRELU(y_tile, x_tile, 0.01f);  // verify signature with MCP
pipe_barrier(PIPE_V);

// Option 2: Decompose using TCMP + TSEL
TEXPANDS(leak_tile, 0.01f);
TMUL(leak_tile, x_tile, leak_tile);  // 0.01 * x
TCMP(mask_tile, x_tile, 0.0f);       // mask = (x > 0)
TSEL(y_tile, mask_tile, x_tile, leak_tile);  // y = mask ? x : 0.01*x
pipe_barrier(PIPE_V);

TSTORE(out_gm, y_tile);
```

**Key points:**
- Prefer fused `TLRELU` if available (verify with MCP)
- `TCMP` generates a boolean mask tile
- `TSEL` performs conditional selection: `mask ? a : b`
- All conditional logic stays on tiles, no scalar branching

---

## EX-§FlashAttention: Vec+Cube Pipeline (from pto-isa CPU tests)

Real flash attention kernel showing Vec↔Cube tile conversion and row-wise softmax.
Source: `pto-isa/tests/cpu/st/testcase/tflashattn/tflashattn_kernel.cpp` — **CPU-SIM compatible**.

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

constexpr int kSeqLen = 64;
constexpr int kHeadDim = 32;

__global__ AICORE void RunTFLASHATTN(__gm__ float *out, __gm__ float *q,
                                      __gm__ float *k, __gm__ float *v)
{
    using GlobalQ = GlobalTensor<float, Shape<1,1,1,kSeqLen,kHeadDim>,
                                 Stride<kSeqLen*kHeadDim,kSeqLen*kHeadDim,
                                        kSeqLen*kHeadDim,kHeadDim,1>>;
    GlobalQ qGlobal(q); // repeat for k, v, out

    using QPlain = Tile<TileType::Vec, float, kSeqLen, kHeadDim,
                        BLayout::RowMajor, kSeqLen, kHeadDim>;
    using ScoresPlain = Tile<TileType::Vec, float, kSeqLen, kSeqLen,
                              BLayout::RowMajor, kSeqLen, kSeqLen>;
    using LeftQ = TileLeft<float, kSeqLen, kHeadDim>;
    using RightKT = TileRight<float, kHeadDim, kSeqLen>;
    using AccScores = TileAcc<float, kSeqLen, kSeqLen>;

    TLOAD(qTile, qGlobal);  TLOAD(kTile, kGlobal);  TLOAD(vTile, vGlobal);

    // Tile type conversion: Vec → Cube via TMOV (zero-cost)
    TMOV(qLeft, qTile);
    TTRANS(ktTile, kTile, kTile);
    TMOV(kRight, ktTile);

    TMATMUL(scoresAcc, qLeft, kRight);
    TMOV(scores, scoresAcc);
    TMULS(scores, scores, 1.0f / std::sqrt(kHeadDim));

    // Row-wise softmax (Vec ops)
    TROWMAX(rowMax, scores, scores);
    TROWEXPANDSUB(scoresCentered, scores, rowMax);
    TEXP(expScores, scoresCentered);
    TROWSUM(rowSum, expScores, expScores);
    TROWEXPANDDIV(probs, expScores, rowSum);

    TMOV(pLeft, probs);
    TMOV(vRight, vTile);
    TMATMUL(outAcc, pLeft, vRight);
    TSTORE(oGlobal, outAcc);
}

void LaunchTFLASHATTN(float *out, float *q, float *k, float *v, void *stream) {
    (void)stream;
    RunTFLASHATTN(out, q, k, v);
}
```

**Key patterns:**
- `TMOV` converts Vec↔Left/Right/Acc (zero-cost tile type cast)
- `TTRANS` transposes a Vec tile (e.g., K → K^T for matmul)
- `TMATMUL` on TileLeft×TileRight→TileAcc (Cube matmul)
- `TROWMAX`, `TROWEXPANDSUB`, `TROWEXPANDDIV` for row-wise softmax
- Direct `Launch*` calls kernel — NO FFTS, NO multi-core dispatch
- Auto memory assignment (`__PTO_AUTO__`) avoids manual UB address maps
- **CPU-SIM compatible** — this exact code runs in the pto-isa CPU test suite

---

## EX-§MatMulL0Pipeline: L1/L0 Memory Hierarchy (from pto-kernels)

Source: `pto-kernels/examples/jit_cpp/matmul_swizzle/matmul_original_pto.cpp`

```cpp
using namespace pto;

template <typename InputT, typename OutputT, uint32_t matrix_size>
AICORE void runKernelSimpleMatMul(__gm__ InputT* a, __gm__ InputT* b,
                                   __gm__ OutputT* c) {
    // L1 tiles: GM staging buffer (Mat, ColMajor)
    using TileL1AB = Tile<TileType::Mat, InputT, matrix_size, matrix_size,
                          BLayout::ColMajor, matrix_size, matrix_size,
                          SLayout::RowMajor, 512>;
    // L0 tiles: Cube compute
    using TileL0A = TileLeft<InputT, matrix_size, matrix_size>;
    using TileL0B = TileRight<InputT, matrix_size, matrix_size>;
    using TileL0C = TileAcc<OutputT, matrix_size, matrix_size>;

    TASSIGN(a_l1, 0x0);
    TASSIGN(b_l1, 0x0 + tile_len * sizeof(InputT));
    // L0A/L0B/L0C: distinct scratchpads, all at 0x0
    TASSIGN(a_l0, 0x0); TASSIGN(b_l0, 0x0); TASSIGN(c_l0, 0x0);

    TLOAD(a_l1, a_global_in);
    TLOAD(b_l1, b_global_in);
    set_flag(PIPE_MTE2, PIPE_MTE1, event_id);
    wait_flag(PIPE_MTE2, PIPE_MTE1, event_id);

    for (uint32_t k_iter = 0; k_iter < k_blocks; k_iter++) {
        TEXTRACT_L0A(a_l0, a_l1, k_iter);
        TEXTRACT_L0B(b_l0, b_l1, k_iter);
        if (k_iter == 0) TMATMUL(c_l0, a_l0, b_l0);
        else             TMATMUL_ACC(c_l0, c_l0, a_l0, b_l0);
    }
    TSTORE(c_global_out, c_l0);
}
```

**Key patterns:**
- L1 tiles: `TileType::Mat` + `ColMajor` — for GM↔L1 data movement
- L0 tiles: `TileLeft`, `TileRight`, `TileAcc` — for Cube computation
- `TEXTRACT_L0A/L0B`: slice L1 tile into L0 (K-dimension tiling)
- `TMATMUL_ACC`: accumulate into existing accumulator (K-dimension loop)
- L0A/L0B/L0C share same UB offset (different physical scratchpads)

---

## EX-§LayernormMultiPhase: UB Reuse Strategy (from pto-kernels)

Source: `pto-kernels/examples/jit_cpp/layernorm/kernel_layernorm.cpp`

```cpp
using namespace pto;

namespace UbLayout {
constexpr unsigned UB_BASE = 0x00000;
// Stats and Chunk phases OVERLAY the same UB region — only one is live at a time.
namespace Stats {
constexpr unsigned X_HALF_BASE = UB_BASE;  // same offset as Chunk
constexpr unsigned PHASE_END = ...;
}
namespace Chunk {
constexpr unsigned X_HALF_BASE = UB_BASE;  // REUSES Stats region
constexpr unsigned GAMMA_HALF_BASE = ...;
}
}  // namespace UbLayout

static_assert(UbLayout::Stats::PHASE_END <= UB_USABLE_BYTES);
```

**Key patterns:**
- Namespace-organized UB layouts make multi-phase kernels readable
- UB regions intentionally OVERLAID between phases (saves memory)
- Each phase starts from `UB_BASE` for clean mental model
- `static_assert` at end validates total usage against 192KB budget

---

## EX-§SwiGLUPingPong: Double Buffer (from pto-kernels)

Source: `pto-kernels/examples/jit_cpp/swiglu/swiglu.cpp`

```cpp
constexpr uint32_t UB_SLOT_BYTES = (192 * 1024) / 6;
constexpr unsigned X0_PING = 0x00000;
constexpr unsigned X1_PING = X0_PING + X0_BUFFER_BYTES;
constexpr unsigned Y_PING  = X1_PING + X1_BUFFER_BYTES;
constexpr unsigned X0_PONG = Y_PING + Y_BUFFER_BYTES;
constexpr unsigned X1_PONG = X0_PONG + X0_BUFFER_BYTES;
constexpr unsigned Y_PONG  = X1_PONG + X1_BUFFER_BYTES;
static_assert(Y_PONG + Y_BUFFER_BYTES <= 192*1024);

for (uint32_t tile = 0; tile < num_tiles; tile++) {
    bool ping = (tile % 2) == 0;
    unsigned x0_addr = ping ? X0_PING : X0_PONG;
    // Load into ping buffer, compute on pong buffer (loaded last iteration)
    TASSIGN(x0_tile, x0_addr);
    TLOAD(x0_tile, x0_gm);
}
```

**Key patterns:**
- Ping-pong double buffering overlaps load with compute
- `ping ? PING_ADDR : PONG_ADDR` selects active buffer
- `static_assert` validates UB budget
- ALIGN_UP/DIV_ROUNDUP macros for dimension rounding
- All conditional logic stays on tiles, no scalar branching
