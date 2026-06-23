# Kernel Build & Validation Patterns

> **Stable ID**: `BUILD-§`
> **Source**: `examples/pto-kernels/src/tasks/pto_compile.py` and `examples/pto-kernels/src/common.py`

## Compilation

Kernels are compiled with `bisheng` from the active CANN toolkit (default 9.0.0)
in CCE mode. Source `set_env.sh` first and resolve all toolkit paths from
`$ASCEND_HOME_PATH` — do NOT hardcode a CANN version path.

```bash
source /usr/local/Ascend/cann/set_env.sh   # -> ASCEND_HOME_PATH (default: cann-9.0.0)
"$ASCEND_HOME_PATH/bin/bisheng" -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=gnu++17 \
  --cce-aicore-arch=dav-c220 \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-function-stack-size=0x8000 \
  -Wno-macro-redefined -Wno-ignored-attributes \
  -I<kernel_dir> -I<example>/include \
  -I<pto_isa_root> -I<pto_isa_root>/include \
  -I"$ASCEND_HOME_PATH/include" \
  -I"$ASCEND_HOME_PATH/pkg_inc" \
  -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
  -I"$ASCEND_HOME_PATH/pkg_inc/profiling" \
  kernel.cpp -o kernel.so
```

Key flags:
- `-xcce` — CCE language mode; auto-defines `__CCE_AICORE__`, `__DAV_C220_VEC__`
- `--cce-aicore-arch=dav-c220` — target NPU architecture (A2/A3)
- `-std=gnu++17` — C++17 with GNU extensions (NOT c++20)
- `-DMEMORY_BASE` — required by PTO memory model
- No `-D__CPU_SIM` — CCE provides its own device runtime, no GCC STL needed
- No scalar math functions (expf, logf, etc.) — use PTO tile ops (TEXP, etc.)

## call_kernel: Standard Host Entrypoint

```cpp
extern "C" void call_kernel(
    uint32_t block_dim, void* stream,
    uint8_t* input0, uint8_t* input1, uint8_t* output, uint8_t* workspace,
    int64_t total_work, int64_t bt, int64_t k, int64_t nt, int64_t hv)
{
  uint32_t ffts_len = 0; uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  uint32_t blocks = (block_dim > 0) ? block_dim : 1;
  launch_my_kernel<<<blocks, nullptr, stream>>>(
      input0, input1, output, workspace,
      total_work, bt, k, nt, hv);
}
```

The `<<<...>>>` syntax is a CCE compiler extension available in `-xcce` mode.

## Platform Guards

Kernel compute bodies:
```cpp
AICORE void stage_kernel(...) {
  set_ffts_base_addr(0);
#if defined(__DAV_C220_VEC__) || defined(__CPU_SIM)
  auto vid = get_subblockid(); if (vid != 0) return;
  set_mask_norm(); set_vector_mask(-1, -1);
  // ... compute body
#endif
}
```

Launch functions:
```cpp
extern "C" __global__ AICORE void launch_my_kernel(...) {
#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
  stage_kernel(...);
#endif
}
```

## C19: Reading Post-Vec-Op Values

Scalar math functions (expf, logf, etc.) are NOT available in CCE device code.
To read tile values after TEXP/TMUL/TADD/etc., use the TSTORE→GM→TLOAD round-trip:

```cpp
// Compute exp(gate) on tile
TEXP(gate_tile, gate_tile); pipe_barrier(PIPE_V);
// C19 round-trip: store to GM, reload; GetValue is then safe
GF wsg(reinterpret_cast<__gm__ float*>(workspace), shape);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(wsg, gate_tile); pipe_barrier(PIPE_ALL);
TLOAD(gate_tile, wsg);
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
// Now gate_tile.GetValue(j) returns exp(gate[j]) — safe!
float exp_gate_j = gate_tile.GetValue(j);
```

## msprof op simulator Validation

```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH"
msprof op simulator --output=<dir> --aic-metrics=PipeUtilization \
    --launch-count=1 --soc-version=Ascend910B1 \
    python validation_script.py kernel.so
```

Output dir must be non-world-writable (`chmod 700`). Validation scripts must NOT
call `torch.npu.synchronize()` (hangs) and must compare on-device without `.cpu()`.
