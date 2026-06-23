# NPU Launch Patterns

Reference patterns for launching PTO kernels on Ascend NPU via ctypes.

## Required Imports

```python
import torch
import torch_npu  # noqa: F401
import ctypes
import argparse
import sys
import time
```

**Note**: `import torch_npu` must appear directly at module level, not inside try/except or version check.

## Device Setup

```python
torch.npu.set_device(0)
device = torch.device("npu")
```

**Never use** `"npu:0"` or `"npu:1"` — use `"npu"` and set device via `torch.npu.set_device(0)`.

## Stream Pointer Extraction

```python
stream = torch.npu.current_stream()
stream_ptr = getattr(stream, "_as_parameter_", None)
```

**Critical**: Use `_as_parameter_` attribute, NOT `stream.stream_ptr` or `stream.npu_stream`.

**Do NOT wrap stream_ptr in ctypes.c_void_p()** — the `_as_parameter_` attribute already returns a c_void_p object. Wrapping it again causes TypeError.

Pass `stream_ptr` directly in the call_kernel args list:
```python
call_kernel(block_dim, stream_ptr, tensor_ptr, ...)
```

## Tensor Allocation on NPU

### Standard allocation
```python
tensor = torch.randn(shape, dtype=torch.float32, device='npu')
```

### Deterministic seeding (CPU generator + NPU transfer)
```python
g = torch.Generator().manual_seed(seed)
tensor = torch.randn(shape, generator=g).to(device)
```

**Critical**: Do NOT pass `device='npu'` to randn when also passing a CPU generator:
```python
# WRONG — crashes:
tensor = torch.randn(shape, generator=g, device='npu')

# CORRECT:
tensor = torch.randn(shape, generator=g).to(device)
```

## Synchronization

Always synchronize before and after kernel calls:

```python
torch.npu.synchronize()
call_kernel(...)
torch.npu.synchronize()
```

## ctypes Loading and Launch Pattern

```python
lib = ctypes.CDLL(so_path)
call_kernel = lib.call_kernel

# Define argtypes from KernelABI.arguments
call_kernel.argtypes = [
    ctypes.c_uint32,  # block_dim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # input tensor
    ctypes.c_void_p,  # output tensor
    ctypes.c_int64,   # total_work
    ctypes.c_int64,   # problem dim 1 (e.g., bt)
    ctypes.c_int64,   # problem dim 2 (e.g., k)
    # ... all arguments from KernelABI.arguments in exact order
]
call_kernel.restype = None

# Get tensor pointers
input_ptr = input_tensor.data_ptr()
output_ptr = output_tensor.data_ptr()

# Compute total_work and block_dim
total_work = ...  # derive from StageSpec dimensions
block_dim = min(total_work, 256)  # or from StageSpec

# Call kernel
call_kernel(
    block_dim,
    stream_ptr,
    input_ptr,
    output_ptr,
    total_work,
    bt,
    k,
    # ... all problem dimensions from StageSpec.problem
)
```

**Critical**: The argtypes list must include ALL arguments from KernelABI.arguments in exact order. Never skip or reorder arguments.

## Dimension Derivation from StageSpec

### Extract problem dimensions
```python
# From StageSpec.problem dict — extract only keys that exist
problem = stage_spec['stage']['problem']

# Read dimensions that are present in StageSpec.problem
# Do NOT use .get() with hard-coded fallback defaults
bt = problem['BT'] if 'BT' in problem else None
k = problem['K'] if 'K' in problem else None
hv = problem['HV'] if 'HV' in problem else None
h = problem['H'] if 'H' in problem else None
v = problem['V'] if 'V' in problem else None

# If a dimension is missing, it means this stage does not use it
# Do NOT invent a default value
```

**Never hard-code dimension values** — always derive from StageSpec.problem.

### Compute total_work
```python
# Derive from StageSpec output shapes
# Example: output shape is [B, HV, NT, BT, K]
B = 1  # batch size
HV = problem['HV']
NT = n_seq  # from --n-seq CLI arg
BT = l_seg  # from --l-seg CLI arg
K = problem['K']

total_work = B * HV * NT * BT
```

**Never pass total_work=0 or block_dim=0** — these cause DDR crashes.

### Distinguish helper constants from logical dimensions

**Helper constants** (compile-time, not ABI-visible):
- `COL_BLOCK` — tile width for workspace allocation
- `VISIBLE_SOURCE_BT` — internal block size for tail handling
- `WORKSPACE_SIZE` — buffer size for intermediate results

**Logical dimensions** (ABI-visible, from StageSpec.problem):
- `BT` — chunk size
- `K` — head dimension
- `HV` — value head count
- `H` — query/key head count
- `V` — value dimension

Only lift helper constants into runnable artifacts when they are truly ABI-visible logical constraints.

## CLI Argument Pattern

```python
parser = argparse.ArgumentParser()
parser.add_argument('input', help='Path to kernel .so')
parser.add_argument('--stage-spec', required=True, help='Path to StageSpec JSON file')
parser.add_argument('--n-seq', type=int, default=16, help='Number of sequences (NT)')
parser.add_argument('--l-seg', type=int, default=128, help='Segment length (BT)')
parser.add_argument('--num-tests', type=int, default=10, help='Number of validation cases')
# ... other args
args = parser.parse_args()

import json
with open(args.stage_spec, 'r') as f:
    stage_spec = json.load(f)

so_path = args.input
n_seq = args.n_seq
l_seg = args.l_seg
```

**CLI semantics must stay aligned with StageSpec relations**:
- `--n-seq` stays sequence-derived (NT)
- `--l-seg` stays chunk/BT-derived
- Do not remap `--n-seq` to batch size
- Do not fix logical BT to a constant unless StageSpec proves it

## Simulation Compatibility (msprof op simulator)

Scripts generated by this skill must work under `msprof op simulator --soc-version=Ascend910B1`.

### Do NOT call synchronize

```python
# WRONG — hangs in simulation
torch.npu.synchronize()
call_kernel(...)
torch.npu.synchronize()

# CORRECT — no synchronize
call_kernel(...)
```

### Compare on-device (no .cpu())

```python
# WRONG — .cpu() triggers synchronize, hangs in simulation
actual = A_out.cpu()

# CORRECT — compare on-device
expected = reference_model(*inputs_npu)   # compute reference on NPU
check_accuracy(A_out, expected, desc)      # compare on NPU
```

### Kernel-side patterns for simulation safety

These patterns prevent hardware faults caught by `msprof op simulator`:

**Work-item loop bound must match memory stride**: `loop_bound = total_work / elements_per_iteration`.
The `total_work` passed by the ABI is the total output element count. If each
iteration processes `N` elements, the loop must iterate `total_work / N` times,
and the memory offset multiplies by `N`:

```cpp
// General pattern: elements_per_iteration and stride_per_row derived from kernel layout
int64_t rows_per_group = /* rows processed per work item */;
int64_t cols_per_row   = /* elements per row in source tensor */;
int64_t cols_per_out   = /* elements per row in output tensor */;
int64_t num_iters = (rows_per_group > 0) ? total_work / rows_per_group : 0;
for (int64_t gi = get_block_idx(); gi < num_iters; gi += block_num) {
    g_base = g_prefix + gi * rows_per_group * cols_per_row;
    a_base = A_output + gi * rows_per_group * cols_per_out;
}
```

The `elements_per_iteration` value is kernel-specific — trace the memory
addressing to verify: `max_index * elements_per_iteration * sizeof(elem)` must
stay within the tensor allocation.

**Block count guard**: Prevent infinite loop when simulation returns 0:

```cpp
int64_t block_num = static_cast<int64_t>(get_block_num());
if (block_num <= 0) block_num = 1;
```

**TCMP dtype consistency**: All tiles must use the same element type:

```cpp
// WRONG: dst is int32_t, src tiles are float
// CORRECT: dst, src0, src1 all use float
using MaskRow = Tile<TileType::Vec, float, 1, MAX_BT, ...>;
```
