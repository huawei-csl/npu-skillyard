# Validation Script Patterns

Reference patterns for generating ValidationScript that checks numerical correctness of PTO kernels.

## Complete ValidationScript Template

```python
#!/usr/bin/env python3
"""Validation script for {stage_name} kernel.

Generates deterministic test cases, runs reference model on NPU,
runs kernel on NPU, and compares results on-device with two-tier accuracy check.
Works with both real NPU hardware and msprof op simulator.
"""

import torch
import torch_npu  # noqa: F401
import ctypes
import argparse
import sys
import time
import numpy as np

# ============================================================================
# Device setup
# ============================================================================
torch.npu.set_device(0)
device = torch.device("npu")

# ============================================================================
# Reference model (inline or import)
# ============================================================================
{REFERENCE_MODEL_CODE}

# ============================================================================
# Test case generation
# ============================================================================
def generate_default_cases(stage_spec):
    """Generate DEFAULT_CASES from StageSpec.problem and production_dimensions.
    
    Returns list of dicts with keys: bt, nt, seed, description
    """
    problem = stage_spec['stage']['problem']
    prod_dims = stage_spec['stage'].get('production_dimensions', {})
    
    # Derive BT values from StageSpec (not hard-coded)
    # Use problem['BT'] as baseline, generate 6 values around it
    base_bt = problem['BT']  # BT must exist in StageSpec.problem for stages that use it
    bt_values = [base_bt // 4, base_bt // 2, base_bt, base_bt * 2, base_bt * 3, base_bt * 4]
    bt_values = [max(4, bt) for bt in bt_values]  # ensure minimum BT=4
    
    # Map BT → NT (derive from StageSpec if available, else use heuristic)
    # Heuristic: smaller BT needs larger NT to keep total tokens reasonable
    bt_to_nt = {bt: max(1, 256 // bt) for bt in bt_values}
    
    cases = []
    for bt in bt_values:
        nt = bt_to_nt[bt]
        # Two seeds per BT value
        for seed in [42, 123]:
            cases.append({
                'bt': bt,
                'nt': nt,
                'seed': seed,
                'description': f'BT={bt}, NT={nt}, seed={seed}'
            })
    return cases

# ============================================================================
# Kernel launch wrapper
# ============================================================================
def setup_kernel(so_path, stage_spec):
    """Load kernel .so and configure ctypes interface."""
    lib = ctypes.CDLL(so_path)
    call_kernel = lib.call_kernel
    
    # Define argtypes from KernelABI.arguments (passed via stage_spec or closure)
    # This must match the exact order from KernelABI
    call_kernel.argtypes = [
        ctypes.c_uint32,  # block_dim
        ctypes.c_void_p,  # stream
        # ... tensor pointers ...
        ctypes.c_int64,   # total_work
        # ... problem dimensions from stage_spec['stage']['problem'] ...
    ]
    call_kernel.restype = None
    return call_kernel

def call_kernel_wrapper(call_kernel, input_tensors, output_tensors, stage_spec, n_seq, l_seg):
    """Call kernel with proper argument ordering from KernelABI."""
    stream = torch.npu.current_stream()
    stream_ptr = getattr(stream, "_as_parameter_", None)
    
    problem = stage_spec['stage']['problem']
    bt = problem['BT']
    k = problem['K'] if 'K' in problem else None
    hv = problem['HV'] if 'HV' in problem else None
    
    # Compute total_work from StageSpec dimensions
    # Use production_dimensions if available, otherwise derive from problem
    B = 1
    prod_dims = stage_spec['stage'].get('production_dimensions', {})
    if 'HV' in prod_dims:
        hv_val = prod_dims['HV']
    elif hv is not None:
        hv_val = hv
    else:
        hv_val = 1  # fallback for stages without HV
    
    total_work = B * hv_val * n_seq * l_seg
    block_dim = min(total_work, 256)
    
    # Get tensor pointers
    tensor_ptrs = [t.data_ptr() for t in input_tensors + output_tensors]
    
    call_kernel(
        block_dim,
        stream_ptr,
        *tensor_ptrs,
        total_work,
        l_seg,  # bt
        k,
        # ... other problem dimensions in KernelABI order ...
    )

# ============================================================================
# Two-tier accuracy check
# ============================================================================
def check_accuracy(actual, expected, description, rtol=1e-3, atol=1e-3):
    """Two-tier accuracy check:
    1. Primary: torch.testing.assert_close
    2. Fallback: RMSE / mean(|expected|) and R²
    """
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        print(f"✓ {description}: PASS (assert_close)")
        return True
    except AssertionError as e:
        # Fallback: compute RMSE and R²
        diff = (actual - expected).abs()
        rmse = torch.sqrt((diff ** 2).mean()).item()
        mean_abs_expected = expected.abs().mean().item()
        
        if mean_abs_expected < 1e-9:
            # Near-zero signal: require absolute RMSE < 5e-4
            if rmse < 5e-4:
                print(f"✓ {description}: PASS (near-zero RMSE={rmse:.6f})")
                return True
            else:
                print(f"✗ {description}: FAIL (near-zero RMSE={rmse:.6f} > 5e-4)")
                return False
        
        ratio = rmse / mean_abs_expected
        ss_res = ((expected - actual) ** 2).sum().item()
        ss_tot = ((expected - expected.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        if ratio <= 0.05 and r2 >= 0.99:
            print(f"✓ {description}: PASS (fallback: RMSE/mean={ratio:.4f}, R²={r2:.4f})")
            return True
        else:
            print(f"✗ {description}: FAIL (RMSE/mean={ratio:.4f}, R²={r2:.4f})")
            print(f"  assert_close error: {e}")
            return False

# ============================================================================
# Shape derivation from StageSpec
# ============================================================================
def resolve_shape(symbolic_shape, stage_spec, n_seq, l_seg):
    """Resolve a symbolic shape like ['B', 'HV', 'NT', 'BT', 'K'] to numeric tuple.
    
    Maps symbolic dimension names to values from:
    - stage.problem for compile-time constants
    - CLI args for runtime values (NT from --n-seq, BT from --l-seg)
    - Fixed defaults (B=1)
    """
    problem = stage_spec['stage']['problem']
    
    # Dimension name → value mapping
    # CLI overrides take precedence, then problem dict, then sensible defaults
    dim_map = {
        'B': 1,           # batch size, usually 1 for single-stage testing
        'T': n_seq * l_seg,  # total tokens
        'NT': n_seq,      # number of chunks
        'BT': l_seg,      # chunk size
    }
    # Override with problem dict values for compile-time constants
    dim_map.update(problem)
    # CLI values override problem defaults for runtime dimensions
    dim_map['NT'] = n_seq
    dim_map['BT'] = l_seg
    
    shape = []
    for dim_name in symbolic_shape:
        if dim_name in dim_map:
            shape.append(dim_map[dim_name])
        else:
            raise ValueError(
                f"Unknown dimension '{dim_name}' in shape {symbolic_shape}. "
                f"Available dimensions: {list(dim_map.keys())}"
            )
    return tuple(shape)

# ============================================================================
# Validation case runner
# ============================================================================
def validate_case(call_kernel, case, stage_spec):
    """Run one validation case: generate data, run reference, run kernel, compare."""
    bt = case['bt']
    nt = case['nt']
    seed = case['seed']
    description = case['description']
    
    # Generate deterministic input on CPU, then move to NPU
    g = torch.Generator().manual_seed(seed)
    
    # Derive input shapes from StageSpec.stage.inputs (not hard-coded)
    inputs = stage_spec['stage']['inputs']
    input_tensors_cpu = []
    for inp in inputs:
        shape = resolve_shape(inp['shape'], stage_spec, nt, bt)
        dtype = getattr(torch, inp.get('dtype', 'float32'))
        tensor = torch.randn(shape, generator=g, dtype=dtype)
        input_tensors_cpu.append(tensor)
    
    input_tensors_npu = [t.to(device) for t in input_tensors_cpu]
    
    # Run reference model on-device (NPU) — avoids .cpu() sync that hangs in simulation
    expected_npu = reference_model(*input_tensors_npu)
    
    # Allocate output on NPU (derive shape from reference output)
    if isinstance(expected_npu, torch.Tensor):
        actual_npu = [torch.zeros_like(expected_npu)]
    else:
        actual_npu = [torch.zeros_like(e) for e in expected_npu]
    
    # Run kernel
    call_kernel_wrapper(call_kernel, input_tensors_npu, actual_npu, stage_spec, nt, bt)
    
    # Compare on-device — no .cpu() copy needed (avoids synchronize hang)
    if isinstance(expected_npu, torch.Tensor):
        success = check_accuracy(actual_npu[0], expected_npu, description)
    else:
        success = True
        for i, (exp, act) in enumerate(zip(expected_npu, actual_npu)):
            if not check_accuracy(act, exp, f"{description} output[{i}]"):
                success = False
    
    return success

# ============================================================================
# CLI and main
# ============================================================================
# Embed the StageSpec directly so the script is self-contained and
# does NOT require --stage-spec at runtime (downstream tasks cannot
# auto-discover it).  Replace the placeholder dict below with the
# actual StageSpec JSON content — do NOT use file loading.
STAGE_SPEC = {
    "stage": {
        "name": "{stage_name}",
        "problem": {{"K": 128, "V": 128, ...}},
        "production_dimensions": {{"HV": 256, ...}},
        "inputs": [...],
        "outputs": [...]
    }
}

def main():
    parser = argparse.ArgumentParser(description='Validate {stage_name} kernel')
    parser.add_argument('input', help='Path to kernel .so')
    parser.add_argument('--stage-spec', required=False, default=None, help='Path to StageSpec JSON file (overrides embedded)')
    parser.add_argument('--n-seq', type=int, default=16, help='Number of sequences (NT)')
    parser.add_argument('--l-seg', type=int, default=128, help='Segment length (BT)')
    parser.add_argument('--num-tests', type=int, default=10, help='Max test cases to run')
    args = parser.parse_args()
    
    # Load StageSpec: external file overrides embedded
    import json
    if args.stage_spec:
        with open(args.stage_spec, 'r') as f:
            stage_spec = json.load(f)
    else:
        stage_spec = STAGE_SPEC
    
    # Setup kernel
    call_kernel = setup_kernel(args.input, stage_spec)
    
    # Generate test cases
    cases = generate_default_cases(stage_spec)
    cases = cases[:args.num_tests]
    
    # Run validation
    passed = 0
    failed = 0
    for case in cases:
        if validate_case(call_kernel, case, stage_spec):
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
```

## DEFAULT_CASES Generation Algorithm

**Goal**: Generate at least 6 BT values with ≥2 seeds each, derived from StageSpec.

```python
def generate_default_cases(stage_spec):
    problem = stage_spec['stage']['problem']
    base_bt = problem['BT']  # BT must exist in StageSpec.problem
    
    # Generate 6 BT values around the baseline
    bt_values = [
        base_bt // 4,   # toy
        base_bt // 2,   # small
        base_bt,        # baseline
        base_bt * 2,    # large
        base_bt * 3,    # stress
        base_bt * 4,    # extreme
    ]
    bt_values = [max(4, bt) for bt in bt_values]
    
    # Map BT → NT (smaller BT needs larger NT for reasonable total tokens)
    bt_to_nt = {bt: max(1, 256 // bt) for bt in bt_values}
    
    cases = []
    for bt in bt_values:
        nt = bt_to_nt[bt]
        for seed in [42, 123]:  # two seeds per BT
            cases.append({'bt': bt, 'nt': nt, 'seed': seed, ...})
    
    return cases
```

**Never hard-code** `BT=4,16,32,64,96,128` — derive from StageSpec.problem['BT'].

## Two-Tier Accuracy Check

```python
def check_accuracy(actual, expected, description, rtol=1e-3, atol=1e-3):
    # Tier 1: strict assert_close
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        # Tier 2: RMSE/R² fallback
        rmse = torch.sqrt(((actual - expected) ** 2).mean()).item()
        mean_abs = expected.abs().mean().item()
        
        if mean_abs < 1e-9:
            # Near-zero: absolute RMSE check
            return rmse < 5e-4
        
        ratio = rmse / mean_abs
        ss_res = ((expected - actual) ** 2).sum().item()
        ss_tot = ((expected - expected.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return ratio <= 0.05 and r2 >= 0.99
```

## Reference Model Integration

### Option 1: Inline the reference model
```python
# Paste the entire ReferenceModel.py content here
def reference_model(input_tensor):
    # ... implementation from ReferenceModel ...
    return output_tensor
```

### Option 2: Import from file (if ReferenceModel is staged)
```python
import sys
sys.path.insert(0, '/path/to/staged/reference_model.py')
from reference_model import reference_model
```

**Prefer Option 1 (inline)** for standalone scripts.

## CLI Argument Template

```python
parser = argparse.ArgumentParser()
parser.add_argument('input', help='Path to kernel .so')
parser.add_argument('--stage-spec', required=True, help='Path to StageSpec JSON file')
parser.add_argument('--n-seq', type=int, default=16, help='Number of sequences (NT)')
parser.add_argument('--l-seg', type=int, default=128, help='Segment length (BT)')
parser.add_argument('--num-tests', type=int, default=10, help='Max test cases')
parser.add_argument('--profile-only', action='store_true', help='Skip accuracy checks')
args = parser.parse_args()
```

## Common Pitfalls

1. **Hard-coded dimensions**: Never use `HV=256, H=16, K=128, V=128` — derive from StageSpec.problem
2. **Toy BT values**: Never use `BT=4,16,32` — derive from StageSpec.problem['BT']
3. **CPU generator + NPU device**: Always `.to(device)` after `torch.randn(..., generator=g)`
4. **Double-wrapping stream_ptr**: Never `ctypes.c_void_p(stream_ptr)`
5. **Using `.cpu()` to compare**: This triggers `torch.npu.synchronize()` which hangs in msprof simulator. Always compute reference on NPU and compare on-device.
6. **Wrong argtypes order**: Must match KernelABI.arguments exactly
7. **Using `torch.npu.synchronize()`**: Hangs in simulation mode. Never call synchronize — the kernel returns synchronously and on-device comparison doesn't need it.
