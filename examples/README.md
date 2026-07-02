# Examples

Self-contained PyTorch programs to test the `npu-skillyard` pipeline, grouped by
difficulty. Each is a plain `torch.nn.Module` (no external deps) that the pipeline
decomposes into stages, generates PTO kernels for, validates on a real NPU, and
benchmarks.

The tiers step through the pipeline's capabilities: **tier1** is pure elementwise Vec
(one stage), **tier2** adds the Cube matrix engine + multi-stage fan-out + fusion,
**tier3** adds softmax (Vec reduction + fp-overflow stability), **tier4** is a full
multi-projection block.

## Prerequisites

- The plugin installed (`/plugin marketplace add huawei-csl/npu-skillyard` then
  `/plugin install npu-skillyard@npu-skillyard`), **or** run from a clone of this repo.
- A host with **CANN** (`/usr/local/Ascend/cann/set_env.sh`), a **`torch_npu`** Python,
  a real **Ascend NPU**, and **pto-isa** (auto-cloned by Preflight if absent).

These programs declare no shapes, so each comes with a **ready contract** below — hand it
to the pipeline so Phase 0's autonomy gate proceeds instead of stopping for confirmation.

## How to run

Open a fresh `claude` session from your host project (the one with `.venv` + CANN +
`third_party/pto-isa`), then paste a prompt like this (fill in `<...>`):

```
Use the stage-pipeline agent to run the full PTO pipeline on
<absolute path to the example .py>.
- output_dir: <a fresh dir, e.g. ~/tmp/test_<name>/out>
- pto_python: <abs path to your torch_npu python, e.g. /path/wfpy/.venv/bin/python>
- pto_isa_root: <abs path to pto-isa, e.g. /path/wfpy/third_party/pto-isa>
- platform: a2a3
- Use this contract verbatim (confidence high): <the contract line for this example>
- optimize: false. Produce the structured ref/ src/ reports/ output and the README with graphs.
```

Preflight resolves paths as **arg > env var > autodetect > default**, so if you run from a
host project laid out that way you can omit `pto_python` / `pto_isa_root`. Set `optimize: true`
(and drop `optimize: false`) once a stage passes and you want the optimization campaign.
The run always delivers ONE integrated kernel `kernel_chain_<algo>` (Phase 7 Part A — the
validated per-stage kernels stitched into a single stream-ordered `call_kernel`). The
tightly-coupled compute-fused "mix" (Part B) is **opt-in** — add `fuse: true` to attempt it,
and even then it ships only if it measurably beats the composed chain.

**Via the workflow instead:** `cd` into a clone of this repo and ask Claude to run the
`pto-pipeline-parallel` workflow with the same fields as JSON args (add `contract`). If the
per-stage worker agent doesn't resolve by bare name, pass
`worker_agent: "npu-skillyard:pto-stage-worker"`.

---

## tier1-elementwise -- pure Vec, one stage (the floor)

Elementwise maps: no reduction, no Cube, no cross-element dependency. Fastest to pass;
good for smoke-testing an environment. All fp16, tolerance rtol=0.02 atol=0.02.

| Example | Computes | Ready contract |
|---|---|---|
| `gelu_program.py` | `F.gelu(x)` | dtype float16; dims M=256, N=2048; `x:[M,N]`; sweep N over [2048,4096,8192] |
| `silu_program.py` | `F.silu(x)` | dtype float16; dims M=256, N=2048; `x:[M,N]`; sweep N over [2048,4096,8192] |
| `relu_add_program.py` | `F.relu(x+y)` | dtype float16; dims M=256, N=2048; `x:[M,N]`, `y:[M,N]`; sweep N over [2048,4096,8192] |
| `add_mul_program.py` | `(x+bias)*scale` | dtype float16; dims M=256, N=2048; `x/bias/scale:[M,N]`; sweep N over [2048,4096,8192] |
| `mlp_gated_program.py` | `silu(gate)*up` | dtype float16; dims M=256, N=2048; `gate/up:[M,N]`; sweep N over [2048,4096,8192] |
| `rotary_pair_program.py` | RoPE + add on q,k | dtype float16; dims R=256, D=128 (even); `q/k/cos/sin:[R,D]`; sweep R over [256,512,1024] |

`gelu` is the verified reference run (PASS, fp64 rel-err ~2.9e-4, 98.6% headroom).

## tier2-matmul-mlp -- Cube engine + multi-stage + fusion (the "medium" step)

Introduces the **Cube matrix engine** (dense GEMMs: `TMATMUL`, L1/L0 staging, `TEXTRACT`),
real multi-stage parallel fan-out, and a genuine compute-fusion target -- none of which
tier1 touches.

| Example | Computes | Ready contract |
|---|---|---|
| `mlp_full_program.py` | `down(silu(x@gate_w^T) * (x@up_w^T))` | dtype float16; dims M=256 (seq), K=512 (hidden), I=1024 (intermediate); `x:[M,K]`, `gate_w/up_w:[I,K]`, `down_w:[K,I]`; sweep M over [256,512,1024] |

## tier3-attention -- Cube + softmax reduction (medium-high)

Adds a **softmax** (row-max, exp, row-sum, divide) on top of the Cube GEMMs -- exercising
the Vec reduction rules and fp-overflow stability. The flash-attention archetype; the
bridge toward the full KDA workload.

| Example | Computes | Ready contract |
|---|---|---|
| `sdpa_program.py` | `softmax(qk^T/sqrt(d))@v` | dtype float16; dims B=1, H=8, S=512, D=128; `q/k/v:[B,H,S,D]`; sweep S over [512,1024,2048] |
| `sdpa_causal_program.py` | causal SDPA (with mask) | dtype float16; dims B=1, H=8, S=512, D=128; `q/k/v:[B,H,S,D]`; sweep S over [512,1024,2048] |

## tier4-blocks -- full self-attention block (high)

Q/K/V/O projections + head split/merge + SDPA -- many stages and layout work. A realistic
transformer sub-block. Shapes are evidenced in the source (`num_heads=4`, `head_dim=32`).

| Example | Computes | Ready contract |
|---|---|---|
| `attention_block_program.py` | proj -> heads -> SDPA -> merge -> o_proj | dtype float16; dims B=1, S=256 (seq), hidden=128, heads=4, head_dim=32; `x:[B,S,128]`, `q_w/k_w/v_w/o_w:[128,128]`; sweep S over [256,512,1024] |

---

## Tips

- Start at tier1 to confirm the environment, then move up. A tier1 pass proves
  preflight + decompose + kernel-gen + compile + real-NPU validate + benchmark + report.
- The output is a structured run dir (`ref/` inputs, `src/` generated kernels, `reports/`
  graphs + `report.md`) plus a top-level `README.md` narrating what was achieved and any
  blockers -- see the main [README](../README.md#run-output).
- fp16 tolerance is ~2e-2, not ~1e-5 -- a pass tighter than that is suspicious, and an
  fp16 "FAIL" at 1e-5 is not a real bug.
