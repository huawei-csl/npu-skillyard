---
name: torch-algorithm-to-pto-stages
description: "Decompose a PyTorch algorithm source file into named tile-computation stages, produce a stage plan with per-stage tensor interfaces, problem dimensions, lowering hints, and pure-torch reference implementations. Use when the input is a single torch.nn.Module or a functional PyTorch algorithm and the goal is to identify computational stages that each map to a single PTO tile kernel. Triggers: algorithm decomposition, stage extraction, stage plan, KDA stages, GDN stages, chunked attention stage breakdown, algorithm to PTO."
---

# Torch Algorithm to PTO Stages

Read a PyTorch algorithm source file and decompose it into named tile-computation stages.

## Inputs

You receive one required file input:

- `AlgorithmSource` (`.py`): a PyTorch module or function implementing the algorithm

The source may use:
- plain PyTorch ops (`torch.matmul`, `torch.triu`, `torch.exp`, `torch.cumsum`, etc.)
- control flow (Python `for` loops, `if` statements)
- helper functions
- `torch.nn.Module` subclasses

## Goal

Produce a `stage_plan.json` artifact that:

1. Identifies the **algorithmic stages** — coherent tile-computation units
2. For each stage, extracts the **tensor interface** (inputs, outputs, shapes, dtypes)
3. Records **problem dimensions** and shape constants
4. Generates a **pure-torch reference implementation** for each stage
5. Suggests **PTO instruction families** for lowering

The stage plan feeds downstream kernel generation. This skill does not emit kernel C++ source.

## Shape & Precision Contract (establish this FIRST)

Before decomposing, establish a single model-level **Shape & Precision Contract**
that becomes the source of truth for every per-stage shape, the validation
tolerance, and the benchmark sweep. Per-stage interface shapes are DERIVED from
this contract as expressions over its symbolic dimensions -- never invented
independently per stage. This is what prevents a pipeline from silently running on
a toy shape (e.g. a single-chunk, low-head, fp32 config) that no one chose.

The contract is algorithm-agnostic. Do NOT assume KDA / attention-specific names.
The algorithm declares its OWN symbolic dimension names based on what the source
actually uses (e.g. `seq`, `heads`, `head_dim`, `hidden`, `chunk`, `M`, `N`, `K`,
`vocab`).

### Source-tiered research (assign a tier + source to every dim and the dtype)

Find a value for each dimension and the dtype, and record WHERE it came from and
HOW confident you are:

- **Tier 1 (high)** -- explicit in the provided material: argument defaults,
  docstrings, a `__main__` / example block, shape literals in adjacent test or
  benchmark files, or a config the source reads. Algorithm-agnostic.
- **Tier 2 (medium)** -- the algorithm is recognizably a member of a known family
  and the value follows that family's established convention (only when the family
  is identifiable from the source).
- **Tier 3 (low)** -- no evidence in the source; a generic heuristic default. Flag
  loudly: a Tier-3 dim is the signal that a human should confirm the contract before
  an expensive downstream run.

Reading SHAPES, DTYPES, or dim values from an external reference / benchmark / config
file is allowed -- dimensions are not kernel logic. (Reading another kernel's
*implementation* source is a separate provenance concern enforced by the caller.)

### Locked vs free dimensions

Mark a dim `locked: true` when fixed by an architectural / lowering constraint (a
tile layout that only compiles at one size, a hardware-fixed lane width) rather than
freely chosen; mark `locked: false` for a user-tunable workload knob (sequence
length, batch, head count). Record `locked_reason` when known. Downstream generation
may DISCOVER a new constraint and amend the contract -- that feedback must be
preserved, never used to silently override a user-supplied value.

### A CORRECTNESS SHAPE IS NOT A BENCHMARK SHAPE -- the sweep must be able to discriminate

Tier 1 is the gold standard for CORRECTNESS, and it is a **trap for PERFORMANCE**. The
richest Tier-1 source is usually a unit test, and a unit test's shapes are chosen to
exercise the code path cheaply. Reusing them as the benchmark sweep asks the hardware to
distinguish two implementations on a problem too small to distinguish them.

**This has already cost a headline.** `grouped_matmul` took `M_total=1792` from
`test_npu_grouped_matmul` -- every dim Tier 1, `confidence: high`, gate correctly passed.
At that shape the production config runs **12.41 us** against the vendor's **12.40 us**,
and **2.72 us of that (22%) is empty-launch floor neither kernel controls**. Both arms
were HOST-BOUND (ours 21.6-37.3 us enqueue against 20-21 us of device work; the vendor
63.8-63.9 against 61.5-63.0), so the event harness read **3.0x** where the device profiler
read **1.001x** -- the entire apparent win was the gap between a `ctypes` launch and a
framework op. An earlier run of the same algorithm at `M=16384` -- 9.1x the work for only
3.9x the time, the fixed cost showing through -- was device-bound and discriminating.

**So price the largest sweep point BEFORE committing the contract:**

1. **Estimate device time per call at the largest sweep point.** FLOPs or bytes against the
   part's ceiling is enough; you are checking an order of magnitude, not calibrating.
2. **Compare it against the FIXED COST -- which is launch PLUS ramp, and is bigger than
   you think.** The empty-launch probe (~2.7 us single-engine, ~4.8 us MIX on A2/A3) is a
   LOWER BOUND, not the number. Measured on `grouped_matmul`, the true fixed cost was
   **9.63 us -- 3.5x the noop probe** -- the rest being weight load, pipeline fill and
   group dispatch. Pricing only the launch floor calls a shape adequate three sizes before
   it is. **Read the fixed cost as the INTERCEPT of device time against the sweep dim**,
   fitted inside one regime (see below); that captures ramp, a noop probe cannot.
3. **The largest sweep point must have fixed cost under ~10% of the FASTER arm's device
   time.** Below that, the ratio is a property of the two overheads. `grouped_matmul` at
   its Tier-1 shape was **80% fixed cost**, and because the two arms' fixed costs were
   within 8% of each other, it read as parity while the kernels differed by 1.82x.
4. **These are separate from INSTRUMENT admissibility.** An arm is HOST-BOUND when its
   enqueue exceeds its device time (a `ctypes` launch enqueues in 10-12 us; a `torch_npu`
   framework op in 50-70 us). That bars EVENT and WALL-CLOCK timing for that arm -- it does
   NOT make the shape undiscriminating, because the device profiler reads on-device
   duration and is unaffected. Do not merge the two tests: one picks the instrument, the
   other picks the shape.
5. **Fit inside ONE regime.** A sweep wide enough to discriminate is usually wide enough to
   cross a cache-capacity break, and a fit spanning the break returns nonsense -- on this
   case a 0.41 us intercept for one arm and a NEGATIVE one for the other. Watch the
   marginal cost per unit between adjacent points: a simultaneous jump in BOTH arms is a
   regime change, not a trend. Fit below it, and report the spilled point separately.

**When the Tier-1 shape is non-discriminating, that is a REPORTABLE CONTRACT DEFECT, and it
does NOT license substituting a bigger number.** The standing rule holds: a discovered
constraint amends the contract, never silently replaces a user-supplied or source-evidenced
dim. Do this instead:

* Keep the Tier-1 shape as the **validation** sweep -- it is entirely valid there.
* **Propose** an added benchmark point large enough to discriminate, with its tier (a size
  not in the source is Tier 2 at best) and the arithmetic that justifies it.
* Set `confidence` to `needs-confirmation` on the basis of the benchmark point alone, and
  **STOP at the autonomy gate.** A correctness contract that is Tier 1 throughout can still
  be `needs-confirmation` for benchmarking; say which of the two is unconfirmed.
* If the run proceeds anyway on the small shape, the report must say **"this shape cannot
  discriminate"** and must NOT present the resulting parity as a finding. Parity measured
  below the discrimination threshold is an absence of evidence, not evidence of absence.

Record the check in the contract as `bench_discrimination` so downstream phases can see the
verdict rather than re-deriving it.

### Contract shape

Emit the contract as a top-level `shape_contract` block in the stage plan:

```json
"shape_contract": {
  "dtype": {"value": "float16", "tier": 1, "source": "tests/bench_x.py build_inputs"},
  "batch": {"value": 1, "tier": 1, "source": "docstring"},
  "dims": {
    "<dim_name>": {"value": 128, "tier": 1, "role": "head_dim",
                   "locked": true, "locked_reason": "tile layout baked for 128",
                   "source": "reference benchmark args"}
  },
  "sweep_axis": {"dim": "<dim_name>", "values": [4096, 8192, 32768]},
  "bench_discrimination": {
    "largest_point": {"<dim_name>": 32768},
    "est_device_us_per_call": 118.0,
    "fixed_cost_us": 9.63,
    "fixed_cost_method": "intercept of device time vs sweep dim, fitted below the cache break",
    "launch_floor_us": 2.72,
    "launch_floor_note": "noop-launch probe: a LOWER BOUND on fixed cost, not the number",
    "slowest_arm_enqueue_us": 64.0,
    "verdict": "discriminating | NON-DISCRIMINATING",
    "note": "<if non-discriminating: the proposed larger point, its tier, and the arithmetic>"
  },
  "tolerance": {"rtol": 0.02, "atol": 0.02, "derived_from": "dtype=float16"},
  "confidence": "high | needs-confirmation",
  "notes": "<anything the caller should see before committing a long run>"
}
```

Set `confidence` to `high` ONLY if EVERY dim and the dtype is Tier 1 (directly
evidenced in the source); if ANY value is Tier 2 or Tier 3, set `needs-confirmation`
(a family-convention guess is still unconfirmed for this algorithm). Derive
`tolerance` from the dtype (fp32 ~1e-5; fp16/bf16 ~2e-2
relative -- accumulation will not match fp32). Per-stage `inputs`/`outputs` shapes
and `problem` values MUST be consistent with the contract and use its symbolic dim
names (or values derived from them), so every stage shape traces back to one
contract entry.

## What a Stage Is

A stage is a **named tile-level computation** — not a single `mul` or `slice` op.

Stages are separated by **dataflow boundaries**: when one block of computation produces a tensor that is consumed by a qualitatively different block, that's a stage boundary.

Examples of stage boundaries:
- A gate cumsum that produces a prefix-sum tensor consumed by a matrix-product stage
- A KKT matrix build that produces a triangular matrix consumed by a solver
- A correction-term computation (u, w) that feeds a sequential state recurrence
- A sequential state pass that produces snapshots consumed by an output pass

Examples of what is NOT a separate stage:
- A single `torch.add` or `torch.mul` — these are ops within a stage
- A reshape/permute immediately before a compute — this is a layout preparation, part of the enclosing stage
- A dtype cast (.float(), .to()) — these are implementation detail

## Stage Count Heuristic

The number of stages is not fixed in advance. Infer it from the source.

Use a generic split-benefit vs boundary-cost heuristic:

- Increase split benefit when a boundary introduces one or more of these:
  - a loop-carried or stateful dependency
  - a reusable semantic intermediate consumed by a qualitatively different block
  - a major shape/domain change, such as switching from sequence tiles to state matrices or from per-token work to chunk-local matrices
  - a distinct dominant lowering family, such as layout/prefix work vs dense contraction vs recurrent scan
  - an independently testable mathematical subproblem whose reference can stand alone cleanly
- Increase boundary cost when a proposed split mostly adds one or more of these:
  - a layout-only peel-off with no independent semantic value
  - a tiny single-consumer intermediate that only forwards data into the next compute block
  - extra materialization, memory traffic, or ABI surface without simplifying generation, validation, or later stitching
  - a split whose reference implementation is not meaningfully simpler than keeping the blocks together

Split only when the estimated benefit clearly exceeds the interface/materialization cost.

### OCCUPANCY IS A SPLIT CRITERION -- compute each stage's parallel width BEFORE choosing

The list above weighs semantics and bytes and says nothing about whether a stage can fill
the machine. That omission has cost a measured **2.7x**, twice, on the same algorithm: a
plan folded three blocks into one stage whose only parallel axis was a **contract dim as
small as 4**, on a part with 48 cores. The seam arithmetic was immaculate -- intermediate
1024 B against 77.9 MB of input, `O(B)` against `O(B*V)`, about as cheap as a boundary
gets -- and the plan was still wrong, because a nearly-free seam says nothing about
whether either side can use the device.

**For every candidate stage, write down its parallel width: the number of independent
work items the stage can issue, as an expression over contract dims.** Then:

* **A stage whose parallel width is bounded by a SMALL contract dim cannot fill the
  device.** Compare it against the part's core count (A2/A3: 48 AIV / 24 AIC). A width of
  `B` where `B` can be 4 leaves ~92% of the machine idle no matter how good the kernel is,
  and no amount of Phase 6.5 will recover it -- the optimizer searches schedules, not
  decompositions.
* **Folding work INTO a narrow stage is a strong boundary cost**, and it is the failure
  mode this rule exists to catch. Merging a wide block into a narrow one does not save a
  seam; it drags the wide work down to the narrow width.
* **Splitting to WIDEN is a first-class split benefit**, even when the seam is not free.
  If one candidate boundary yields stages of width `O(B)` and `O(B*V)` while another
  yields two stages both `O(B*V)`, the second is better on occupancy grounds even at a
  larger intermediate. Price both.

**Record `parallel_width` per stage in the plan, and the imbalance ratio per boundary.**
Where two adjacent stages differ by more than ~4x in achievable width at the SMALLEST
contract shape, say so explicitly and justify keeping them together. A measured 20.7x
imbalance at the small end has been observed to survive the whole pipeline and land as a
2.7x end-to-end loss.

**If you defer a split, pre-register the flip condition with a number.** State the
measurement that would overturn the choice ("adopt the column split if stage A at the
smallest shape exceeds stage B by more than 2x"), and CHECK IT in Phase 6. This works --
the condition above fired at 20.7x and correctly identified the decomposition as the
binding defect. A deferred split with a pre-registered trigger is a decision; a deferred
split without one is a guess that never gets revisited.

Practical rule:

- prefer fewer, semantically complete stages when adjacent blocks share the same dominant lowering family and the intermediate has no independent reuse value
- prefer more stages when combining the blocks would mix incompatible concerns such as prep/layout, triangular or dense contraction, and recurrent state update into one stage

## Workflow

1. **Read the source** — understand the full algorithm, not just one function
2. **Establish the Shape & Precision Contract** (see the section above) — run the
   source-tiered research, assign a tier + source to the dtype and every dimension,
   mark locked vs free, and set `confidence`. If a contract was supplied to you, use
   it verbatim and skip the research.
3. **Identify stage boundaries** — look for dataflow breaks, different problem shapes, qualitatively different computation patterns, and apply the split-benefit vs boundary-cost heuristic above
3. **For each stage**, extract every field in the schema below:
   - `name` (e.g., `attention_score`, `softmax_norm`, `state_update`, `output_projection`)
   - `description` (one sentence of what the stage computes)
   - `inputs` (list of tensors: name, shape, dtype, role)
   - `outputs` (list of tensors: name, shape, dtype, role)
   - `problem` (dimension constants: tile_size, feature_dim, sequence_dim, batch_dim, etc.)
   - `code_region` (source line range, e.g., `"lines 62-73"`)
   - `instruction_families` (PTO instruction names verified via npu-coding MCP)
   - `parallel_width` (expression over contract dims for the number of independent work
    items this stage can issue, plus its value at the SMALLEST contract shape -- see
    the occupancy criterion above)
  - `lowering_hint` (free-text: dominant parallel axis, reduction axis, tile shape constraints)
   - `reference_source` (self-contained pure-torch function for this stage — see Reference Implementation Rules)
   - `evidence_gaps` (list of uncertainties; empty list if all fields are confirmed)
4. **Write per-stage reference** — a standalone pure-torch function that computes just this stage's math, no control flow from other stages
5. **Use npu-coding MCP** — for each stage, verify which PTO instruction families apply
6. **Produce `stage_plan.json`**

## TENSOR LAYOUT IS PART OF THE CONTRACT -- and it determines DROP-IN status

A contract that fixes shape, dtype and tolerance still does **not** say what the caller must
hand you. Two kernels with identical shapes and dtypes can require **different memory layouts**,
and a kernel whose required layout differs from the reference's is **not a drop-in replacement**
however identical its signature looks.

This is not hypothetical. A generated `grouped_matmul_swiglu_quant` kernel reached its best
result by consuming a **repacked** weight `[E, N/nt, K, nt]`; the vendor operator requires
`FRACTAL_NZ` and **errors outright** on a plain `ND` weight. Same shapes, same dtypes, three
mutually incompatible layouts.

**Record a `tensor_layouts` block in the contract, one entry per input and output:**

```json
"tensor_layouts": {
  "weight": {
    "reference_layout": "ND [E,K,N]",
    "kernel_layout":    "PACKED [E,N/nt,K,nt]",
    "drop_in":          false,
    "transform":        "host-side repack, 1.641 ms, one-time",
    "amortizable":      true,
    "amortizes_over":   "read-only, persists across invocations (model weight)",
    "coupled_to":       ["nt=256"],
    "tier":             1
  }
}
```

**The fields that matter, and why:**

* **`reference_layout`** -- what the source algorithm / vendor operator expects. Establish it in
  Phase 0 like any other Tier-1 fact; if the vendor requires a cast, **probe whether it is
  mandatory or merely optimal** (that one errors on `ND` was found by probe, not by reading docs).
* **`kernel_layout`** -- what the kernel you shipped actually consumes. Unknown until the
  optimizer's layout axis resolves, so Phase 0 records the reference and Phase 7 fills this in.
* **`drop_in`** -- literally `kernel_layout == reference_layout`. **A `false` here is a
  first-class deliverable fact and must reach the report and the README**, not be discovered by
  a user whose weights silently produce garbage.
* **`transform` / `amortizable` / `amortizes_over`** -- who pays, how much, and over what. A
  transform on a read-only operand that persists across calls amortizes to nothing; the same
  transform on a per-call activation does not, and usually kills the idea.
* **`coupled_to`** -- **the field people forget.** If the layout is a function of a tuning
  parameter, retuning that parameter **invalidates every stored tensor** the caller prepared.
  A layout coupled to a tile width is materially worse to deploy than a stable framework format
  reachable through a public API, even at identical transform cost. Say so.

**Rule: if `drop_in` is false, the benchmark must ALSO report the drop-in number** -- the same
kernel consuming the reference layout -- so a reader who cannot repack sees what they would get.
Reporting only the repacked number silently changes the interface and calls it a speedup.

## SEAM ANALYSIS -- do this for every boundary, and record it

A stage boundary is not free. Every seam pays a **GM round trip** of the intermediate, because
on A2/A3 the only legal tile-to-tile moves are `Mat -> Left|Right|Bias|Scaling`, `Vec -> Vec`
and `Acc -> Mat` (`a2a3/TMov.hpp:200-204`). There is **no `Acc -> Vec`**, so a Cube result
reaching Vec *must* transit GM.

That cost is usually negligible and occasionally fatal. **Which one it is, is computable before
any kernel exists**, so compute it:

For each seam, record in the stage plan:

```json
"seams": [{
  "from": "qk_scores", "to": "softmax_rows",
  "engines": "cube->vec",
  "intermediate_bytes": 67108864,
  "input_bytes": 8388608,
  "amplification": 8.0,
  "l2_bytes": 201326592,
  "l2_resident": true,
  "growth": "O(S^2) vs inputs O(S*D)"
}]
```

**Read it like this:**

| condition | meaning | action |
|---|---|---|
| `intermediate_bytes` <= `input_bytes` | the seam is nearly free | **compose** -- this is the normal case |
| intermediate grows at the **same order** as inputs | fixed overhead, bounded | compose; fusion is a constant-factor prize |
| intermediate grows at a **HIGHER order** than inputs | **amplification is unbounded in the sweep dim** | **fusion is mandatory**, not an optimization |
| `intermediate_bytes` > L2 (192 MB on A2) | the round trip streams to HBM | re-tile so the live set fits, or fuse |

**The asymptotic row is the one that matters.** Measured on `flash_attention_grad`, where the
intermediate is `[B,N,S,S]` and the inputs are `[B,N,S,D]`:

| S | ours MB | vendor MB | amplification |
|---|---|---|---|
| 128 | 28.3 | 8.2 | 3.43x |
| 512 | 305.1 | 33.0 | 9.25x |
| 1024 | 1122.2 | 66.0 | **17.00x** |

The amplification **doubles every time S doubles**, because ours is O(S^2) and the fused
alternative is O(S*D). No per-stage optimization can close that -- both attention cases spent
their full 15-attempt budgets and reached 99.5% of their measured bandwidth ceilings while
still losing, because they were solving the wrong problem.

**So: run the seam analysis in Phase 1 and let it gate the plan.** If a seam shows
higher-order growth, say so in the plan and flag the case as *fusion-required* before Phase 3
generates a single artifact. That converts 15 wasted attempts into an up-front decision.

### What the seam analysis DOES and DOES NOT predict -- checked against 13 cases

Retro-tested against every multi-stage case in the campaign:

| seam growth | cases | outcome |
|---|---|---|
| **HIGHER order** than inputs | `attention_sdpa`, `flash_attention_grad` | **both LOSE and DEGRADE with size** (1.16->2.28x, 1.09->3.35x) |
| **SMALLER** than inputs | `cross_entropy_loss`, `rms_norm_backward`, `moe_token_permute`, `top_k_top_p`, `group_norm_silu`, `hans_compress` | **all compose cleanly** -- 5 wins (1.19x-5.69x), 1 blocked on unrelated ISA grounds |
| **SAME order** | `ffn`, `grouped_matmul`, `grouped_matmul_swiglu_quant`, `kv_rmsnorm_rope_cache` | bounded: parity to 1.51x slower, **not degrading** |

**A cheap seam is NOT a verdict that the plan is good.** This table ranks boundaries by
seam growth and says nothing about occupancy. A plan in the "SMALLER than inputs" row --
the safest row here -- was measured **2.7x slower end to end** than a finer decomposition
of the same algorithm, because the cheap seam sat next to a stage whose parallel width was
bounded by a contract dim of 4. Read this table together with the occupancy criterion
above, never instead of it.

**It predicts "can per-stage tuning close this gap?" -- not "will we win?".** Only the
higher-order row is a structural verdict. The same-order losses are ordinary Cube-efficiency
gaps against hand-tuned vendor matmuls, and they may well be closable; the rule says nothing
about them either way. Do not use it to excuse a loss.

**The signature to trust is DEGRADATION WITH THE SWEEP DIMENSION**, not the loss itself. A
constant-factor loss is an optimization problem. A loss whose ratio grows with the sweep is a
decomposition problem, and no attempt budget will fix it.

**A seam being expensive does not mean the decomposition is wrong** -- it means the composed
form is a correctness scaffold, not the deliverable. Validate the stages independently (that is
the whole value of decomposing), then fuse, then check the fused kernel against the composed one
bit-for-bit.

## Stage Plan Schema

```json
{
  "schema_version": "stage_plan_v1",
  "algorithm": "<name derived from the code>",
  "source": "<original source filename>",
  "shape_contract": { "...": "see Shape & Precision Contract section above" },
  "stages": [
    {
      "name": "<stage name>",
      "description": "<what this stage computes>",
      "stage_index": 0,
      "inputs": [
        {
          "name": "<tensor name>",
          "shape": [1, 256, 8, 128],
          "dtype": "float32",
          "role": "input"
        }
      ],
      "outputs": [
        {
          "name": "<tensor name>",
          "shape": [1, 256, 8, 128],
          "dtype": "float32",
          "role": "output"
        }
      ],
      "problem": {
        "tile_size": 64,
        "feature_dim": 128,
        "head_dim": 8
      },
      "code_region": "<source line range implementing this stage, e.g. lines 62-73>",
      "instruction_families": ["TLOAD", "TADD", "TSTORE"],
      "lowering_hint": "<free-text: dominant parallel axis, reduction axis, tile shape constraints — the agent prompt may provide domain-specific guidance>",
      "reference_source": "<per-stage reference python code>",
      "evidence_gaps": ["<reason for uncertainty if any field is unconfirmed>"]
    }
  ]
}
```

## Output Contract

Return only the complete stage plan as raw JSON:

- no JSON envelope (`{"outputs": {"StagePlan": "..."}}`)
- no markdown fences
- no commentary before or after the JSON
- the response is the file body — the workflow framework handles file routing

## npu-coding MCP Integration

Use the npu-coding MCP server to verify instruction families. The server name, transport,
and URL are configured by the agent runtime — do not hardcode them.

Available tools:
- `get_cpp_intrinsic` — C++ intrinsic signature
- `get_constraints` — per-backend dtype and shape constraints
- `get_instruction` — full instruction detail
- `list_categories` — instruction categories
- `search_instructions` — text search across instructions
- `get_assembly_format` — assembly syntax
- `get_examples` — code examples

For each stage you identify:

1. Call `get_cpp_intrinsic` to verify each instruction in the family exists
2. Call `get_constraints` to check dtype and shape constraints
3. Record verified instructions in `instruction_families` and guidance in `lowering_hint`

Do not claim an instruction family is usable unless the MCP confirms:
- The instruction exists
- It supports the required dtypes
- Shape constraints are met

If an instruction is not found, record it as an evidence gap. Do not fabricate instruction names.

## Reference Implementation Rules

For each stage, write a pure-torch reference function that:

1. Takes exactly the input tensors listed in the stage spec
2. Returns exactly the output tensors listed in the stage spec
3. Uses only standard PyTorch ops (no custom C++ extensions)
4. Is deterministic (same inputs → same outputs every time)
5. Does not import from the original algorithm source — it must be self-contained
6. Includes a `def reference_model(...)` entrypoint named after the stage

The reference must match the mathematical intent of the stage, not the implementation details of the original code. If the original code uses a specific loop structure or dtype cast, the reference should preserve the math but may simplify the implementation.

Write the completed function into the stage's `reference_source` field.

## Self-Check Before Return

Before returning the stage plan, verify:

- [ ] Top-level keys present: `schema_version`, `algorithm`, `source`, `shape_contract`, `stages`
- [ ] `shape_contract` has `dtype`, `dims` (each with `value`, `tier`, `source`, `locked`), `tolerance`, and `confidence`; `confidence` is `high` iff EVERY dim and the dtype is Tier 1, else `needs-confirmation`
- [ ] `shape_contract` has `bench_discrimination` with the largest sweep point priced against the launch floor AND the slowest arm's enqueue cost; a `NON-DISCRIMINATING` verdict sets `confidence: needs-confirmation` and STOPS at the autonomy gate with a PROPOSED larger point (never a silently substituted one)
- [ ] Each stage has: `name`, `stage_index`, `inputs`, `outputs`, `problem`, `instruction_families`, `reference_source`, `evidence_gaps`
- [ ] All shapes are lists of integers or contract symbolic dimension names, not empty
- [ ] All dtypes match the contract dtype and are valid torch dtypes (float32, float16, bfloat16, int32, etc.)
- [ ] All `problem` values are numeric literals or contract symbolic dim names — and every value traces back to a `shape_contract` entry
- [ ] `instruction_families` entries verified against npu-coding MCP where possible
- [ ] `reference_source` is non-empty Python code for each stage
- [ ] Any uncertain fields are documented in `evidence_gaps`, not guessed
- [ ] Output is raw JSON — no markdown fences, no envelope, no commentary

## Non-goals

- Do not generate `kernel_source` C++ in this stage plan.
- Do not include scalar loop bodies as pseudo-kernel placeholders.
- Keep output strictly to algorithm stage decomposition metadata.

## AN EXTERNAL BENCHMARK'S PUBLISHED BASELINE IS NOT A MEASUREMENT OF YOUR MACHINE

When the contract's comparison target comes from a third-party benchmark (cann-bench or
similar), the published per-case baseline is a number collected on **someone else's device
under their protocol**. Before it enters the contract as a target:

1. **Find what the baseline actually executes.** It is usually a named reference function,
   not a mystery. (cann-bench: `scripts/baseline/refs/levelN.py` -- for sigmoid it is
   literally `torch.sigmoid(inputs[0])`, dispatching to `aclnnSigmoid_SigmoidAiCore_Sigmoid`.)
2. **Run it on your hardware and reproduce the published number.** Record the ratio per case.
3. **Replicate their measurement protocol** or the reproduction will fail -- see the optimizer
   skill 3.14. Measured: unmatched protocol gave published/measured spanning 0.43x-1.96x;
   matched, it was 1.011 median.
4. **Use their comparator, not your own.** Reimplementing an accuracy standard from its prose
   is how you ship a gate that is stricter than the official one in one domain and absent in
   another. Import theirs (cann-bench: `kernel_eval.utils.compare.compare_tensors`, which
   loads standalone with only a PYTHONPATH) and call it with the arguments their evaluator
   uses -- including the same-precision `native_output` reference, whose absence makes the
   small-value branch *harsher*, not softer.

Record in the contract, per case: the published baseline, **your measured value for the same
reference**, and the protocol. A contract that cites only the published number cannot tell a
real regression from a protocol mismatch.

## YOUR OWN HARNESS HAS A FLOOR -- FIND IT BEFORE READING FIXED COST

Device-event timing of a back-to-back launch loop reports `max(device_time, host_enqueue)`.
Measure the floor directly with an empty kernel: on this host it was **~4 us/launch**, with
`npu.Event` device time and host enqueue time agreeing to 3% -- i.e. the event timer was
reporting the enqueue rate, not the kernel.

Consequences for Phase 0's `bench_discrimination` block:

* a case whose device time is below that floor measures the harness, and its ratio is not a
  kernel result;
* a fitted fixed cost extrapolates to the floor, so quote **how much of it is launch** --
  here at most ~3.9 us of a fitted 6.3 us, leaving ~2.4 us of genuine kernel prologue;
* never compare your fitted fixed cost against an *inferred* vendor fixed cost. Measure both
  on the same instrument or say you have not.

Where the scoring harness uses the profiler (kernel-only device time), measure there too, so
your number and the scorer's measure the same thing. On the op above the two instruments
agreed within a few percent once every case sat above the floor -- which is the result you
want to be able to state, rather than assume.

## Evidence Gaps

If you cannot determine a stage boundary, shape, dtype, or instruction family with confidence, record the uncertainty in the stage entry as `evidence_gaps` rather than guessing.

If multiple stage decompositions are plausible, prefer the one with the lower boundary cost unless the more split version clearly improves semantic reuse, independent validation, or lowering separation.

Never invent shapes, dtypes, or instruction names that are not justified by the source code or MCP evidence.
