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
   - `lowering_hint` (free-text: dominant parallel axis, reduction axis, tile shape constraints)
   - `reference_source` (self-contained pure-torch function for this stage — see Reference Implementation Rules)
   - `evidence_gaps` (list of uncertainties; empty list if all fields are confirmed)
4. **Write per-stage reference** — a standalone pure-torch function that computes just this stage's math, no control flow from other stages
5. **Use npu-coding MCP** — for each stage, verify which PTO instruction families apply
6. **Produce `stage_plan.json`**

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

## Evidence Gaps

If you cannot determine a stage boundary, shape, dtype, or instruction family with confidence, record the uncertainty in the stage entry as `evidence_gaps` rather than guessing.

If multiple stage decompositions are plausible, prefer the one with the lower boundary cost unless the more split version clearly improves semantic reuse, independent validation, or lowering separation.

Never invent shapes, dtypes, or instruction names that are not justified by the source code or MCP evidence.
