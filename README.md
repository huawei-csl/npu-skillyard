# npu-skillyard

A Claude Code **plugin** for turning **PyTorch algorithms into validated, benchmarked
Ascend PTO-ISA kernels**. It bundles the skills, agents, and an MCP server that drive a
full pipeline: decompose -> generate artifacts -> generate kernels -> validate on real NPU
-> benchmark -> optimize -> fuse.

This repository is **both a plugin and a single-entry marketplace**, so it can be installed
directly.

## Install

```text
/plugin marketplace add huawei-csl/npu-skillyard
/plugin install npu-skillyard@npu-skillyard
```

Or from a full URL:

```text
/plugin marketplace add https://github.com/huawei-csl/npu-skillyard.git
```

After install, the skills are namespaced under the plugin, e.g.
`/npu-skillyard:torch-algorithm-to-pto-stages`.

## What's included

### Skills (`skills/`)
- **torch-algorithm-to-pto-stages** -- decompose a PyTorch module/function into named
  tile-computation stages + a Shape & Precision Contract (`stage_plan.json`).
- **pto-stage-artifact-generator-local** -- generate per-stage validation + benchmark scripts.
- **pto-stage-kernel-generator-v2** -- generate one PTO C++ kernel per stage (defines the
  C-series critical rules, the archetype decision tree, and the C24 compile recipe).
- **pto-kernel-optimizer** -- drive a correct kernel toward a performance target
  (measure -> decide -> attack -> re-measure).

### Agents (`agents/`)
- **stage-pipeline** -- full-pipeline orchestrator (Phases 0-7) with parallel per-stage
  fan-out. This is the **installable orchestration agent**.
- **pto-stage-worker** -- single-stage worker (artifacts + kernel + compile/validate/repair),
  designed to be fanned out one-per-stage.

### MCP server (`.mcp.json`)
- **npu-coding-mcp** -- serves PTO-ISA / AscendC / CCE / Runtime documentation used to
  verify every instruction family. Auto-registers on install and is **fetched and run
  directly from GitHub** ([huawei-csl/npu-coding-mcp](https://github.com/huawei-csl/npu-coding-mcp))
  via `uvx` -- no manual clone or `pip install`:

  ```json
  "command": "uvx",
  "args": ["--from", "git+https://github.com/huawei-csl/npu-coding-mcp.git@main",
           "npu-coding-mcp", "serve", "--stdio"]
  ```

  > **Prerequisite:** [`uv`](https://docs.astral.sh/uv/) must be installed on the user's
  > machine (`curl -LsSf https://astral.sh/uv/install.sh | sh`). `uvx` then handles the
  > rest: it clones the repo, installs deps, builds the FTS5 doc indexes on first `serve`
  > (no API key required), and caches everything -- the first launch is slow (~tens of
  > seconds), subsequent launches are fast.
  >
  > **Pinning / updates:** the `@main` ref tracks the default branch but `uvx` caches by
  > URL, so it will not auto-update. Pin to a tag/commit for reproducibility
  > (`...npu-coding-mcp.git@<rev>`); refresh a cached install with `uv cache clean`.
  >
  > **No-`uv` fallback:** clone + `pip install -e .` into a venv and point `.mcp.json` at
  > that interpreter instead:
  > `"command": "/path/to/.venv/bin/python", "args": ["-m", "npu_coding_mcp", "serve", "--stdio"]`.

### Workflow (`.claude/workflows/`)
- **pto-pipeline-parallel.js** -- the parallel variant of the pipeline (decompose once, fan
  out per-stage workers, then benchmark serially, optimize, fuse).

  > Workflows are **not** an installable plugin component. Two ways to use it (this repo
  > ships both):
  > 1. **Clone this repo** and work inside it -- the workflow loads automatically.
  > 2. **Plugin-only users:** copy `.claude/workflows/pto-pipeline-parallel.js` into your
  >    own project's `.claude/workflows/`. For an install-only orchestration path, use the
  >    bundled **stage-pipeline** agent instead, which needs no workflow file.

## Runtime prerequisites

The pipeline runs against a **host project** that supplies the build environment (not this
plugin):

- CANN toolkit at `/usr/local/Ascend/cann` (source `set_env.sh`; `bisheng` compiler).
- A Python venv with `torch_npu` (`.venv/bin/python`).
- `third_party/pto-isa` and an example include dir (e.g. `examples/megakda-pto/include`).
- Real Ascend NPU hardware (the authoritative validation gate) + the msprof simulator
  (advisory pre-filter).

See [CLAUDE.md](./CLAUDE.md) for the architecture, the phase model, and the non-negotiable
rules (provenance boundary, real-NPU gate, CPU-fp64 reference, coverage gate).

## Repository layout

```
npu-skillyard/
  .claude-plugin/
    plugin.json          # plugin manifest
    marketplace.json     # single-entry marketplace (source ".")
  skills/                # the four PTO skills
  agents/                # stage-pipeline, pto-stage-worker
  .mcp.json              # bundled npu-coding-mcp (needs launch config)
  .claude/
    workflows/           # pto-pipeline-parallel.js (clone-or-copy, not installable)
    settings.json        # in-repo dogfooding config
  CLAUDE.md
  README.md
```

## Developing / dogfooding in this repo

Because the components live at the repo root (plugin layout) rather than under
`.claude/`, to exercise them while developing here, install the local plugin:

```text
/plugin marketplace add ./
/plugin install npu-skillyard@npu-skillyard
```

Run `claude plugin validate .` before publishing.
