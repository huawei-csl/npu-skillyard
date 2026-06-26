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
- A Python interpreter with `torch_npu`.
- `pto-isa` headers (auto-cloned by Preflight from `gitcode.com/cann/pto-isa` by default).
- Real Ascend NPU hardware (the authoritative validation gate) + the msprof simulator
  (advisory pre-filter).

> `kernel_common.h` (the single boilerplate header every kernel includes) is **bundled** in
> the plugin at `include/kernel_common.h` -- Preflight drops it into the run dir, so you do
> **not** need an external example include directory. It only pulls CANN + pto-isa headers,
> both resolved above.

**Paths are not hardcoded.** Both drivers start with a **Preflight** step that resolves
each path in priority order (**explicit arg → env var → autodetect → documented default**)
and validates it before any work:

| Path | arg | env var | default |
|---|---|---|---|
| python (torch_npu) | `pto_python` | `$PTO_PYTHON` | `./.venv/bin/python` |
| pto-isa root | `pto_isa_root` | `$PTO_LIB_PATH` | `./third_party/pto-isa` |
| include dir (`kernel_common.h`) | `include_dir` | `$PTO_INCLUDE_DIR` | bundled `include/` (copied into the run dir) |
| pto-isa clone URL | `pto_isa_repo` | `$PTO_ISA_REPO` | `https://gitcode.com/cann/pto-isa.git` |

CANN, `bisheng`, and the NPU device **cannot be auto-installed** -- if any is missing,
Preflight STOPs early with a clear message instead of failing mid-run. `pto-isa` is just
source: if its path is absent, Preflight clones it automatically (from `pto_isa_repo` >
`$PTO_ISA_REPO` > the default `gitcode.com/cann/pto-isa`).

**`torch_npu` is detect-and-stop by default** (it's tightly version-coupled to the installed
CANN release -- a wrong pin can silently produce wrong numerics, so the default is to bring
your own). To provision it automatically, pass `bootstrap_venv: true`: when no working
torch_npu python is found, Preflight creates a venv (`bootstrap_venv_path`, default
`./.venv-npu`) and installs `torch`/`torch_npu` matched to the detected CANN version
(`torch_version` / `torch_npu_version` override the pins), then **re-validates** and still
STOPs if the result can't see the NPU.

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
  include/               # bundled kernel_common.h (the only build header kernels include)
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
