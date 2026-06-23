#!/usr/bin/env python3
"""Deterministically add ``abi`` to every stage in a stage_plan.json.

Derives the call_kernel argument list from the stage's inputs, outputs,
problem dimensions, and instruction families — all already present in the
stage plan.  No LLM involvement.

Usage:
    python enrich_stage_plan.py <stage_plan.json>

Reads the file in-place, writes it back with an ``abi`` field added to
each stage.  Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CTYPES_MAP: dict[str, str] = {
    "uint32_t": "ctypes.c_uint32",
    "void*": "ctypes.c_void_p",
    "uint8_t*": "ctypes.c_void_p",
    "int64_t": "ctypes.c_int64",
    "int32_t": "ctypes.c_int32",
}


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _build_abi(stage: dict[str, Any]) -> dict[str, Any]:
    """Derive a deterministic kernel ABI from one stage dict."""
    stage_name = str(stage.get("name") or "stage")
    inputs = stage.get("inputs") if isinstance(stage.get("inputs"), list) else []
    outputs = stage.get("outputs") if isinstance(stage.get("outputs"), list) else []
    problem = stage.get("problem") if isinstance(stage.get("problem"), dict) else {}
    families = _as_str_list(stage.get("instruction_families"))

    needs_workspace = any(
        f in families for f in ("TMATMUL", "TMATMUL_ACC", "TTRI", "TGEMV")
    )

    args: list[dict[str, Any]] = [
        {"index": 0, "name": "block_dim", "ctype": "uint32_t"},
        {"index": 1, "name": "stream", "ctype": "void*"},
    ]
    idx = 2

    for inp in inputs:
        name = str(inp.get("name") or f"in{idx}").strip()
        args.append({"index": idx, "name": name, "ctype": "uint8_t*"})
        idx += 1

    for out in outputs:
        name = str(out.get("name") or f"out{idx}").strip()
        args.append({"index": idx, "name": name, "ctype": "uint8_t*"})
        idx += 1

    if needs_workspace:
        args.append({"index": idx, "name": "workspace", "ctype": "uint8_t*"})
        idx += 1

    args.append({"index": idx, "name": "total_work", "ctype": "int64_t"})
    idx += 1

    _existing_names = {a["name"] for a in args}
    for k, v in problem.items():
        dim_name = str(k).lower().strip()
        if dim_name in _existing_names:
            continue
        if isinstance(v, str):
            sym = v.strip()
            if sym and sym.replace("_", "").isalnum() and sym not in _existing_names:
                args.append({"index": idx, "name": sym, "ctype": "int64_t"})
                _existing_names.add(sym)
                idx += 1
        elif isinstance(v, (int, float)):
            args.append({"index": idx, "name": dim_name, "ctype": "int64_t"})
            _existing_names.add(dim_name)
            idx += 1

    for a in args:
        a["ctypes"] = CTYPES_MAP.get(a["ctype"], "ctypes.c_int64")

    sig_parts = [f"{a['ctype']} {a['name']}" for a in args]
    signature = f"extern \"C\" void call_kernel({', '.join(sig_parts)})"

    return {
        "schema_version": "kernel_abi_v1",
        "entrypoint_symbol": "call_kernel",
        "signature": signature,
        "launch_kind": "host_stub_function_launch",
        "launch_symbol": None,
        "needs_workspace": needs_workspace,
        "arguments": args,
    }


def enrich(path: str) -> bool:
    """Read *path*, add ``abi`` to every stage, write back.  Return True on success."""
    try:
        data = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"enrich_stage_plan: cannot read {path}: {exc}", file=sys.stderr)
        return False

    if not isinstance(data, dict) or not isinstance(data.get("stages"), list):
        print(f"enrich_stage_plan: {path} is not a valid stage plan", file=sys.stderr)
        return False

    for i, stage in enumerate(data["stages"]):
        if not isinstance(stage, dict):
            print(f"enrich_stage_plan: stages[{i}] is not an object — skipping", file=sys.stderr)
            continue
        stage["abi"] = _build_abi(stage)

    Path(path).write_text(json.dumps(data, indent=2) + "\n")
    return True


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv
    if len(argv) < 2:
        print(f"Usage: {argv[0]} <stage_plan.json>", file=sys.stderr)
        return 2
    return 0 if enrich(argv[1]) else 1


if __name__ == "__main__":
    sys.exit(main())
