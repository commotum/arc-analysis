"""
Compare tasks from two sources against combined/<ID>/<ID>.json to find matches by content.

Sources scanned:
  - Solutions: solutions/tasks/taskNNN.json (ignores the optional 'arc-gen' field)
  - Original:  tasks/training/*.json and tasks/evaluation/*.json

Robust matching logic (format invariant):
  - Canonicalize each train/test example pair (input/output) up to:
      • example order (treated as a multiset)
      • color relabeling (consistent remap across input+output, background 0 kept as 0)
      • any of 8 dihedral transforms (rotations/flips), applied jointly to input+output
  - A task signature is the multiset of canonical example fingerprints.
  - Build reverse index from signature hash -> combined ID(s).
  - For each source task, compute signature hash and look up matching combined IDs.

Outputs:
  - Prints summary counts to stdout.
  - Writes combined/task_matches.md with sections for each source and coverage stats.
  - Writes combined/task_matches.json with mapping details.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


SOLUTIONS_TASKS_DIR = Path("solutions/tasks")
ORIG_TRAIN_DIR = Path("tasks/training")
ORIG_EVAL_DIR = Path("tasks/evaluation")
COMBINED_DIR = Path("combined")


def canonical_task_payload(obj: dict) -> dict:
    # Keep only 'train' and 'test' keys to avoid extraneous fields like 'arc-gen'
    return {"train": obj.get("train", []), "test": obj.get("test", [])}


def dihedral_transforms(grid: List[List[int]]) -> List[List[List[int]]]:
    # Return the 8 dihedral transforms of a grid: rotations + mirror variants
    def rot90(g):
        h, w = len(g), len(g[0])
        return [[g[h - 1 - i][j] for i in range(h)] for j in range(w)]

    def rot180(g):
        return [row[::-1] for row in reversed(g)]

    def rot270(g):
        h, w = len(g), len(g[0])
        return [[g[i][w - 1 - j] for i in range(h)] for j in range(w)]

    def flip_h(g):  # horizontal mirror (left-right)
        return [row[::-1] for row in g]

    i = grid
    r90 = rot90(i)
    r180 = rot180(i)
    r270 = rot270(i)
    f = flip_h(i)
    f90 = rot90(f)
    f180 = rot180(f)
    f270 = rot270(f)
    return [i, r90, r180, r270, f, f90, f180, f270]


def remap_colors_pair(inp: List[List[int]], out: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    # Build a consistent mapping across input+output, keeping 0 -> 0 if present
    def flatten(g):
        for row in g:
            for v in row:
                yield v

    colors = set(flatten(inp)) | set(flatten(out))
    mapping: Dict[int, int] = {}
    next_id = 0
    if 0 in colors:
        mapping[0] = 0
        next_id = 1
    for v in list(flatten(inp)) + list(flatten(out)):
        if v not in mapping:
            mapping[v] = next_id
            next_id += 1

    def apply_map(g):
        return [[mapping[v] for v in row] for row in g]

    return apply_map(inp), apply_map(out)


def example_fingerprint(inp: List[List[int]], out: List[List[int]]) -> str:
    # Canonical fingerprint over dihedral transforms and color relabeling.
    # Apply same transform to input+output, choose lexicographically smallest string.
    reps = []
    for t_idx in range(8):
        in_t = dihedral_transforms(inp)[t_idx]
        out_t = dihedral_transforms(out)[t_idx]
        in_m, out_m = remap_colors_pair(in_t, out_t)
        h_i, w_i = len(in_m), len(in_m[0])
        h_o, w_o = len(out_m), len(out_m[0])
        s = (
            f"I{h_i}x{w_i}=" + ",".join("".join(map(str, row)) for row in in_m)
            + "|"
            + f"O{h_o}x{w_o}=" + ",".join("".join(map(str, row)) for row in out_m)
        )
        reps.append(s)
    return min(reps)


def task_signature(obj: dict) -> str:
    # Multiset of example fingerprints across train+test, order-invariant
    payload = canonical_task_payload(obj)
    fps = []
    for split in ("train", "test"):
        for ex in payload.get(split, []):
            fps.append(example_fingerprint(ex["input"], ex["output"]))
    cnt = Counter(fps)
    # Deterministic string: sorted items with multiplicity
    parts = [f"{k}*{cnt[k]}" for k in sorted(cnt)]
    data = "\n".join(parts)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_combined_index() -> Dict[str, List[str]]:
    """Return reverse index: hash -> [combined_id,...]."""
    rev: Dict[str, List[str]] = {}
    if not COMBINED_DIR.is_dir():
        return rev
    for sub in COMBINED_DIR.iterdir():
        if not sub.is_dir():
            continue
        p = sub / f"{sub.name}.json"
        if not p.is_file():
            continue
        try:
            h = task_signature(json.loads(p.read_text()))
        except Exception:
            continue
        rev.setdefault(h, []).append(sub.name)
    return rev


def scan_solutions() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not SOLUTIONS_TASKS_DIR.is_dir():
        return mapping
    for p in sorted(SOLUTIONS_TASKS_DIR.glob("task*.json")):
        try:
            h = task_signature(json.loads(p.read_text()))
        except Exception:
            continue
        mapping[p.name] = h
    return mapping


def scan_original() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for d in (ORIG_TRAIN_DIR, ORIG_EVAL_DIR):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                h = task_signature(json.loads(p.read_text()))
            except Exception:
                continue
            # Prefer training if duplicate stems exist
            mapping.setdefault(p.stem, h)
    return mapping


def make_report_rows(source_map: Dict[str, str], rev_index: Dict[str, List[str]]) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    matched: List[Tuple[str, List[str]]] = []
    unmatched: List[str] = []
    for name, h in source_map.items():
        ids = sorted(rev_index.get(h, []))
        if ids:
            matched.append((name, ids))
        else:
            unmatched.append(name)
    return matched, unmatched


def main() -> None:
    rev = load_combined_index()
    sol = scan_solutions()
    orig = scan_original()

    sol_matched, sol_unmatched = make_report_rows(sol, rev)
    orig_matched, orig_unmatched = make_report_rows(orig, rev)

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    # Write markdown report
    lines: List[str] = []
    lines.append("# Task Match Report\n")
    lines.append("")
    lines.append(f"Combined IDs: {sum(len(v) for v in rev.values())} unique hashes: {len(rev)}")
    lines.append("")
    lines.append("## Solutions/tasks")
    lines.append("")
    lines.append(f"Total: {len(sol)}  |  Matched: {len(sol_matched)}  |  Unmatched: {len(sol_unmatched)}\n")
    lines.append("| Solution Task | Matching Combined ID(s) | Status |")
    lines.append("| --- | --- | --- |")
    for name, ids in sorted(sol_matched):
        status = "unique" if len(ids) == 1 else "multiple"
        lines.append(f"| {name} | {', '.join(ids)} | {status} |")
    for name in sorted(sol_unmatched):
        lines.append(f"| {name} | — | none |")
    lines.append("")
    lines.append("## Original tasks (training/evaluation)")
    lines.append("")
    lines.append(f"Total: {len(orig)}  |  Matched: {len(orig_matched)}  |  Unmatched: {len(orig_unmatched)}\n")
    lines.append("| Original ID | Matching Combined ID(s) | Status |")
    lines.append("| --- | --- | --- |")
    for name, ids in sorted(orig_matched):
        status = "unique" if len(ids) == 1 else "multiple"
        lines.append(f"| {name} | {', '.join(ids)} | {status} |")
    for name in sorted(orig_unmatched):
        lines.append(f"| {name} | — | none |")

    (COMBINED_DIR / "task_matches.md").write_text("\n".join(lines) + "\n")

    # JSON output
    out_json = {
        "solutions": {
            "matched": {name: ids for name, ids in sol_matched},
            "unmatched": sol_unmatched,
        },
        "original": {
            "matched": {name: ids for name, ids in orig_matched},
            "unmatched": orig_unmatched,
        },
    }
    (COMBINED_DIR / "task_matches.json").write_text(json.dumps(out_json, indent=2))

    # Console summary
    print(f"Solutions: {len(sol)} total | matched {len(sol_matched)} | unmatched {len(sol_unmatched)}")
    print(f"Original:  {len(orig)} total | matched {len(orig_matched)} | unmatched {len(orig_unmatched)}")
    print(f"Reports: {COMBINED_DIR / 'task_matches.md'} and {COMBINED_DIR / 'task_matches.json'}")


if __name__ == "__main__":
    main()
