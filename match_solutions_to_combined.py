"""
Match tasks from solutions/ tasks against combined/<ID>/<ID>.json by content.

- Scans solution tasks under solutions/tasks (files named taskNNN.json)
- Scans combined tasks under combined/*/*.json (file named <ID>.json)
- Computes a canonical JSON hash for each task and matches by hash equality
  to find which combined ID corresponds to each solution task file.

Outputs:
  - Prints a summary of counts and match coverage.
  - Writes combined/solutions_match.md with a table of matches and statuses.
  - Writes combined/solutions_match.json mapping solution filenames to IDs.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List


SOLUTIONS_TASKS_DIR = Path("solutions/tasks")
COMBINED_DIR = Path("combined")


def canonical_hash(obj: object) -> str:
    # Stable JSON with sorted keys; list order is preserved
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_solution_tasks() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not SOLUTIONS_TASKS_DIR.is_dir():
        return mapping
    for p in sorted(SOLUTIONS_TASKS_DIR.glob("task*.json")):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        mapping[p.name] = canonical_hash(obj)
    return mapping


def load_combined_tasks() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not COMBINED_DIR.is_dir():
        return mapping
    for sub in COMBINED_DIR.iterdir():
        if not sub.is_dir():
            continue
        # Combined directories contain <ID>.json; skip non-matching
        cand = sub / f"{sub.name}.json"
        if not cand.is_file():
            continue
        try:
            obj = json.loads(cand.read_text())
        except Exception:
            continue
        mapping[sub.name] = canonical_hash(obj)
    return mapping


def main() -> None:
    sol = load_solution_tasks()
    cmb = load_combined_tasks()

    # Invert combined mapping: hash -> list of IDs (defensive; collisions unlikely but allowed)
    inv: Dict[str, List[str]] = {}
    for cid, h in cmb.items():
        inv.setdefault(h, []).append(cid)

    matched: Dict[str, List[str]] = {}
    unmatched: List[str] = []
    for fname, h in sol.items():
        ids = inv.get(h, [])
        if ids:
            matched[fname] = sorted(ids)
        else:
            unmatched.append(fname)

    # Compute combined IDs that were matched by at least one solution task
    matched_ids = {cid for ids in matched.values() for cid in ids}
    unmatched_ids = sorted(set(cmb.keys()) - matched_ids)

    # Prepare report lines
    report_lines: List[str] = []
    report_lines.append("# Solutions ↔ Combined Task Match Report\n")
    report_lines.append("")
    report_lines.append(f"Solutions tasks: {len(sol)}  |  Combined IDs: {len(cmb)}")
    report_lines.append(f"Matched solutions: {len(matched)}  |  Unmatched solutions: {len(unmatched)}")
    report_lines.append(f"Matched combined IDs: {len(matched_ids)}  |  Unmatched combined IDs: {len(unmatched_ids)}\n")
    report_lines.append("")
    report_lines.append("## Per-solution matches")
    report_lines.append("")
    report_lines.append("| Solution Task | Matching Combined ID(s) | Status |")
    report_lines.append("| --- | --- | --- |")
    for fname in sorted(sol.keys()):
        ids = matched.get(fname, [])
        status = "unique" if len(ids) == 1 else ("multiple" if len(ids) > 1 else "none")
        rep_ids = ", ".join(ids) if ids else "—"
        report_lines.append(f"| {fname} | {rep_ids} | {status} |")
    report_lines.append("")
    report_lines.append("## Unmatched combined IDs")
    report_lines.append("")
    if unmatched_ids:
        report_lines.append(", ".join(unmatched_ids))
    else:
        report_lines.append("(none)")

    # Write outputs
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    (COMBINED_DIR / "solutions_match.md").write_text("\n".join(report_lines) + "\n")
    (COMBINED_DIR / "solutions_match.json").write_text(json.dumps(matched, indent=2))

    # Console summary
    print(f"Solutions tasks: {len(sol)} | Combined IDs: {len(cmb)}")
    print(f"Matched solutions: {len(matched)} | Unmatched solutions: {len(unmatched)}")
    print(f"Matched combined IDs: {len(matched_ids)} | Unmatched combined IDs: {len(unmatched_ids)}")
    print(f"Wrote report to {COMBINED_DIR / 'solutions_match.md'} and JSON to {COMBINED_DIR / 'solutions_match.json'}")


if __name__ == "__main__":
    main()

