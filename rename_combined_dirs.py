"""
Rename combined/<ORIG_ID>/ directories to task_[SOLUTION-ID]_[ORIGINAL-ID].

Uses the mapping produced by compare_task_sets.py (combined/task_matches.json)
to map solution taskNNN.json -> ORIGINAL-ID, then inverts it to rename
combined/<ORIGINAL-ID>/ -> combined/task_[NNN]_[ORIGINAL-ID].

Idempotent: skips if destination already exists with the correct name.
"""

from __future__ import annotations

import json
from pathlib import Path

COMBINED_DIR = Path("combined")
MATCHES_JSON = COMBINED_DIR / "task_matches.json"


def load_solution_to_id_map() -> dict[str, str]:
    if not MATCHES_JSON.is_file():
        raise FileNotFoundError(
            f"Mapping file not found: {MATCHES_JSON}. Run compare_task_sets.py first."
        )
    data = json.loads(MATCHES_JSON.read_text())
    matched = data.get("solutions", {}).get("matched", {})
    # matched: dict of solution filename -> [orig_id]
    out: dict[str, str] = {}
    for sol_name, ids in matched.items():
        if ids:
            out[sol_name] = ids[0]
    return out


def solution_name_to_index(sol_name: str) -> str:
    # Expect format taskNNN.json
    stem = Path(sol_name).stem
    if not stem.startswith("task"):
        raise ValueError(f"Unexpected solution task name: {sol_name}")
    idx = stem[len("task") :]
    return idx


def main() -> None:
    sol_to_id = load_solution_to_id_map()
    # Build orig_id -> sol_idx map (if duplicates, last wins — not expected)
    orig_to_idx = {orig: solution_name_to_index(sol) for sol, orig in sol_to_id.items()}

    renamed = 0
    skipped = 0
    for orig_id, sol_idx in sorted(orig_to_idx.items()):
        src = COMBINED_DIR / orig_id
        dst = COMBINED_DIR / f"task_{sol_idx}_{orig_id}"
        if not src.exists() and dst.exists():
            skipped += 1
            continue
        if not src.exists():
            # Neither src nor dst? nothing to do
            skipped += 1
            continue
        if dst.exists():
            # Destination already exists; assume already renamed
            skipped += 1
            continue
        src.rename(dst)
        renamed += 1
    print(f"Renamed {renamed} directories; skipped {skipped}.")


if __name__ == "__main__":
    main()

