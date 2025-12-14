"""
Copy each solution file from solutions/solutions/taskNNN.py into the
corresponding combined/task_[NNN]_[ORIG_ID]/ directory, naming it
solution_[NNN]_[ORIG_ID].py.

Relies on combined/task_matches.json produced by compare_task_sets.py.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


SOL_SOLUTIONS_DIR = Path("solutions/solutions")
COMBINED_DIR = Path("combined")
MATCHES_JSON = COMBINED_DIR / "task_matches.json"


def load_solution_to_id_map() -> dict[str, str]:
    if not MATCHES_JSON.is_file():
        raise FileNotFoundError(
            f"Mapping file not found: {MATCHES_JSON}. Run compare_task_sets.py first."
        )
    data = json.loads(MATCHES_JSON.read_text())
    matched = data.get("solutions", {}).get("matched", {})
    out: dict[str, str] = {}
    for sol_name, ids in matched.items():
        if ids:
            out[sol_name] = ids[0]
    return out


def main() -> None:
    sol_to_id = load_solution_to_id_map()
    copied = 0
    missing_dir = 0
    skipped = 0

    for p in sorted(SOL_SOLUTIONS_DIR.glob("task*.py")):
        stem = p.stem  # taskNNN
        idx = stem.replace("task", "")
        key = f"task{idx}.json"
        orig_id = sol_to_id.get(key)
        if not orig_id:
            skipped += 1
            continue
        dest_dir = COMBINED_DIR / f"task_{idx}_{orig_id}"
        if not dest_dir.is_dir():
            # Fallback to old layout combined/<orig_id>/
            alt_dir = COMBINED_DIR / orig_id
            if alt_dir.is_dir():
                dest_dir = alt_dir
            else:
                missing_dir += 1
                continue
        dest = dest_dir / f"solution_{idx}_{orig_id}.py"
        shutil.copy2(p, dest)
        copied += 1

    print(f"Copied {copied} solutions. Skipped {skipped}. Missing dest dirs {missing_dir}.")


if __name__ == "__main__":
    main()

