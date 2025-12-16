"""
Rename files in combined/<ORIG_ID>/ to include both solution index and original ARC ID.

New names inside each folder:
  - task_[SOLUTION-ID]_[ORIGINAL-ID].json
  - generator_[SOLUTION-ID]_[ORIGINAL-ID].py
  - verifier_[SOLUTION-ID]_[ORIGINAL-ID].py

Relies on mapping from compare_task_sets.py (combined/task_matches.json), which maps
solutions/tasks/taskNNN.json -> [ORIGINAL-ID]. If that file is absent, exit with message.

Idempotent: if target names already exist, it skips renaming for that item.
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
        if not ids:
            continue
        out[sol_name] = ids[0]
    return out


def solution_name_to_index(sol_name: str) -> str:
    # Expect format taskNNN.json
    stem = Path(sol_name).stem
    if not stem.startswith("task"):
        raise ValueError(f"Unexpected solution task name: {sol_name}")
    idx = stem[len("task") :]
    if not idx.isdigit() or len(idx) != 3:
        # Keep as-is but warn by returning original
        return idx
    return idx


def rename_for_id(orig_id: str, sol_idx: str) -> dict[str, bool]:
    d = COMBINED_DIR / orig_id
    changed = {"json": False, "gen": False, "ver": False}
    if not d.is_dir():
        return changed

    # JSON file
    src_json = d / f"{orig_id}.json"
    dst_json = d / f"task_{sol_idx}_{orig_id}.json"
    if dst_json.exists():
        pass
    elif src_json.exists():
        src_json.rename(dst_json)
        changed["json"] = True

    # Generator file: prefer generator_<id>.py, fallback generate_<id>.py
    gen_src_candidates = [d / f"generator_{orig_id}.py", d / f"generate_{orig_id}.py"]
    gen_src = next((p for p in gen_src_candidates if p.exists()), None)
    gen_dst = d / f"generator_{sol_idx}_{orig_id}.py"
    if gen_dst.exists():
        pass
    elif gen_src and gen_src.exists():
        gen_src.rename(gen_dst)
        changed["gen"] = True

    # Verifier file: support verify_<id>.py or verifier_<id>.py
    ver_src_candidates = [d / f"verify_{orig_id}.py", d / f"verifier_{orig_id}.py"]
    ver_src = next((p for p in ver_src_candidates if p.exists()), None)
    ver_dst = d / f"verifier_{sol_idx}_{orig_id}.py"
    if ver_dst.exists():
        pass
    elif ver_src and ver_src.exists():
        ver_src.rename(ver_dst)
        changed["ver"] = True

    return changed


def main() -> None:
    sol_map = load_solution_to_id_map()
    # Build reverse map from orig_id -> solution index
    rev: dict[str, str] = {}
    for sol_name, orig_id in sol_map.items():
        rev[orig_id] = solution_name_to_index(sol_name)

    total = 0
    j = g = v = 0
    missing = []
    for orig_id, sol_idx in rev.items():
        total += 1
        res = rename_for_id(orig_id, sol_idx)
        j += int(res["json"]) ; g += int(res["gen"]) ; v += int(res["ver"]) 
        # Check presence afterwards
        d = COMBINED_DIR / orig_id
        if not (d / f"task_{sol_idx}_{orig_id}.json").exists():
            missing.append((orig_id, "json"))
        if not (d / f"generator_{sol_idx}_{orig_id}.py").exists():
            missing.append((orig_id, "generator"))
        if not (d / f"verifier_{sol_idx}_{orig_id}.py").exists():
            missing.append((orig_id, "verifier"))

    print(f"Processed {total} combined tasks. Renamed: json={j}, generator={g}, verifier={v}.")
    if missing:
        print(f"Missing expected files for {len(missing)} entries. Examples:")
        for tup in missing[:10]:
            print("  ", tup)


if __name__ == "__main__":
    main()

