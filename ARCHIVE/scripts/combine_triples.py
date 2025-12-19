"""
Combine ARC triples (generator, verifier, task json) into combined/<ID>/.

Scans:
  - Generators: re-arc/generators/generate_*.py or generator_*.py
  - Verifiers:  re-arc/verifiers/verify_*.py
  - Tasks:      tasks/training/*.json and tasks/evaluation/*.json

For each task ID/name present under tasks/, if both generator and verifier exist
with the same ID, creates combined/<ID>/ and copies:
  - generator file (renamed to generator_<ID>.py if source is generate_<ID>.py)
  - verifier file (verify_<ID>.py)
  - task json (<ID>.json)

Also writes combined/missing.md listing task IDs/names missing a generator or verifier.
"""

from __future__ import annotations

import re
from pathlib import Path
import shutil
from datetime import datetime


GEN_DIR = Path("re-arc/generators")
VER_DIR = Path("re-arc/verifiers")
TASK_TRAIN_DIR = Path("tasks/training")
TASK_EVAL_DIR = Path("tasks/evaluation")
COMBINED_DIR = Path("combined")


GEN_PAT = re.compile(r"^(?:generate|generator)_([0-9a-f]{8})\.py$")
VER_PAT = re.compile(r"^verify_([0-9a-f]{8})\.py$")


def scan_generators() -> dict[str, Path]:
    out: dict[str, Path] = {}
    if GEN_DIR.is_dir():
        for p in GEN_DIR.glob("*.py"):
            m = GEN_PAT.match(p.name)
            if m:
                out[m.group(1)] = p
    return out


def scan_verifiers() -> dict[str, Path]:
    out: dict[str, Path] = {}
    if VER_DIR.is_dir():
        for p in VER_DIR.glob("*.py"):
            m = VER_PAT.match(p.name)
            if m:
                out[m.group(1)] = p
    return out


def scan_tasks() -> dict[str, Path]:
    # Prefer training over evaluation if both exist for the same ID.
    out: dict[str, Path] = {}
    # Insert training first, then do not override with evaluation.
    for d in (TASK_TRAIN_DIR, TASK_EVAL_DIR):
        if d.is_dir():
            for p in d.glob("*.json"):
                out.setdefault(p.stem, p)
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_with_name(src: Path, dst_dir: Path, dst_name: str | None = None) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / (dst_name if dst_name else src.name)
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    gens = scan_generators()
    vers = scan_verifiers()
    tasks = scan_tasks()

    ensure_dir(COMBINED_DIR)

    created = 0
    missing_rows: list[tuple[str, bool, bool]] = []

    all_ids = sorted(set(tasks.keys()) | set(gens.keys()) | set(vers.keys()))
    for tid in all_ids:
        has_task = tid in tasks
        has_gen = tid in gens
        has_ver = tid in vers
        if has_task and has_gen and has_ver:
            dest = COMBINED_DIR / tid
            ensure_dir(dest)
            # Copy generator; normalize name to generator_<ID>.py
            gen_src = gens[tid]
            gen_dst_name = gen_src.name
            m = GEN_PAT.match(gen_src.name)
            if m and gen_src.name.startswith("generate_"):
                gen_dst_name = f"generator_{tid}.py"
            copy_with_name(gen_src, dest, gen_dst_name)
            # Copy verifier (keep name)
            ver_src = vers[tid]
            copy_with_name(ver_src, dest)
            # Copy task json as <ID>.json
            copy_with_name(tasks[tid], dest, f"{tid}.json")
            created += 1
        else:
            # Record IDs missing either generator or verifier (or task json)
            missing_rows.append((tid, has_gen, has_ver))

    # Write missing.md summary
    missing_md = COMBINED_DIR / "missing.md"
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Missing Triples (generated {ts})\n")
    lines.append("")
    lines.append("| Task ID/Name | Generator | Verifier |")
    lines.append("| --- | --- | --- |")
    for tid, has_gen, has_ver in missing_rows:
        g = "Yes" if has_gen else "No"
        v = "Yes" if has_ver else "No"
        lines.append(f"| {tid} | {g} | {v} |")
    missing_md.write_text("\n".join(lines) + "\n")

    print(f"Created {created} combined task directories under '{COMBINED_DIR}'.")
    print(f"Wrote missing summary to {missing_md} ({len(missing_rows)} entries).")


if __name__ == "__main__":
    main()
