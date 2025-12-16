"""
Rename combined/task_[NNN]_[ORIG_ID] directories to [NNN]_[ORIG_ID].

Safe and idempotent: skips when destination exists.
"""

from __future__ import annotations

from pathlib import Path


COMBINED_DIR = Path("combined")


def main() -> None:
    if not COMBINED_DIR.is_dir():
        print(f"No combined directory at {COMBINED_DIR}")
        return
    renamed = 0
    skipped = 0
    for sub in sorted(COMBINED_DIR.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if not name.startswith("task_"):
            skipped += 1
            continue
        new_name = name[len("task_") :]
        dst = COMBINED_DIR / new_name
        if dst.exists():
            skipped += 1
            continue
        sub.rename(dst)
        renamed += 1
    print(f"Renamed {renamed} directories; skipped {skipped}.")


if __name__ == "__main__":
    main()

