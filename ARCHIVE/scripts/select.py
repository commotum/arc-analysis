#!/usr/bin/env python3
"""
Select 10 random folders from the `collective/` directory and append their
original task IDs to `easy.md` (one per line).

Usage:
  python select.py            # selects 10 and appends to easy.md

Notes:
  - The task ID is parsed as the substring after the last underscore in the
    folder name (e.g., '001_007bbfb7' -> '007bbfb7').
  - If there are fewer than 10 folders, all available folders are used.
"""

from __future__ import annotations

import os
import random
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent


def list_task_dirs(collective_dir: Path) -> list[Path]:
    if not collective_dir.exists() or not collective_dir.is_dir():
        raise FileNotFoundError(f"collective directory not found: {collective_dir}")
    return [p for p in collective_dir.iterdir() if p.is_dir()]


def extract_task_id(folder_name: str) -> str:
    # Take everything after the last underscore; fallback to full name if none
    if "_" in folder_name:
        return folder_name.rsplit("_", 1)[-1]
    return folder_name


def main() -> None:
    root = get_repo_root()
    collective_dir = root / "collective"
    easy_md = root / "easy.md"

    task_dirs = list_task_dirs(collective_dir)
    if not task_dirs:
        raise RuntimeError("No task folders found under 'collective/'.")

    k = 10
    sample_size = min(k, len(task_dirs))
    selected = random.sample(task_dirs, sample_size)

    # Extract IDs and append to easy.md
    ids = [extract_task_id(p.name) for p in selected]
    with easy_md.open("a", encoding="utf-8") as f:
        for tid in ids:
            f.write(f"{tid}\n")

    # Print for convenience/visibility
    print("Appended the following IDs to easy.md:")
    for tid in ids:
        print(tid)


if __name__ == "__main__":
    main()

