from __future__ import annotations

from pathlib import Path
from typing import List


def _list_filenames(dir_path: Path) -> List[str]:
    """Return a sorted list of regular file names in a directory (non-recursive).

    If the directory does not exist, returns an empty list.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted(p.name for p in dir_path.iterdir() if p.is_file())


# Base directory: repository root where this file lives
_ROOT = Path(__file__).resolve().parent

# ARC-AGI v1 directories
_ARC1_TRAIN_DIR = _ROOT / "ARC-AGI" / "data" / "training"
_ARC1_EVAL_DIR = _ROOT / "ARC-AGI" / "data" / "evaluation"

# ARC-AGI v2 directories
_ARC2_TRAIN_DIR = _ROOT / "ARC-AGI-2" / "data" / "training"
_ARC2_EVAL_DIR = _ROOT / "ARC-AGI-2" / "data" / "evaluation"


# Public lists of filenames (basenames including extensions)
arc_agi_1_train: List[str] = _list_filenames(_ARC1_TRAIN_DIR)
arc_agi_1_evaluate: List[str] = _list_filenames(_ARC1_EVAL_DIR)
arc_agi_2_train: List[str] = _list_filenames(_ARC2_TRAIN_DIR)
arc_agi_2_evaluate: List[str] = _list_filenames(_ARC2_EVAL_DIR)


__all__ = [
    "arc_agi_1_train",
    "arc_agi_1_evaluate",
    "arc_agi_2_train",
    "arc_agi_2_evaluate",
]


if __name__ == "__main__":
    # Print brief summary when executed directly
    print(f"arc_agi_1_train: {len(arc_agi_1_train)} files")
    print(f"arc_agi_1_evaluate: {len(arc_agi_1_evaluate)} files")
    print(f"arc_agi_2_train: {len(arc_agi_2_train)} files")
    print(f"arc_agi_2_evaluate: {len(arc_agi_2_evaluate)} files")

    # Also write lists to text files (one filename per line)
    def _write_list(out_path: Path, items: List[str]) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(f"{item}\n")

    out_dir = _ROOT / "lists"
    outputs = {
        out_dir / "arc-agi-1-train.txt": arc_agi_1_train,
        out_dir / "arc-agi-1-evaluate.txt": arc_agi_1_evaluate,
        out_dir / "arc-agi-2-train.txt": arc_agi_2_train,
        out_dir / "arc-agi-2-evaluate.txt": arc_agi_2_evaluate,
    }

    for path, items in outputs.items():
        _write_list(path, items)
        print(f"wrote {len(items)} lines to {path.relative_to(_ROOT)}")
