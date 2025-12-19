from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any


ROOT = Path(__file__).resolve().parent

# Lists of filenames (basenames) to use as authoritative sets
LISTS_DIR = ROOT / "lists"
ARC1_TRAIN_LIST = LISTS_DIR / "arc-agi-1-train.txt"
ARC1_EVAL_LIST = LISTS_DIR / "arc-agi-1-evaluate.txt"
ARC2_TRAIN_LIST = LISTS_DIR / "arc-agi-2-train.txt"
ARC2_EVAL_LIST = LISTS_DIR / "arc-agi-2-evaluate.txt"

# Source directories
DATA_DIR = ROOT / "DATA"
SRC_ARC1_TRAIN = DATA_DIR / "ARC-1-TRAIN"
SRC_ARC2_TRAIN = DATA_DIR / "ARC-2-TRAIN"
SRC_ARC1_TEST = DATA_DIR / "ARC-1-TEST"
SRC_ARC2_TEST = DATA_DIR / "ARC-2-TEST"

# Destination directories
DEST_ALL_TRAIN = DATA_DIR / "ARC-ALL-TRAIN"
DEST_ALL_TEST = DATA_DIR / "ARC-ALL-TEST"


def read_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"List not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_equal(a: Path, b: Path) -> bool:
    with a.open("r", encoding="utf-8") as fa, b.open("r", encoding="utf-8") as fb:
        try:
            ja = json.load(fa)
            jb = json.load(fb)
        except json.JSONDecodeError:
            # Fallback: raw bytes if not valid JSON for some reason
            fa.seek(0)
            fb.seek(0)
            return fa.read() == fb.read()
    return ja == jb


def _norm_number(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    raise TypeError("Grid elements must be numbers")


def _norm_grid(g: Any) -> List[List[int]]:
    # Normalize a grid (list of lists of numbers) to nested lists of ints
    if not isinstance(g, list):
        raise TypeError("Grid must be a list")
    out: List[List[int]] = []
    for row in g:
        if not isinstance(row, list):
            raise TypeError("Grid rows must be lists")
        out.append([_norm_number(v) for v in row])
    return out


def _extract_grids(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the relevant grid content from an ARC task JSON.

    Returns a dict with keys:
      - train: List[Tuple[input_grid, output_grid]]
      - test_inputs: List[input_grid]
      - test_outputs: Optional[List[output_grid]] present only if outputs exist in all test cases
    """
    res: Dict[str, Any] = {}
    train = []
    for ex in obj.get("train", []) or []:
        if not isinstance(ex, dict):
            continue
        i = ex.get("input")
        o = ex.get("output")
        if i is None or o is None:
            continue
        train.append((_norm_grid(i), _norm_grid(o)))
    res["train"] = train

    test_inputs: List[List[List[int]]] = []
    test_outputs: List[List[List[int]]] = []
    outputs_present = True
    for ex in obj.get("test", []) or []:
        if not isinstance(ex, dict):
            continue
        i = ex.get("input")
        if i is None:
            continue
        test_inputs.append(_norm_grid(i))
        if "output" in ex and ex["output"] is not None:
            test_outputs.append(_norm_grid(ex["output"]))
        else:
            outputs_present = False
    res["test_inputs"] = test_inputs
    if outputs_present and len(test_outputs) == len(test_inputs):
        res["test_outputs"] = test_outputs
    return res


def grids_equal(a: Path, b: Path) -> bool:
    with a.open("r", encoding="utf-8") as fa, b.open("r", encoding="utf-8") as fb:
        ja = json.load(fa)
        jb = json.load(fb)
    ga = _extract_grids(ja if isinstance(ja, dict) else {})
    gb = _extract_grids(jb if isinstance(jb, dict) else {})
    # Compare train pairs
    if len(ga.get("train", [])) != len(gb.get("train", [])):
        return False
    for (ai, ao), (bi, bo) in zip(ga.get("train", []), gb.get("train", [])):
        if ai != bi or ao != bo:
            return False
    # Compare test inputs
    if ga.get("test_inputs", []) != gb.get("test_inputs", []):
        return False
    # Compare test outputs only if both have them
    ao = ga.get("test_outputs")
    bo = gb.get("test_outputs")
    if (ao is None) != (bo is None):
        # One side lacks outputs: ignore outputs difference
        return True
    if ao is not None and bo is not None and ao != bo:
        return False
    return True


def verify_duplicates(names: Iterable[str], a_dir: Path, b_dir: Path) -> Tuple[int, List[Tuple[str, Path, Path]]]:
    """Verify duplicates between two dirs for given names.

    Returns (count_verified, mismatches) where mismatches is a list of (name, path_a, path_b).
    Only names that exist in both a_dir and b_dir are considered duplicates.
    """
    verified = 0
    mismatches: List[Tuple[str, Path, Path]] = []
    for name in names:
        pa = a_dir / name
        pb = b_dir / name
        if pa.is_file() and pb.is_file():
            if json_equal(pa, pb) or grids_equal(pa, pb):
                verified += 1
            else:
                mismatches.append((name, pa, pb))
    return verified, mismatches


def copy_union(names_a: Iterable[str], names_b: Iterable[str], src_a: Path, src_b: Path, dest: Path) -> Tuple[int, int]:
    """Copy union of names from src_a and src_b into dest.

    Returns (copied, skipped_existing)
    """
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    all_names = []
    seen = set()
    for name in names_a:
        if name not in seen:
            seen.add(name)
            all_names.append(name)
    for name in names_b:
        if name not in seen:
            seen.add(name)
            all_names.append(name)

    for name in all_names:
        dst = dest / name
        if dst.exists():
            skipped += 1
            continue
        src = src_a / name
        if not src.is_file():
            src = src_b / name
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found in either dir for {name}: {src_a} or {src_b}")
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def main() -> None:
    # Read lists
    arc1_train = read_list(ARC1_TRAIN_LIST)
    arc2_train = read_list(ARC2_TRAIN_LIST)
    arc1_eval = read_list(ARC1_EVAL_LIST)
    arc2_eval = read_list(ARC2_EVAL_LIST)

    # Verify duplicates for TRAIN
    train_dupes = sorted(set(arc1_train).intersection(arc2_train))
    vcount, mismatches = verify_duplicates(train_dupes, SRC_ARC1_TRAIN, SRC_ARC2_TRAIN)
    print(f"TRAIN duplicates: {len(train_dupes)}, verified equal: {vcount}")
    if mismatches:
        print("TRAIN mismatches found:")
        for name, pa, pb in mismatches[:10]:
            print(f" - {name}: {pa} != {pb}")
        raise SystemExit(f"Aborting due to {len(mismatches)} TRAIN mismatches")

    # Verify duplicates for TEST/EVAL
    eval_dupes = sorted(set(arc1_eval).intersection(arc2_eval))
    vcount2, mismatches2 = verify_duplicates(eval_dupes, SRC_ARC1_TEST, SRC_ARC2_TEST)
    print(f"TEST duplicates: {len(eval_dupes)}, verified equal: {vcount2}")
    if mismatches2:
        print("TEST mismatches found:")
        for name, pa, pb in mismatches2[:10]:
            print(f" - {name}: {pa} != {pb}")
        raise SystemExit(f"Aborting due to {len(mismatches2)} TEST mismatches")

    # Copy unions
    copied_train, skipped_train = copy_union(arc1_train, arc2_train, SRC_ARC1_TRAIN, SRC_ARC2_TRAIN, DEST_ALL_TRAIN)
    print(f"Copied TRAIN: {copied_train} (skipped existing: {skipped_train}) -> {DEST_ALL_TRAIN}")

    copied_test, skipped_test = copy_union(arc1_eval, arc2_eval, SRC_ARC1_TEST, SRC_ARC2_TEST, DEST_ALL_TEST)
    print(f"Copied TEST: {copied_test} (skipped existing: {skipped_test}) -> {DEST_ALL_TEST}")


if __name__ == "__main__":
    main()
