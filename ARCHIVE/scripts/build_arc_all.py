from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent

# Inputs: filename lists
LISTS = ROOT / "lists"
ARC1_TRAIN_LIST = LISTS / "arc-agi-1-train.txt"
ARC2_TRAIN_LIST = LISTS / "arc-agi-2-train.txt"
ARC1_TEST_LIST = LISTS / "arc-agi-1-evaluate.txt"
ARC2_TEST_LIST = LISTS / "arc-agi-2-evaluate.txt"

# Source and destination directories
DATA = ROOT / "DATA"
ARC1_TRAIN_DIR = DATA / "ARC-1-TRAIN"
ARC2_TRAIN_DIR = DATA / "ARC-2-TRAIN"
ARC1_TEST_DIR = DATA / "ARC-1-TEST"
ARC2_TEST_DIR = DATA / "ARC-2-TEST"

ALL_TRAIN_DIR = DATA / "ARC-ALL-TRAIN"
ALL_TEST_DIR = DATA / "ARC-ALL-TEST"


def read_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_grid(g: Any) -> Tuple[Tuple[int, ...], ...]:
    if not isinstance(g, list):
        raise TypeError("Grid must be a list")
    out: List[Tuple[int, ...]] = []
    for row in g:
        if not isinstance(row, list):
            raise TypeError("Grid rows must be lists")
        out.append(tuple(int(v) for v in row))
    return tuple(out)


def ex_key_train(ex: Dict[str, Any]) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
    return (norm_grid(ex["input"]), norm_grid(ex["output"]))


def ex_key_test(ex: Dict[str, Any]) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...] | None]:
    inp = norm_grid(ex["input"]) if "input" in ex else None
    outp = norm_grid(ex["output"]) if "output" in ex else None
    return (inp, outp)


def content_multiset_equal_train(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    if len(a) != len(b):
        return False
    return Counter(ex_key_train(x) for x in a) == Counter(ex_key_train(x) for x in b)


def content_multiset_equal_test(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    if len(a) != len(b):
        return False
    return Counter(ex_key_test(x) for x in a) == Counter(ex_key_test(x) for x in b)


def sort_key_train(ex: Dict[str, Any]) -> Tuple[int, int, Tuple[int, ...], int, int, Tuple[int, ...]]:
    i = ex["input"]
    o = ex["output"]
    return (
        len(i), len(i[0]) if i else 0, tuple(v for row in i for v in row),
        len(o), len(o[0]) if o else 0, tuple(v for row in o for v in row),
    )


def sort_key_test(ex: Dict[str, Any]) -> Tuple[int, int, Tuple[int, ...], int, int, Tuple[int, ...]]:
    i = ex.get("input") or []
    o = ex.get("output") or []
    return (
        len(i), len(i[0]) if i else 0, tuple(v for row in i for v in row),
        len(o), len(o[0]) if o else 0, tuple(v for row in o for v in row),
    )


def canonicalize_task(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only train/test with sorted, normalized lists; ensure integers
    train = obj.get("train", []) or []
    test = obj.get("test", []) or []
    # Normalize numeric types to int
    def norm_ex_train(ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "input": [[int(v) for v in row] for row in ex["input"]],
            "output": [[int(v) for v in row] for row in ex["output"]],
        }
    def norm_ex_test(ex: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"input": [[int(v) for v in row] for row in ex["input"]]}
        if "output" in ex:
            out["output"] = [[int(v) for v in row] for row in ex["output"]]
        return out
    train_n = [norm_ex_train(ex) for ex in train]
    test_n = [norm_ex_test(ex) for ex in test]
    train_sorted = sorted(train_n, key=sort_key_train)
    test_sorted = sorted(test_n, key=sort_key_test)
    # Construct with train first, then test (preserve key order by construction)
    return {
        "train": train_sorted,
        "test": test_sorted,
    }


def _indent(level: int) -> str:
    return "  " * level


def _format_row(row: List[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in row) + "]"


def _format_grid(grid: List[List[int]], level: int) -> str:
    # Render a 2D grid so that each row is on a single line
    lines = []
    lines.append(_indent(level) + "[")
    for i, row in enumerate(grid):
        comma = "," if i < len(grid) - 1 else ""
        lines.append(_indent(level + 1) + _format_row(row) + comma)
    lines.append(_indent(level) + "]")
    return "\n".join(lines)


def _format_example_train(ex: Dict[str, Any], level: int) -> str:
    def kv_grid(key: str, grid: List[List[int]], lvl: int) -> str:
        glines = _format_grid(grid, lvl + 0).splitlines()
        # Replace first line indent '[' with on-the-same-line after key
        first = glines[0].lstrip()  # expects '['
        out = [_indent(lvl) + f'"{key}": ' + first]
        out.extend(glines[1:])
        return "\n".join(out)

    lines = []
    lines.append(_indent(level) + "{")
    lines.append(kv_grid("input", ex["input"], level + 1) + ",")
    lines.append(kv_grid("output", ex["output"], level + 1))
    lines.append(_indent(level) + "}")
    return "\n".join(lines)


def _format_example_test(ex: Dict[str, Any], level: int) -> str:
    def kv_grid(key: str, grid: List[List[int]], lvl: int) -> str:
        glines = _format_grid(grid, lvl + 0).splitlines()
        first = glines[0].lstrip()
        out = [_indent(lvl) + f'"{key}": ' + first]
        out.extend(glines[1:])
        return "\n".join(out)

    lines = []
    lines.append(_indent(level) + "{")
    if "output" in ex:
        lines.append(kv_grid("input", ex["input"], level + 1) + ",")
        lines.append(kv_grid("output", ex["output"], level + 1))
    else:
        lines.append(kv_grid("input", ex["input"], level + 1))
    lines.append(_indent(level) + "}")
    return "\n".join(lines)


def dump_task_pretty(obj: Dict[str, Any]) -> str:
    # Produce JSON with train first, then test; each grid row on a single line
    train = obj.get("train", []) or []
    test = obj.get("test", []) or []
    buf = io.StringIO()
    buf.write("{\n")
    # train
    buf.write(_indent(1) + '"train": [\n')
    for i, ex in enumerate(train):
        buf.write(_format_example_train(ex, 2))
        if i < len(train) - 1:
            buf.write(",")
        buf.write("\n")
    buf.write(_indent(1) + "],\n")
    # test
    buf.write(_indent(1) + '"test": [\n')
    for i, ex in enumerate(test):
        buf.write(_format_example_test(ex, 2))
        if i < len(test) - 1:
            buf.write(",")
        buf.write("\n")
    buf.write(_indent(1) + "]\n")
    buf.write("}\n")
    return buf.getvalue()


def write_task(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dump_task_pretty(obj)
    path.write_text(text, encoding="utf-8")


def build_split(
    names_a: Iterable[str], names_b: Iterable[str], src_a: Path, src_b: Path, dest: Path,
) -> Tuple[int, int, int]:
    """Build a union split into dest.

    Returns (written, duplicates_equiv, duplicates_mismatch)
    """
    names_all = []
    seen = set()
    for n in names_a:
        if n not in seen:
            seen.add(n)
            names_all.append(n)
    for n in names_b:
        if n not in seen:
            seen.add(n)
            names_all.append(n)

    written = 0
    dup_equiv = 0
    dup_mismatch = 0

    for name in names_all:
        pa = src_a / name
        pb = src_b / name
        dst = dest / name
        if pa.exists() and pb.exists():
            # duplicate: check content equivalence ignoring order
            A = load_json(pa)
            B = load_json(pb)
            a_tr = A.get("train", []) or []
            b_tr = B.get("train", []) or []
            a_ts = A.get("test", []) or []
            b_ts = B.get("test", []) or []
            equiv = content_multiset_equal_train(a_tr, b_tr) and content_multiset_equal_test(a_ts, b_ts)
            if equiv:
                dup_equiv += 1
                # Canonicalize one side (either) for deterministic output
                C = canonicalize_task(A)
                write_task(dst, C)
                written += 1
            else:
                dup_mismatch += 1
                # Fallback: prefer A, but still canonicalize and emit
                C = canonicalize_task(A)
                write_task(dst, C)
                written += 1
        else:
            # single-source case
            src = pa if pa.exists() else pb
            C = canonicalize_task(load_json(src))
            write_task(dst, C)
            written += 1

    return written, dup_equiv, dup_mismatch


def main() -> None:
    # Read lists
    arc1_train = read_list(ARC1_TRAIN_LIST)
    arc2_train = read_list(ARC2_TRAIN_LIST)
    arc1_test = read_list(ARC1_TEST_LIST)
    arc2_test = read_list(ARC2_TEST_LIST)

    ALL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    ALL_TEST_DIR.mkdir(parents=True, exist_ok=True)

    w_train, eq_train, mm_train = build_split(arc1_train, arc2_train, ARC1_TRAIN_DIR, ARC2_TRAIN_DIR, ALL_TRAIN_DIR)
    print(f"Wrote TRAIN: {w_train} files -> {ALL_TRAIN_DIR}")
    print(f"  duplicates content-equivalent: {eq_train}, mismatches: {mm_train}")

    w_test, eq_test, mm_test = build_split(arc1_test, arc2_test, ARC1_TEST_DIR, ARC2_TEST_DIR, ALL_TEST_DIR)
    print(f"Wrote TEST: {w_test} files -> {ALL_TEST_DIR}")
    print(f"  duplicates content-equivalent: {eq_test}, mismatches: {mm_test}")


if __name__ == "__main__":
    main()
