from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent
LISTS = ROOT / "lists"


def read_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def bytes_identical(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes()


def json_equal(a: Path, b: Path) -> bool:
    # Parsed JSON equality: dict order-insensitive, list order-sensitive
    return load_json(a) == load_json(b)


def norm_grid(g: Any) -> Tuple[Tuple[int, ...], ...]:
    if not isinstance(g, list):
        raise TypeError("Grid must be a list")
    out: List[Tuple[int, ...]] = []
    for row in g:
        if not isinstance(row, list):
            raise TypeError("Grid rows must be lists")
        out.append(tuple(int(v) for v in row))
    return tuple(out)


def ex_key(ex: Dict[str, Any]) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
    return (norm_grid(ex["input"]), norm_grid(ex["output"]))


def train_seq_equal(a_obj: Dict[str, Any], b_obj: Dict[str, Any]) -> bool:
    a_tr = a_obj.get("train", []) or []
    b_tr = b_obj.get("train", []) or []
    if len(a_tr) != len(b_tr):
        return False
    return all(ex_key(x) == ex_key(y) for x, y in zip(a_tr, b_tr))


def train_multiset_equal(a_obj: Dict[str, Any], b_obj: Dict[str, Any]) -> bool:
    a_tr = a_obj.get("train", []) or []
    b_tr = b_obj.get("train", []) or []
    if len(a_tr) != len(b_tr):
        return False
    ca = Counter(ex_key(x) for x in a_tr)
    cb = Counter(ex_key(x) for x in b_tr)
    return ca == cb


def test_seq_equal(a_obj: Dict[str, Any], b_obj: Dict[str, Any]) -> bool:
    a_ts = a_obj.get("test", []) or []
    b_ts = b_obj.get("test", []) or []
    if len(a_ts) != len(b_ts):
        return False
    for x, y in zip(a_ts, b_ts):
        xi = norm_grid(x["input"]) if "input" in x else None
        yi = norm_grid(y["input"]) if "input" in y else None
        xo = norm_grid(x["output"]) if "output" in x else None
        yo = norm_grid(y["output"]) if "output" in y else None
        if xi != yi or xo != yo:
            return False
    return True


def test_multiset_equal(a_obj: Dict[str, Any], b_obj: Dict[str, Any]) -> bool:
    a_ts = a_obj.get("test", []) or []
    b_ts = b_obj.get("test", []) or []
    if len(a_ts) != len(b_ts):
        return False
    def tkey(ex: Dict[str, Any]):
        return (
            norm_grid(ex["input"]) if "input" in ex else None,
            norm_grid(ex["output"]) if "output" in ex else None,
        )
    return Counter(tkey(x) for x in a_ts) == Counter(tkey(x) for x in b_ts)


def classify(a_path: Path, b_path: Path) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "file_a": str(a_path),
        "file_b": str(b_path),
    }
    res["bytes_identical"] = bytes_identical(a_path, b_path)
    A = load_json(a_path)
    B = load_json(b_path)
    res["json_equal"] = A == B
    # lengths
    atr = A.get("train", []) or []
    btr = B.get("train", []) or []
    ats = A.get("test", []) or []
    bts = B.get("test", []) or []
    res["train_len_a"] = len(atr)
    res["train_len_b"] = len(btr)
    res["test_len_a"] = len(ats)
    res["test_len_b"] = len(bts)
    # content checks
    res["train_seq_equal"] = train_seq_equal(A, B)
    res["train_multiset_equal"] = train_multiset_equal(A, B)
    res["test_seq_equal"] = test_seq_equal(A, B)
    res["test_multiset_equal"] = test_multiset_equal(A, B)

    # classification
    if res["bytes_identical"]:
        res["classification"] = "identical_bytes"
    elif res["json_equal"]:
        res["classification"] = "identical_json"
    elif res["train_seq_equal"] and res["test_seq_equal"]:
        res["classification"] = "content_same_order"
    elif res["train_multiset_equal"] and res["test_seq_equal"]:
        res["classification"] = "content_same_diff_train_order"
    elif res["train_multiset_equal"] and res["test_multiset_equal"]:
        res["classification"] = "content_same_order_insensitive"
    else:
        res["classification"] = "different_content"
    return res


def write_report(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # keep a consistent column order
    cols = [
        "name",
        "file_a",
        "file_b",
        "bytes_identical",
        "json_equal",
        "train_len_a",
        "train_len_b",
        "train_seq_equal",
        "train_multiset_equal",
        "test_len_a",
        "test_len_b",
        "test_seq_equal",
        "test_multiset_equal",
        "classification",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    # Read lists
    arc1_train = read_list(LISTS / "arc-agi-1-train.txt")
    arc2_train = read_list(LISTS / "arc-agi-2-train.txt")
    arc1_eval = read_list(LISTS / "arc-agi-1-evaluate.txt")
    arc2_eval = read_list(LISTS / "arc-agi-2-evaluate.txt")

    # Resolve paths
    d = ROOT / "DATA"
    p1t = d / "ARC-1-TRAIN"
    p2t = d / "ARC-2-TRAIN"
    p1e = d / "ARC-1-TEST"
    p2e = d / "ARC-2-TEST"

    train_dupes = sorted(set(arc1_train) & set(arc2_train))
    test_dupes = sorted(set(arc1_eval) & set(arc2_eval))

    train_rows: List[Dict[str, Any]] = []
    for name in train_dupes:
        row = {"name": name}
        row.update(classify(p1t / name, p2t / name))
        train_rows.append(row)

    test_rows: List[Dict[str, Any]] = []
    for name in test_dupes:
        row = {"name": name}
        row.update(classify(p1e / name, p2e / name))
        test_rows.append(row)

    out_dir = LISTS
    write_report(train_rows, out_dir / "arc-duplicates-train-report.csv")
    write_report(test_rows, out_dir / "arc-duplicates-test-report.csv")

    # Summary
    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        cnt: Dict[str, int] = {}
        for r in rows:
            cnt[r["classification"]] = cnt.get(r["classification"], 0) + 1
        return cnt

    train_summary = summarize(train_rows)
    test_summary = summarize(test_rows)
    summary_lines = []
    summary_lines.append("TRAIN duplicates: {}".format(len(train_rows)))
    for k, v in sorted(train_summary.items()):
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("TEST duplicates: {}".format(len(test_rows)))
    for k, v in sorted(test_summary.items()):
        summary_lines.append(f"  {k}: {v}")
    (out_dir / "arc-duplicates-summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Wrote:")
    print(" -", out_dir / "arc-duplicates-train-report.csv")
    print(" -", out_dir / "arc-duplicates-test-report.csv")
    print(" -", out_dir / "arc-duplicates-summary.txt")


if __name__ == "__main__":
    main()

