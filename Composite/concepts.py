#!/usr/bin/env python3
"""
Parse ARC concepts from REFERENCE/code-golf/oh-barnacles.ipynb.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "REFERENCE"
    / "code-golf"
    / "oh-barnacles.ipynb"
)

BULLET_RE = re.compile(r"^\s*[*+-]\s+(.*\S)\s*$")
HEADER_RE = re.compile(r"^\s*#\s*\[([^\]]+)\]\s*([0-9a-fA-F]{8})\.json\s*$")
SERIAL_RE = re.compile(r"(\d{1,3})")


def _strip_newline(line: str) -> str:
    return line[:-1] if line.endswith("\n") else line


def _iter_cell_lines(cell: dict) -> Iterable[str]:
    source = cell.get("source", [])
    if isinstance(source, str):
        lines = source.splitlines()
    else:
        lines = [_strip_newline(line) for line in source]
    return lines


def _normalize_serial(serial: str) -> str:
    serial = str(serial)
    match = SERIAL_RE.search(serial)
    if not match:
        raise ValueError(f"Invalid serial value: {serial!r}")
    return match.group(1).zfill(3)


def _normalize_arc_id(arc_id: str) -> str:
    arc_id = str(arc_id).lower()
    if arc_id.endswith(".json"):
        arc_id = arc_id[:-5]
    return arc_id


def parse_concepts(notebook_path: Path = DEFAULT_NOTEBOOK_PATH) -> Dict[Tuple[str, str], List[str]]:
    data = json.loads(Path(notebook_path).read_text())
    concepts_by_key: Dict[Tuple[str, str], List[str]] = {}
    current_key: Tuple[str, str] | None = None
    capture_active = False
    concepts_started = False
    for cell in data.get("cells", []):
        cell_type = cell.get("cell_type")
        for line in _iter_cell_lines(cell):
            match = HEADER_RE.match(line)
            if match:
                serial = _normalize_serial(match.group(1))
                arc_id = _normalize_arc_id(match.group(2))
                key = (serial, arc_id)
                if key in concepts_by_key:
                    raise ValueError(
                        f"Duplicate concepts entry for serial {serial} ARC_ID {arc_id}"
                    )
                concepts_by_key[key] = []
                current_key = key
                capture_active = True
                concepts_started = False
                continue
            if not capture_active or current_key is None:
                continue
            if cell_type not in {"markdown", "raw"}:
                continue
            bullet = BULLET_RE.match(line)
            if bullet:
                concepts_by_key[current_key].append(bullet.group(1))
                concepts_started = True
                continue
            if line.strip() and concepts_started:
                capture_active = False
    missing = sorted(key for key, concepts in concepts_by_key.items() if not concepts)
    if missing:
        example = ", ".join(f"{s} {a}" for s, a in missing[:5])
        raise ValueError(f"Missing concepts for {len(missing)} tasks: {example}")
    return concepts_by_key


def get_concepts(serial: str, arc_id: str, notebook_path: Path = DEFAULT_NOTEBOOK_PATH) -> List[str]:
    mapping = parse_concepts(notebook_path)
    serial = _normalize_serial(serial)
    arc_id = _normalize_arc_id(arc_id)
    key = (serial, arc_id)
    if key in mapping:
        return mapping[key]
    alt_serial = next((s for (s, a) in mapping if a == arc_id), None)
    alt_arc = next((a for (s, a) in mapping if s == serial), None)
    if alt_serial is not None:
        raise KeyError(
            f"Concepts not found for serial {serial} ARC_ID {arc_id}; "
            f"ARC_ID {arc_id} exists under serial {alt_serial}."
        )
    if alt_arc is not None:
        raise KeyError(
            f"Concepts not found for serial {serial} ARC_ID {arc_id}; "
            f"serial {serial} exists under ARC_ID {alt_arc}."
        )
    raise KeyError(f"Concepts not found for serial {serial} ARC_ID {arc_id}.")


def _format_python_list(concepts: List[str]) -> str:
    lines = ["CONCEPTS = ["]
    for concept in concepts:
        lines.append(f'    "{concept}",')
    lines.append("]")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        sys.stderr.write("Usage: concepts.py SERIAL ARC_ID\n")
        return 2
    serial, arc_id = args
    concepts = get_concepts(serial, arc_id)
    sys.stdout.write(_format_python_list(concepts) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
