#!/usr/bin/env python3
"""
Build SERIAL <-> ARC_ID maps from ARC-1-TRAIN.

SERIAL is 1-indexed and zero-padded to 3 digits based on alphabetical ARC_IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_TRAIN_DIR = (
    Path(__file__).resolve().parents[1] / "DATA" / "ARC-1-TRAIN"
)


def list_arc_ids(train_dir: Path = DEFAULT_TRAIN_DIR) -> List[str]:
    return sorted(p.stem for p in train_dir.glob("*.json"))


def build_serial_map(train_dir: Path = DEFAULT_TRAIN_DIR) -> Dict[str, str]:
    arc_ids = list_arc_ids(train_dir)
    return {f"{i + 1:03d}": arc_id for i, arc_id in enumerate(arc_ids)}


def build_reverse_map(train_dir: Path = DEFAULT_TRAIN_DIR) -> Dict[str, str]:
    serial_to_arc = build_serial_map(train_dir)
    return {arc_id: serial for serial, arc_id in serial_to_arc.items()}


def build_pair_maps(train_dir: Path = DEFAULT_TRAIN_DIR) -> Tuple[Dict[str, str], Dict[str, str]]:
    serial_to_arc = build_serial_map(train_dir)
    return serial_to_arc, {arc_id: serial for serial, arc_id in serial_to_arc.items()}


def main() -> None:
    serial_to_arc = build_serial_map()
    arc_to_serial = {arc_id: serial for serial, arc_id in serial_to_arc.items()}
    payload = {
        "count": len(serial_to_arc),
        "serial_to_arc": serial_to_arc,
        "arc_to_serial": arc_to_serial,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
