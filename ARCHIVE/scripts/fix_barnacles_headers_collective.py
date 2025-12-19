"""
Fix off-by-one headers in barnacles_*.py files under collective/.

For each folder collective/NNN_ORIGID/ that contains barnacles_NNN_ORIGID.py,
replace the first header line inside the file that looks like:
    # [XXX] YYYYYYYY.json
with the correct values derived from the folder name:
    # [NNN] ORIGID.json

This script is idempotent and only updates files where the header differs.
"""

from __future__ import annotations

import re
from pathlib import Path


COLLECTIVE_DIR = Path("collective")


HEADER_RE = re.compile(r"^\s*#\s*\[\d+\]\s+[0-9a-f]{8}\.json\s*$", flags=re.I | re.M)


def main() -> None:
    if not COLLECTIVE_DIR.is_dir():
        print(f"No collective directory at {COLLECTIVE_DIR}")
        return

    updated = 0
    skipped = 0
    for sub in sorted(COLLECTIVE_DIR.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        # Expect format NNN_ORIGID where NNN is 3 digits and ORIGID is 8 hex chars
        parts = name.split("_", 1)
        if len(parts) != 2:
            continue
        nnn, orig = parts[0], parts[1]
        if not (nnn.isdigit() and len(nnn) == 3 and re.fullmatch(r"[0-9a-f]{8}", orig, flags=re.I)):
            continue

        barn_path = sub / f"barnacles_{nnn}_{orig}.py"
        if not barn_path.is_file():
            skipped += 1
            continue

        text = barn_path.read_text(encoding="utf-8")
        correct_line = f"# [{int(nnn)}] {orig}.json"

        if HEADER_RE.search(text):
            new_text, n = HEADER_RE.subn(correct_line, text, count=1)
            if n > 0 and new_text != text:
                barn_path.write_text(new_text, encoding="utf-8")
                updated += 1
            else:
                skipped += 1
        else:
            # No existing header to replace; skip
            skipped += 1

    print(f"Updated {updated} barnacles headers; skipped {skipped}.")


if __name__ == "__main__":
    main()

