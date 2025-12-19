"""
Extract solutions from oh-barnacles.ipynb and write each as
barnacles_[NNN]_[ORIGID].py into the corresponding combined/[NNN]_[ORIGID]/.

Also include the nearby commentary/classification from the notebook as a
top-of-file docstring for each script. We capture the contiguous Markdown
cells immediately preceding each solution code cell.

Relies on combined/task_matches.json (from compare_task_sets.py) to map
solution indices NNN to original ARC IDs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# Resolve notebook and destination/mapping directories flexibly:
# - Prefer local 'oh-barnacles.ipynb'; fallback to 'code-golf/oh-barnacles.ipynb'
# - Prefer 'combined' dir; fallback to 'collective' (matches this repo layout)
NB_PATH = Path("oh-barnacles.ipynb")
if not NB_PATH.is_file():
    alt_nb = Path("code-golf/oh-barnacles.ipynb")
    NB_PATH = alt_nb if alt_nb.is_file() else NB_PATH

_combined = Path("combined")
_collective = Path("collective")
DEST_DIR = _combined if _combined.is_dir() else _collective
COMBINED_DIR = DEST_DIR  # backwards-compat name used below
MATCHES_JSON = DEST_DIR / "task_matches.json"


def load_sol_to_orig() -> dict[str, str]:
    if not MATCHES_JSON.is_file():
        raise FileNotFoundError(
            f"Mapping file not found: {MATCHES_JSON}. Run compare_task_sets.py first."
        )
    data = json.loads(MATCHES_JSON.read_text())
    matched = data.get("solutions", {}).get("matched", {})
    out: dict[str, str] = {}
    for sol_name, ids in matched.items():
        if ids:
            out[sol_name] = ids[0]
    return out


def extract_writefile_cells(nb: dict) -> list[tuple[int, str, str]]:
    """Return list of (cell_index, NNN, code) for each writefile cell."""
    res: list[tuple[int, str, str]] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src_lines = cell.get("source", [])
        src = "".join(src_lines)
        m = re.search(r"^%%writefile\s+task(\d{3})\.py\s*$", src, flags=re.M)
        if not m:
            continue
        nnn = m.group(1)
        # Strip the writefile magic line
        code = re.sub(r"^%%writefile\s+task\d{3}\.py\s*\n", "", src, flags=re.M)
        res.append((idx, nnn, code))
    return res


def extract_commented_writefile_cells(nb: dict) -> list[tuple[int, str]]:
    """Return list of (cell_index, NNN) for commented writefile cells (unsolved)."""
    res: list[tuple[int, str]] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        m = re.search(r"^\s*#%%writefile\s+task(\d{3})\.py\s*$", src, flags=re.M)
        if m:
            res.append((idx, m.group(1)))
    return res


def collect_surrounding_markdown(nb: dict, cell_idx: int) -> str:
    """Collect contiguous markdown blocks immediately preceding a code cell.

    Only use the markdown that directly precedes the code cell. This avoids
    accidentally including the next task's header/classification (which often
    appears in following markdown), preventing off-by-one mixups in docstrings.
    """
    cells = nb.get("cells", [])
    # Preceding only
    i = cell_idx - 1
    pre: list[str] = []
    while i >= 0 and cells[i].get("cell_type") == "markdown":
        text = "".join(cells[i].get("source", []))
        pre.append(text)
        i -= 1
    pre.reverse()
    return "\n\n".join(pre).strip()


def extract_tag_lines(md: str) -> list[str]:
    """Extract simple '* tag' lines from markdown text (one per line)."""
    tags: list[str] = []
    for line in md.splitlines():
        m = re.match(r"^\s*\*\s*(.+?)\s*$", line)
        if m:
            tags.append(m.group(1).strip())
    return tags


def find_nearby_tags_for_task(nb: dict, cell_idx: int, nnn: str) -> list[str]:
    """Find concept tags for a task by scanning nearby markdown cells.

    Strategy:
      - Look within a small window of markdown cells around the code cell.
      - Identify a header line matching this task number: '# [NNN] <id>.json' with
        optional '-X' like '# [NNN-R] ...'.
      - Collect subsequent lines starting with '* ' as tags, possibly spanning
        the same cell and immediately following markdown cells until a blank line
        or another header-like line.
    """
    cells = nb.get("cells", [])
    header_re = re.compile(rf"^\s*#\s*\[\s*{re.escape(str(int(nnn)))}(?:-[A-Za-z]+)?\s*\]\s+[0-9a-f]{{8}}\.json\s*$", re.I)
    tag_re = re.compile(r"^\s*\*\s*(.+?)\s*$")

    # Candidate markdown cells within window
    start = max(0, cell_idx - 4)
    end = min(len(cells) - 1, cell_idx + 4)
    md_indices = [i for i in range(start, end + 1) if cells[i].get("cell_type") == "markdown"]

    best_block = None
    best_dist = None
    # Find nearest cell containing matching header
    for i in md_indices:
        text = "".join(cells[i].get("source", []))
        for line in text.splitlines():
            if header_re.match(line.strip()):
                dist = abs(cell_idx - i)
                if best_block is None or dist < best_dist:
                    best_block = i
                    best_dist = dist
                break

    if best_block is None:
        # Fallback: search the whole notebook for the matching header, choose
        # the nearest markdown cell containing it; if still none, use preceding tags.
        md_all = [i for i in range(len(cells)) if cells[i].get("cell_type") == "markdown"]
        for i in md_all:
            text = "".join(cells[i].get("source", []))
            for line in text.splitlines():
                if header_re.match(line.strip()):
                    best_block = i
                    best_dist = abs(cell_idx - i)
                    break
            if best_block is not None:
                break
        if best_block is None:
            return extract_tag_lines(collect_surrounding_markdown(nb, cell_idx))

    # Collect tags from the header cell and following markdown cells, until break
    tags: list[str] = []
    i = best_block
    encountered_header = False
    while i <= end and cells[i].get("cell_type") == "markdown":
        text = "".join(cells[i].get("source", []))
        lines = text.splitlines()
        # On the first block, skip lines before header; on later blocks, read from start
        j = 0
        if not encountered_header:
            for j, line in enumerate(lines):
                if header_re.match(line.strip()):
                    encountered_header = True
                    j += 1
                    break
            else:
                j = len(lines)
        # Collect tags until a blank line or another header
        while j < len(lines):
            line = lines[j].strip()
            if not line:
                break
            if header_re.match(line):
                break
            m = tag_re.match(line)
            if m:
                tags.append(m.group(1).strip())
                j += 1
                continue
            # Stop at first non-tag content
            break
        # Stop if we already collected some tags and current block doesn't continue
        if tags and (j >= len(lines) or not tag_re.match(lines[j].strip()) ):
            break
        i += 1

    # Deduplicate while preserving order
    seen = set()
    uniq: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def extract_classification(md: str, nnn: str) -> tuple[str | None, list[str]]:
    """Extract original ID and tags from a classification markdown line for NNN.

    Expected form like: '# [042] 22233c11.json * tag1 * tag2'.
    Returns (orig_id or None, [tags...]).
    """
    orig_id = None
    tags: list[str] = []
    for line in md.splitlines():
        m = re.match(r"^#\s*\[\s*%s\s*\]\s*([0-9a-f]{8})\.json(.*)$" % re.escape(nnn), line.strip(), flags=re.I)
        if m:
            orig_id = m.group(1)
            tail = m.group(2)
            # split on '*' markers
            parts = [p.strip(" -*\t") for p in tail.split('*') if p.strip(" -*\t")]
            if parts:
                tags.extend(parts)
            break
    return orig_id, tags


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    sol_to_orig = load_sol_to_orig()  # e.g., {'task001.json': '007bbfb7'}
    solved_cells = extract_writefile_cells(nb)
    unsolved_cells = extract_commented_writefile_cells(nb)

    written = 0
    skipped = 0

    # Handle solved tasks first (with real code)
    for cell_idx, nnn, code in solved_cells:
        sol_key = f"task{nnn}.json"
        orig_id = sol_to_orig.get(sol_key)
        if not orig_id:
            skipped += 1
            continue
        dest_dir = COMBINED_DIR / f"{nnn}_{orig_id}"
        if not dest_dir.is_dir():
            # If directories not renamed yet, also try task-prefixed variant
            alt = COMBINED_DIR / f"task_{nnn}_{orig_id}"
            dest_dir = alt if alt.is_dir() else dest_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        md_pre = collect_surrounding_markdown(nb, cell_idx)
        tags = find_nearby_tags_for_task(nb, cell_idx, nnn)
        # Compose header docstring always starting with the authoritative mapping
        # '# [NNN] ORIG.json', then optional tags lines from nearby markdown.
        header_lines = [
            '"""',
            f"# [{int(nnn)}] {orig_id}.json",
        ]
        header_lines.extend([f"* {t}" for t in tags])
        header_lines.append('"""')
        header = "\n".join(header_lines) + "\n\n"

        # Compose file content: docstring + code
        content = header + code
        out_path = dest_dir / f"barnacles_{nnn}_{orig_id}.py"
        out_path.write_text(content)
        written += 1

    # Handle unsolved tasks (commented writefile): write docstring and a stub
    for cell_idx, nnn in unsolved_cells:
        sol_key = f"task{nnn}.json"
        orig_id = sol_to_orig.get(sol_key)
        if not orig_id:
            skipped += 1
            continue
        dest_dir = COMBINED_DIR / f"{nnn}_{orig_id}"
        if not dest_dir.is_dir():
            alt = COMBINED_DIR / f"task_{nnn}_{orig_id}"
            dest_dir = alt if alt.is_dir() else dest_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        md_pre = collect_surrounding_markdown(nb, cell_idx)
        tags = find_nearby_tags_for_task(nb, cell_idx, nnn)
        # Minimal docstring with consistent header and optional tags
        doc = [
            '"""',
            f"# [{int(nnn)}] {orig_id}.json",
        ]
        doc.extend([f"* {t}" for t in tags])
        doc.extend([
            '"""',
            "",
        ])
        stub = (
            "def p(I):\n"
            "    \"\"\"Barnacles notebook did not include a solution for this task.\n"
            "    Expected signature: p(I: list[list[int]]) -> list[list[int]].\n"
            "    \"\"\"\n"
            "    raise NotImplementedError('No Barnacles solution in notebook')\n"
        )
        out_path = dest_dir / f"barnacles_{nnn}_{orig_id}.py"
        if not out_path.exists():
            out_path.write_text("\n".join(doc) + stub)
            written += 1

    print(f"Wrote/updated {written} barnacles files; skipped {skipped} (no mapping).")


if __name__ == "__main__":
    main()
