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


NB_PATH = Path("oh-barnacles.ipynb")
COMBINED_DIR = Path("combined")
MATCHES_JSON = COMBINED_DIR / "task_matches.json"


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
    """Collect contiguous markdown blocks immediately preceding and following a code cell."""
    cells = nb.get("cells", [])
    # Preceding
    i = cell_idx - 1
    pre: list[str] = []
    while i >= 0 and cells[i].get("cell_type") == "markdown":
        text = "".join(cells[i].get("source", []))
        pre.append(text)
        i -= 1
    pre.reverse()
    # Following
    j = cell_idx + 1
    post: list[str] = []
    while j < len(cells) and cells[j].get("cell_type") == "markdown":
        text = "".join(cells[j].get("source", []))
        post.append(text)
        j += 1
    doc_parts = []
    if pre:
        doc_parts.append("\n\n".join(pre).strip())
    if post:
        doc_parts.append("\n\n".join(post).strip())
    return "\n\n".join(doc_parts)


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

        commentary = collect_surrounding_markdown(nb, cell_idx)
        header = (f'"""\n{commentary}\n"""\n\n' if commentary else '')

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

        commentary = collect_surrounding_markdown(nb, cell_idx)
        cid, tags = extract_classification(commentary, nnn)
        tag_lines = "\n".join(f"- {t}" for t in tags) if tags else "(none)"
        title = f"Task {nnn} — {orig_id}"
        doc = [
            '"""',
            title,
            "",
            "Classification",
            tag_lines,
            "",
            "Notebook Commentary",
            commentary.strip() if commentary else "(none)",
            "",
            "No solution code present in oh-barnacles.ipynb for this task.",
            '"""',
            "",
        ]
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
