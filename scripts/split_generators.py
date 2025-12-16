"""
Split each generate_* function from re-arc/generators.py
into its own standalone script under re-arc/generators/.

Each generated script includes the necessary imports
(`from dsl import *`, `from utils import *`) and the single
generator function definition.
"""

from __future__ import annotations

import ast
from pathlib import Path


def extract_generators(source: str) -> list[tuple[str, str]]:
    """Return list of (func_name, func_source) for top-level generate_* functions."""
    tree = ast.parse(source)
    lines = source.splitlines()
    results: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("generate_"):
            # end_lineno and lineno are 1-based and inclusive
            start = node.lineno - 1
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is None:
                # Fallback if end_lineno is unavailable: find next top-level def/class or EOF
                end_idx = start
                for sibling in tree.body[tree.body.index(node) + 1:]:
                    if isinstance(sibling, (ast.FunctionDef, ast.ClassDef)):
                        end_idx = sibling.lineno - 2  # previous line index
                        break
                else:
                    end_idx = len(lines) - 1
            else:
                end_idx = end_lineno - 1
            # Slice is inclusive of end_idx
            func_src = "\n".join(lines[start:end_idx + 1])
            results.append((node.name, func_src))
    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    source_path = repo_root / "generators.py"
    out_dir = repo_root / "generators"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = source_path.read_text()
    gens = extract_generators(src)

    header = (
        "import sys\n"
        "from pathlib import Path\n\n"
        "# Ensure parent dir (re-arc) is on sys.path so dsl/utils resolve when run directly\n"
        "sys.path.append(str(Path(__file__).resolve().parents[1]))\n\n"
        "from dsl import *\n"
        "from utils import *\n\n"
    )

    for name, body in gens:
        out_path = out_dir / f"{name}.py"
        content = header + body
        out_path.write_text(content)

    print(f"Wrote {len(gens)} generator scripts to {out_dir}")


if __name__ == "__main__":
    main()
