"""
Split each verify_* function from re-arc/verifiers.py
into its own standalone script under re-arc/verifiers/.

Each generated script includes the necessary imports
(`from dsl import *`) and the single verifier function.
"""

from __future__ import annotations

import ast
from pathlib import Path


def extract_verifiers(source: str) -> list[tuple[str, str]]:
    """Return list of (func_name, func_source) for top-level verify_* functions."""
    tree = ast.parse(source)
    lines = source.splitlines()
    results: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("verify_"):
            start = node.lineno - 1
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is None:
                # Fallback: search next top-level def/class or EOF
                end_idx = start
                for sibling in tree.body[tree.body.index(node) + 1:]:
                    if isinstance(sibling, (ast.FunctionDef, ast.ClassDef)):
                        end_idx = sibling.lineno - 2
                        break
                else:
                    end_idx = len(lines) - 1
            else:
                end_idx = end_lineno - 1
            func_src = "\n".join(lines[start:end_idx + 1])
            results.append((node.name, func_src))
    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    source_path = repo_root / "verifiers.py"
    out_dir = repo_root / "verifiers"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = source_path.read_text()
    funs = extract_verifiers(src)

    header = (
        "import sys\n"
        "from pathlib import Path\n\n"
        "# Ensure parent dir (re-arc) is on sys.path so dsl resolves when run directly\n"
        "sys.path.append(str(Path(__file__).resolve().parents[1]))\n\n"
        "from dsl import *\n\n"
    )

    for name, body in funs:
        out_path = out_dir / f"{name}.py"
        out_path.write_text(header + body)

    print(f"Wrote {len(funs)} verifier scripts to {out_dir}")


if __name__ == "__main__":
    main()

