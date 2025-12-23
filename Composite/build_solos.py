#!/usr/bin/env python3
"""
Generate standalone ARC "solo" scripts for all ARC-1-TRAIN tasks.
"""

from __future__ import annotations

import argparse
import ast
import warnings
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DATA" / "ARC-1-TRAIN"
NOTEBOOK_PATH = ROOT / "REFERENCE" / "code-golf" / "oh-barnacles.ipynb"
SOLUTIONS_DIR = ROOT / "REFERENCE" / "solutions" / "solutions"
RE_ARC_DIR = ROOT / "REFERENCE" / "re-arc"
OUT_DIR = ROOT / "Composite" / "solos"


BUILTINS = set(dir(__builtins__))
RANDOM_NAMES = {"uniform", "sample", "choice", "randint", "shuffle"}
TYPING_NAMES = {
    "Any",
    "Callable",
    "Container",
    "FrozenSet",
    "Tuple",
    "Union",
    "List",
    "Iterable",
}


@dataclass(frozen=True)
class Definition:
    name: str
    source: str
    deps: Set[str]
    order: int


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_pair_map() -> Dict[str, str]:
    sys.path.insert(0, str(ROOT))
    from Composite.pair import build_serial_map  # type: ignore

    mapping = build_serial_map(DATA_DIR)
    if len(mapping) != 400:
        raise ValueError(f"Expected 400 tasks, found {len(mapping)} in {DATA_DIR}")
    return mapping


def _load_concepts_map() -> Dict[Tuple[str, str], List[str]]:
    sys.path.insert(0, str(ROOT))
    from Composite.concepts import parse_concepts  # type: ignore

    concepts = parse_concepts(NOTEBOOK_PATH)
    if len(concepts) != 400:
        raise ValueError(f"Expected 400 concept entries, found {len(concepts)} in {NOTEBOOK_PATH}")
    return concepts


def _iter_cell_lines(cell: dict) -> Iterable[str]:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source.splitlines()
    return [line.rstrip("\n") for line in source]


def _missing_barnacles(serial: str) -> str:
    return "\n".join(
        [
            "def p(*args, **kwargs):",
            f"    raise NotImplementedError(\"Barnacles solution not available for {serial}\")",
        ]
    )


def _extract_barnacles(serial: str, notebook: dict, strict: bool) -> str:
    marker = f"%%writefile task{serial}.py"
    found: Optional[str] = None
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        lines = list(_iter_cell_lines(cell))
        if not lines:
            continue
        if lines[0].strip() == marker:
            body = "\n".join(lines[1:]).strip("\n")
            if not body:
                raise ValueError(f"Empty code cell for serial {serial} in notebook")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    module = ast.parse(body)
            except SyntaxError as exc:
                raise ValueError(f"Invalid barnacles code for serial {serial}: {exc}") from exc
            has_def_p = any(isinstance(node, ast.FunctionDef) and node.name == "p" for node in module.body)
            if not has_def_p:
                rewritten: List[str] = []
                for node in module.body:
                    if isinstance(node, ast.Assign):
                        target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                        if "p" in target_names:
                            if isinstance(node.value, ast.Lambda):
                                func = ast.FunctionDef(
                                    name="p",
                                    args=node.value.args,
                                    body=[ast.Return(value=node.value.body)],
                                    decorator_list=[],
                                )
                                ast.fix_missing_locations(func)
                                rewritten.append(ast.unparse(func))
                            else:
                                expr = ast.unparse(node.value)
                                rewritten.append("def p(*args, **kwargs):")
                                rewritten.append(f"    return ({expr})(*args, **kwargs)")
                            continue
                    seg = ast.get_source_segment(body, node)
                    rewritten.append(seg if seg is not None else ast.unparse(node))
                body = "\n".join(line for line in rewritten if line is not None)
            if not re.search(r"^def\s+p\b", body, flags=re.M):
                raise ValueError(f"Missing p definition in barnacles code for serial {serial}")
            body = re.sub(r"(?<=\d)(and|or)(?=\w)", r" \1 ", body)
            body = re.sub(r"(?<=\d)(and|or)(?=\s)", r" \1", body)
            if found is not None:
                raise ValueError(f"Duplicate barnacles code cell for serial {serial}")
            found = body
    if found is None:
        if strict:
            raise ValueError(f"Barnacles code cell not found for serial {serial}")
        return _missing_barnacles(serial)
    return found


def _missing_compressed(serial: str) -> str:
    return "\n".join(
        [
            "def q(*args, **kwargs):",
            f"    raise NotImplementedError(\"Compressed solution not available for {serial}\")",
        ]
    )


def _extract_compressed(serial: str, strict: bool) -> str:
    path = SOLUTIONS_DIR / f"task{serial}.py"
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"Compressed solution missing: {path}")
        return _missing_compressed(serial)
    text = path.read_text().rstrip()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            module = ast.parse(text)
    except SyntaxError:
        module = None
    if module is None:
        if "p" in text:
            return text + "\n\n" + "def q(*args, **kwargs):\n    return p(*args, **kwargs)"
        if strict:
            raise ValueError(f"Unable to parse compressed solution for serial {serial} in {path}")
        return _missing_compressed(serial)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "p":
            func = ast.FunctionDef(
                name="q",
                args=node.args,
                body=node.body,
                decorator_list=node.decorator_list,
                returns=node.returns,
                type_comment=node.type_comment,
            )
            ast.fix_missing_locations(func)
            return ast.unparse(func)
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if not any(t.id == "p" for t in targets):
                continue
            if isinstance(node.value, ast.Lambda):
                func = ast.FunctionDef(
                    name="q",
                    args=node.value.args,
                    body=[ast.Return(value=node.value.body)],
                    decorator_list=[],
                )
                ast.fix_missing_locations(func)
                return ast.unparse(func)
            expr = ast.unparse(node.value)
            return "\n".join(
                [
                    "def q(*args, **kwargs):",
                    f"    return ({expr})(*args, **kwargs)",
                ]
            )
    if "p" in text:
        return text + "\n\n" + "def q(*args, **kwargs):\n    return p(*args, **kwargs)"
    if strict:
        raise ValueError(f"No compressed solution for serial {serial} in {path}")
    return _missing_compressed(serial)


def _collect_names(node: ast.AST) -> Tuple[Set[str], Set[str]]:
    loads: Set[str] = set()
    stores: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name) -> None:
            if isinstance(n.ctx, ast.Load):
                loads.add(n.id)
            elif isinstance(n.ctx, ast.Store):
                stores.add(n.id)
            self.generic_visit(n)

    Visitor().visit(node)
    return loads, stores


def _defs_from_module(path: Path, order_offset: int) -> Dict[str, Definition]:
    src = path.read_text()
    module = ast.parse(src)
    defs: Dict[str, Definition] = {}
    order = order_offset
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            loads, stores = _collect_names(node)
            arg_names = {arg.arg for arg in node.args.args}
            arg_names |= {arg.arg for arg in node.args.posonlyargs}
            arg_names |= {arg.arg for arg in node.args.kwonlyargs}
            if node.args.vararg:
                arg_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                arg_names.add(node.args.kwarg.arg)
            deps = loads - stores - arg_names - BUILTINS
            source = ast.get_source_segment(src, node)
            if source is None:
                raise ValueError(f"Failed to extract function {node.name} from {path}")
            defs[node.name] = Definition(node.name, source, deps, order)
            order += 1
        elif isinstance(node, ast.Assign):
            loads, _ = _collect_names(node.value)
            deps = loads - BUILTINS
            source = ast.get_source_segment(src, node)
            if source is None:
                raise ValueError(f"Failed to extract assign from {path}")
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defs[target.id] = Definition(target.id, source, deps, order)
                    order += 1
        elif isinstance(node, ast.AnnAssign):
            loads, _ = _collect_names(node.value) if node.value else (set(), set())
            deps = loads - BUILTINS
            source = ast.get_source_segment(src, node)
            if source is None:
                raise ValueError(f"Failed to extract annotated assign from {path}")
            if isinstance(node.target, ast.Name):
                defs[node.target.id] = Definition(node.target.id, source, deps, order)
                order += 1
    return defs


def _resolve_deps(
    initial: Set[str],
    definitions: Dict[str, Definition],
) -> Set[str]:
    needed: Set[str] = set()
    queue = list(initial)
    while queue:
        name = queue.pop()
        if name in needed:
            continue
        if name in definitions:
            needed.add(name)
            queue.extend(definitions[name].deps)
    return needed


def _names_in_function(src: str) -> Set[str]:
    node = ast.parse(src)
    if not node.body or not isinstance(node.body[0], ast.FunctionDef):
        raise ValueError("Expected a single function definition")
    func = node.body[0]
    loads, stores = _collect_names(func)
    arg_names = {arg.arg for arg in func.args.args}
    arg_names |= {arg.arg for arg in func.args.posonlyargs}
    arg_names |= {arg.arg for arg in func.args.kwonlyargs}
    if func.args.vararg:
        arg_names.add(func.args.vararg.arg)
    if func.args.kwarg:
        arg_names.add(func.args.kwarg.arg)
    return loads - stores - arg_names - BUILTINS


def _extract_function_source(path: Path, func_name: str) -> str:
    src = path.read_text()
    module = ast.parse(src)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            seg = ast.get_source_segment(src, node)
            if seg is None:
                raise ValueError(f"Failed to extract {func_name} from {path}")
            return seg
    raise ValueError(f"{func_name} not found in {path}")


def _insert_to_grid(verify_src: str) -> str:
    lines = verify_src.splitlines()
    if any(line.strip() == "I = _to_grid(I)" for line in lines):
        return verify_src
    match = re.match(r"(\s*)def\s+\w+\s*\(", lines[0])
    if not match:
        raise ValueError("Invalid verifier function signature")
    indent = match.group(1) + "    "
    lines.insert(1, f"{indent}I = _to_grid(I)")
    return "\n".join(lines)


def _format_np_array(name: str, grid: List[List[int]]) -> str:
    lines = [f"{name} = np.array(["]
    for row in grid:
        lines.append(f"    {row},")
    lines.append("], dtype=int)")
    return "\n".join(lines)


def _format_concepts(concepts: List[str]) -> str:
    lines = ["CONCEPTS = ["]
    for concept in concepts:
        lines.append(f'    "{concept}",')
    lines.append("]")
    return "\n".join(lines)


def _format_to_grid() -> str:
    return "\n".join(
        [
            "def _to_grid(",
            "    grid: Any",
            ") -> Grid:",
            "    if isinstance(grid, np.ndarray):",
            "        return tuple(tuple(int(x) for x in row.tolist()) for row in grid)",
            "    if isinstance(grid, list):",
            "        return tuple(tuple(int(x) for x in row) for row in grid)",
            "    return grid",
        ]
    )


def _write_task_file(
    serial: str,
    arc_id: str,
    concepts: List[str],
    barnacles_src: str,
    compressed_src: str,
    generator_src: str,
    verifier_src: str,
    generator_defs: List[str],
    verifier_defs: List[str],
    random_imports: List[str],
    data: dict,
    out_dir: Path,
) -> None:
    lines: List[str] = []
    lines.append("# --- Imports ---")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("# --- Metadata ---")
    lines.append(f'ARC_ID = "{arc_id}"')
    lines.append(f'SERIAL = "{serial}"')
    lines.append(f'URL    = "https://arcprize.org/play?task={arc_id}"')
    lines.append("")
    lines.append("# --- Code Golf Concepts ---")
    lines.append(_format_concepts(concepts))
    lines.append("")
    lines.append("# --- Example Grids ---")
    for idx, example in enumerate(data["train"], start=1):
        lines.append(_format_np_array(f"E{idx}_IN", example["input"]))
        lines.append("")
        lines.append(_format_np_array(f"E{idx}_OUT", example["output"]))
        lines.append("")
    lines.append("# --- Test ---")
    tests = data["test"]
    if not tests:
        raise ValueError(f"No test cases for {arc_id}")
    for idx, example in enumerate(tests, start=1):
        name = "T" if idx == 1 else f"T{idx}"
        lines.append(_format_np_array(f"{name}_IN", example["input"]))
        lines.append("")
        lines.append(_format_np_array(f"{name}_OUT", example["output"]))
        lines.append("")
    lines.append("# --- Code Golf Solution (Barnacles) ---")
    lines.append(barnacles_src.rstrip())
    lines.append("")
    lines.append("")
    lines.append("# --- Code Golf Solution (Compressed) ---")
    lines.append(compressed_src.rstrip())
    lines.append("")
    lines.append("")
    lines.append("# --- RE-ARC Generator ---")
    lines.append("from typing import Any, Callable, Container, FrozenSet, Tuple, Union, List, Iterable")
    if random_imports:
        lines.append(f"from random import {', '.join(random_imports)}")
    lines.append("")
    for block in generator_defs:
        lines.append(block)
        lines.append("")
    lines.append(generator_src.rstrip())
    lines.append("")
    lines.append("")
    lines.append("# --- RE-ARC Verifier ---")
    for block in verifier_defs:
        lines.append(block)
        lines.append("")
    lines.append(_format_to_grid())
    lines.append("")
    lines.append(_insert_to_grid(verifier_src).rstrip())
    lines.append("")
    lines.append("")
    lines.append("if __name__ == \"__main__\":")
    lines.append("    examples = [")
    for idx in range(1, len(data["train"]) + 1):
        lines.append(f"        (\"E{idx}\", E{idx}_IN, E{idx}_OUT),")
    if len(tests) == 1:
        lines.append("        (\"T\", T_IN, T_OUT),")
    else:
        for idx in range(1, len(tests) + 1):
            name = "T" if idx == 1 else f"T{idx}"
            lines.append(f"        (\"{name}\", {name}_IN, {name}_OUT),")
    lines.append("    ]")
    lines.append("    for name, inp, expected in examples:")
    lines.append(f"        pred = verify_{arc_id}(inp)")
    lines.append("        assert pred == _to_grid(expected), f\"{name} failed\"")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{serial}_{arc_id}_solo.py"
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ARC solo scripts.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--serial", nargs="*", help="Only generate these serials")
    parser.add_argument("--strict", action="store_true", help="Fail if any source is missing")
    args = parser.parse_args()

    mapping = _load_pair_map()
    concepts_map = _load_concepts_map()
    notebook = _read_json(NOTEBOOK_PATH)

    dsl_defs = _defs_from_module(RE_ARC_DIR / "dsl.py", 0)
    utils_defs = _defs_from_module(RE_ARC_DIR / "utils.py", 10000)
    all_defs: Dict[str, Definition] = {}
    for name, definition in dsl_defs.items():
        all_defs[name] = definition
    for name, definition in utils_defs.items():
        if name in all_defs:
            raise ValueError(f"Duplicate definition for {name} in RE-ARC sources")
        all_defs[name] = definition

    gen_src_cache: Dict[str, str] = {}
    ver_src_cache: Dict[str, str] = {}

    out_dir = Path(args.out_dir)
    serial_filter = set(args.serial or [])

    missing_barnacles: List[str] = []
    for serial, arc_id in mapping.items():
        if serial_filter and serial not in serial_filter:
            continue
        data_path = DATA_DIR / f"{arc_id}.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing task data: {data_path}")
        data = _read_json(data_path)
        concepts = concepts_map.get((serial, arc_id))
        if concepts is None:
            raise ValueError(f"Missing concepts for serial {serial} ARC_ID {arc_id}")

        barnacles_src = _extract_barnacles(serial, notebook, args.strict)
        if "NotImplementedError" in barnacles_src:
            missing_barnacles.append(serial)
        compressed_src = _extract_compressed(serial, args.strict)

        gen_name = f"generate_{arc_id}"
        ver_name = f"verify_{arc_id}"
        if arc_id not in gen_src_cache:
            gen_src_cache[arc_id] = _extract_function_source(RE_ARC_DIR / "generators.py", gen_name)
            ver_src_cache[arc_id] = _extract_function_source(RE_ARC_DIR / "verifiers.py", ver_name)
        generator_src = gen_src_cache[arc_id]
        verifier_src = ver_src_cache[arc_id]

        gen_deps = _resolve_deps(_names_in_function(generator_src), all_defs)
        ver_deps = _resolve_deps(_names_in_function(verifier_src) | {"Grid"}, all_defs)

        used_defs: List[Definition] = [
            definition
            for definition in sorted(all_defs.values(), key=lambda d: d.order)
            if definition.name in gen_deps
        ]
        gen_def_sources = [d.source for d in used_defs]

        remaining = ver_deps - set(d.name for d in used_defs)
        ver_def_sources = [
            d.source
            for d in sorted(all_defs.values(), key=lambda d: d.order)
            if d.name in remaining
        ]

        gen_def_names = {d.name for d in used_defs}
        ver_def_names = {name for name in remaining}
        external_names: Set[str] = set()
        external_names |= _names_in_function(generator_src)
        external_names |= _names_in_function(verifier_src)
        for definition in used_defs:
            external_names |= definition.deps
        for definition in (all_defs[name] for name in ver_def_names):
            external_names |= definition.deps
        external_names -= set(all_defs.keys())
        random_imports = sorted(name for name in external_names if name in RANDOM_NAMES)

        _write_task_file(
            serial=serial,
            arc_id=arc_id,
            concepts=concepts,
            barnacles_src=barnacles_src,
            compressed_src=compressed_src,
            generator_src=generator_src,
            verifier_src=verifier_src,
            generator_defs=gen_def_sources,
            verifier_defs=ver_def_sources,
            random_imports=random_imports,
            data=data,
            out_dir=out_dir,
        )

    if missing_barnacles:
        msg = f"Barnacles solutions missing for {len(missing_barnacles)} serials"
        if args.strict:
            raise ValueError(msg)
        sys.stderr.write(msg + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
