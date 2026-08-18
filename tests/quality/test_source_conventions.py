"""Enforce package documentation and structural-comment conventions.

These checks are source-quality gates rather than runtime unit tests. They keep
module and public-API documentation complete, normalize long section dividers,
and prevent stale local step numbering from reappearing after code moves. Test
modules are included because their fixtures and helper APIs are maintained as
part of the repository's developer documentation.
"""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "pimqc"
PYTHON_FILES = tuple(
    sorted(
        [
            *PACKAGE_ROOT.rglob("*.py"),
            *PROJECT_ROOT.joinpath("tests").rglob("*.py"),
        ]
    )
)
SEPARATOR_RE = re.compile(r"^(?P<indent>\s*)#\s*(?P<char>[=-])\2{3,}\s*$")
NUMBERED_COMMENT_RE = re.compile(
    r"^\s*#\s*(?:\d+(?:\.\d+)*\.\s+|Step\s+\d+:\s+)"
)
PublicDefinition = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


def _public_definitions(tree: ast.Module) -> Iterator[PublicDefinition]:
    """Yield public top-level definitions and public class methods."""
    definition_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    callable_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, definition_types) and (
            not node.name.startswith("_") or node.name == "__init__"
        ):
            yield node
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, callable_types) and (
                    not member.name.startswith("_") or member.name == "__init__"
                ):
                    yield member


def test_modules_and_public_apis_have_docstrings() -> None:
    """Require documentation for every module and public callable."""
    problems = []
    for source_path in PYTHON_FILES:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        module_docstring = ast.get_docstring(tree)
        if module_docstring is None:
            problems.append(f"Missing module docstring: {source_path}")
        elif (
            source_path.is_relative_to(PACKAGE_ROOT)
            and len(module_docstring) < 100
        ):
            problems.append(f"Incomplete module docstring: {source_path}")
        undocumented = [
            node.name
            for node in _public_definitions(tree)
            if ast.get_docstring(node) is None
        ]
        if undocumented:
            problems.append(f"{source_path}: {undocumented}")
    assert not problems, "\n".join(problems)


def test_section_separators_use_the_shared_width() -> None:
    """Keep equals and hyphen section separators aligned to column 79."""
    problems = []
    for source_path in PYTHON_FILES:
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            match = SEPARATOR_RE.match(line)
            if match is None:
                continue
            indent = match.group("indent")
            char = match.group("char")
            expected = f"{indent}# {char * max(4, 77 - len(indent))}"
            if line != expected:
                problems.append(f"{source_path}:{line_number}")
    assert not problems, "Nonstandard separators:\n" + "\n".join(problems)


def test_local_comments_do_not_use_fragile_step_numbers() -> None:
    """Reserve numbered comments for the public pipeline stage sequence."""
    problems = []
    for source_path in PYTHON_FILES:
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if source_path.name == "pipeline.py" and re.match(
                r"^\s*#\s*Step\s+\d{2}:", line
            ):
                continue
            if NUMBERED_COMMENT_RE.match(line):
                problems.append(f"{source_path}:{line_number}")
    assert not problems, "Fragile numbered comments:\n" + "\n".join(problems)
