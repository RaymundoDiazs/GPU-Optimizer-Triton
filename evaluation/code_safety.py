from __future__ import annotations

import ast
from dataclasses import dataclass


ALLOWED_IMPORT_ROOTS = {"math", "torch", "triton", "typing"}
BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exit",
    "exec",
    "input",
    "open",
    "quit",
}
BLOCKED_ATTRIBUTE_ROOTS = {
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    errors: list[str]


def _attribute_root(node: ast.AST) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def validate_generated_code_safety(code: str) -> SafetyResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return SafetyResult(False, [f"syntax_error={exc.msg}"])

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"blocked import: {node.module or '<relative>'}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                errors.append(f"blocked call: {node.func.id}")
            root = _attribute_root(node.func)
            if root in BLOCKED_ATTRIBUTE_ROOTS:
                errors.append(f"blocked call root: {root}")
        elif isinstance(node, ast.Attribute):
            root = _attribute_root(node)
            if root in BLOCKED_ATTRIBUTE_ROOTS:
                errors.append(f"blocked attribute root: {root}")

    return SafetyResult(not errors, sorted(set(errors)))
