import ast
from typing import Any


def ast_to_dict(node: Any) -> Any:
    if isinstance(node, ast.AST):
        result = {node.__class__.__name__: {}}
        for field, value in ast.iter_fields(node):
            if value is None:
                continue
            result[node.__class__.__name__][field] = ast_to_dict(value)
        return result
    if isinstance(node, list):
        return [ast_to_dict(item) for item in node]
    if isinstance(node, (str, int, float, bool, type(None))):
        return node
    return repr(node)


def parse_code(code: str):
    try:
        tree = ast.parse(code)
        return ast_to_dict(tree)
    except SyntaxError:
        return {"error": "Invalid code"}
