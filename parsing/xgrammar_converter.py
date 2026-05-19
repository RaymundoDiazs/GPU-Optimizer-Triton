from typing import Any


def _render_ast(node: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(node, dict):
        name, fields = next(iter(node.items()))
        lines = [f"{prefix}{name}"]
        for field_name, field_value in fields.items():
            if isinstance(field_value, (dict, list)):
                lines.append(f"{prefix}  {field_name}:")
                lines.append(_render_ast(field_value, indent + 2))
            else:
                lines.append(f"{prefix}  {field_name}: {field_value}")
        return "\n".join(lines)
    if isinstance(node, list):
        lines = []
        for item in node:
            lines.append(_render_ast(item, indent))
        return "\n".join(lines)
    return f"{prefix}{node}"


def convert_to_xgrammar(ast_representation: Any) -> str:
    if not isinstance(ast_representation, dict):
        return "XGRAMMAR<invalid>"

    grammar = _render_ast(ast_representation)
    return "XGRAMMAR\n" + grammar
