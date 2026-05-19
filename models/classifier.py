import ast
from typing import Optional


class ProblemClassifier(ast.NodeVisitor):
    def __init__(self):
        self.seen_elementwise = False
        self.seen_reduction = False
        self.seen_matrix = False

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self.seen_elementwise = True
        elif isinstance(node.op, ast.MatMult):
            self.seen_matrix = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._get_call_name(node.func)
        if name in {"sum", "reduce", "torch.sum", "np.sum", "tensorflow.reduce_sum"}:
            self.seen_reduction = True
        if name in {"matmul", "torch.matmul", "np.matmul", "einsum"}:
            self.seen_matrix = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.generic_visit(node)

    def _get_call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._get_call_name(node.value)}.{node.attr}"
        return ""

    def classify(self) -> str:
        if self.seen_matrix:
            return "matrix_operation"
        if self.seen_reduction:
            return "reduction"
        if self.seen_elementwise:
            return "element_wise"
        return "generic"


def classify_problem(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        normalized = code.lower()
        if "+" in normalized or "-" in normalized:
            return "element_wise"
        if "sum" in normalized or "reduce" in normalized:
            return "reduction"
        if "matmul" in normalized or "@" in normalized:
            return "matrix_operation"
        return "generic"

    classifier = ProblemClassifier()
    classifier.visit(tree)
    return classifier.classify()
