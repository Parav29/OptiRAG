"""Safe linear-expression parsing for LLM-emitted expression strings.

We never call ``eval`` on model output. Instead the expression is parsed with
Python's ``ast`` module and walked with a whitelist: numeric literals,
declared variable names, ``+ - * /`` and parentheses. The result is a mapping
``{variable_name: coefficient}`` plus a constant term (under the ``""`` key),
which the solver turns into a PuLP affine expression.

Anything nonlinear (variable * variable, division by a variable, ``**``,
function calls, attribute access, subscripts, ...) is rejected with
``ExpressionError`` — the message is fed back to the Modeling Agent on retry.
"""

import ast

CONST = ""  # key used for the constant term of a linear expression


class ExpressionError(ValueError):
    pass


def parse_linear_expression(expression: str, declared: set[str]) -> dict[str, float]:
    """Parse ``expression`` into ``{var_name: coeff, "": constant}``.

    Only variable names in ``declared`` may appear. Raises ExpressionError on
    any syntax the whitelist does not cover.
    """
    if not expression or not expression.strip():
        raise ExpressionError("Expression is empty.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Not valid arithmetic syntax: {exc.msg}") from exc
    terms = _walk(tree.body, declared)
    # Drop zero coefficients but keep the constant term explicit.
    result = {name: coeff for name, coeff in terms.items() if name == CONST or coeff != 0}
    result.setdefault(CONST, 0.0)
    return result


def _walk(node: ast.expr, declared: set[str]) -> dict[str, float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"Literal {node.value!r} is not a number.")
        return {CONST: float(node.value)}

    if isinstance(node, ast.Name):
        if node.id not in declared:
            raise ExpressionError(
                f"Name {node.id!r} is not a declared decision variable."
            )
        return {node.id: 1.0}

    if isinstance(node, ast.UnaryOp):
        inner = _walk(node.operand, declared)
        if isinstance(node.op, ast.USub):
            return {k: -v for k, v in inner.items()}
        if isinstance(node.op, ast.UAdd):
            return inner
        raise ExpressionError("Only unary + and - are allowed.")

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            left = _walk(node.left, declared)
            right = _walk(node.right, declared)
            sign = 1.0 if isinstance(node.op, ast.Add) else -1.0
            merged = dict(left)
            for name, coeff in right.items():
                merged[name] = merged.get(name, 0.0) + sign * coeff
            return merged
        if isinstance(node.op, ast.Mult):
            left = _walk(node.left, declared)
            right = _walk(node.right, declared)
            if _is_constant(left):
                scalar, expr = left[CONST], right
            elif _is_constant(right):
                scalar, expr = right[CONST], left
            else:
                raise ExpressionError(
                    "Nonlinear term: cannot multiply two expressions that both "
                    "contain variables."
                )
            return {k: v * scalar for k, v in expr.items()}
        if isinstance(node.op, ast.Div):
            left = _walk(node.left, declared)
            right = _walk(node.right, declared)
            if not _is_constant(right):
                raise ExpressionError("Cannot divide by an expression containing variables.")
            if right[CONST] == 0:
                raise ExpressionError("Division by zero.")
            return {k: v / right[CONST] for k, v in left.items()}
        raise ExpressionError(
            f"Operator {type(node.op).__name__} is not allowed; use + - * / only."
        )

    raise ExpressionError(
        f"Syntax element {type(node).__name__} is not allowed in a linear expression."
    )


def _is_constant(terms: dict[str, float]) -> bool:
    return all(name == CONST for name in terms)
