"""Pure-Python structural validation of an OptimizationSpec. No LLM calls.

Checks performed:
  * at least one decision variable is declared
  * variable names are unique and are valid identifiers
  * bounds are internally consistent (lower <= upper)
  * at least one constraint exists
  * every expression (objective + constraints) parses as a linear expression
    and references only declared variable names
  * the objective references at least one declared variable
"""

from src.expr import ExpressionError, parse_linear_expression
from src.schemas import OptimizationSpec, ValidationResult


def validate_spec(spec: OptimizationSpec) -> ValidationResult:
    errors: list[str] = []

    if not spec.variables:
        errors.append("Spec declares no decision variables.")

    declared: set[str] = set()
    for var in spec.variables:
        if not var.name.isidentifier():
            errors.append(f"Variable name {var.name!r} is not a valid identifier.")
        if var.name in declared:
            errors.append(f"Variable {var.name!r} is declared more than once.")
        declared.add(var.name)
        if (
            var.lower_bound is not None
            and var.upper_bound is not None
            and var.lower_bound > var.upper_bound
        ):
            errors.append(
                f"Variable {var.name!r} has lower_bound {var.lower_bound} > "
                f"upper_bound {var.upper_bound}."
            )

    if not spec.constraints:
        errors.append("Spec has no constraints; at least one is required.")

    seen_constraint_names: set[str] = set()
    for con in spec.constraints:
        if con.name in seen_constraint_names:
            errors.append(f"Constraint name {con.name!r} is duplicated.")
        seen_constraint_names.add(con.name)
        try:
            terms = parse_linear_expression(con.expression, declared)
        except ExpressionError as exc:
            errors.append(f"Constraint {con.name!r}: {exc}")
            continue
        if not any(name for name in terms if name):
            errors.append(
                f"Constraint {con.name!r} references no declared variables "
                f"(expression: {con.expression!r})."
            )

    try:
        obj_terms = parse_linear_expression(spec.objective_expression, declared)
        if not any(name for name in obj_terms if name):
            errors.append(
                "Objective expression references no declared variables "
                f"(expression: {spec.objective_expression!r})."
            )
    except ExpressionError as exc:
        errors.append(f"Objective expression: {exc}")

    return ValidationResult(is_valid=not errors, errors=errors)
