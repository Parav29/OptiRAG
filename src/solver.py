"""Translate a validated OptimizationSpec into a PuLP model and solve it (CBC).

Pure Python — no LLM involved. Expressions are parsed with the whitelisting
parser in ``src.expr`` (never raw ``eval``) and rebuilt as PuLP affine
expressions over the declared variables. Binding constraints (zero slack at
the optimum) are reported for the Explainer Agent.
"""

import pulp

from src.expr import CONST, parse_linear_expression
from src.schemas import OptimizationSpec, SolveResult

BINDING_TOL = 1e-6

_CATEGORY = {
    "continuous": pulp.LpContinuous,
    "integer": pulp.LpInteger,
    "binary": pulp.LpBinary,
}


def solve_spec(spec: OptimizationSpec, time_limit: int = 60) -> SolveResult:
    sense = pulp.LpMinimize if spec.objective_sense == "minimize" else pulp.LpMaximize
    problem = pulp.LpProblem(spec.problem_family or "optiagent", sense)

    declared = {v.name for v in spec.variables}
    lp_vars: dict[str, pulp.LpVariable] = {}
    for var in spec.variables:
        lp_vars[var.name] = pulp.LpVariable(
            var.name,
            lowBound=var.lower_bound,
            upBound=var.upper_bound,
            cat=_CATEGORY[var.var_type],
        )

    problem += _to_affine(spec.objective_expression, declared, lp_vars), "objective"

    for con in spec.constraints:
        affine = _to_affine(con.expression, declared, lp_vars)
        if con.sense == "<=":
            problem += affine <= con.rhs, con.name
        elif con.sense == ">=":
            problem += affine >= con.rhs, con.name
        else:
            problem += affine == con.rhs, con.name

    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[status_code]

    if status != "Optimal":
        return SolveResult(status=status)

    variable_values = {name: (var.value() or 0.0) for name, var in lp_vars.items()}
    binding = [
        name
        for name, constraint in problem.constraints.items()
        if constraint.slack is not None and abs(constraint.slack) <= BINDING_TOL
    ]
    return SolveResult(
        status=status,
        objective_value=pulp.value(problem.objective),
        variable_values=variable_values,
        binding_constraints=binding,
    )


def _to_affine(expression: str, declared: set[str], lp_vars: dict) -> pulp.LpAffineExpression:
    terms = parse_linear_expression(expression, declared)
    constant = terms.pop(CONST, 0.0)
    return pulp.LpAffineExpression(
        [(lp_vars[name], coeff) for name, coeff in terms.items()], constant=constant
    )
