"""Explainer Agent: turn (spec, solve result, context) into a plain-English
explanation, including binding-constraint commentary."""

import json

from src.llm import Usage, text_call
from src.schemas import (
    ExplainedSolution,
    OptimizationSpec,
    RetrievedContext,
    SolveResult,
)

_SYSTEM = """You explain optimization results to a non-expert decision maker.
Given the model spec and the solver output, write a concise explanation that:
1. States the optimal decision in plain English (what to buy/ship/open and the
   total cost/profit), rounding numbers sensibly.
2. Names the binding constraints (the ones met with equality at the optimum)
   and explains in one sentence each why hitting that limit matters — these
   are the bottlenecks; relaxing them would improve the objective.
3. If the solve status is not Optimal (e.g. Infeasible/Unbounded), explain the
   likely cause in terms of the stated requirements instead.
Keep it under ~250 words. Do not invent numbers not present in the result."""


def explain(
    spec: OptimizationSpec,
    result: SolveResult,
    context: RetrievedContext,
    usage: Usage | None = None,
) -> ExplainedSolution:
    payload = {
        "problem_family": spec.problem_family,
        "objective_sense": spec.objective_sense,
        "objective_expression": spec.objective_expression,
        "constraints": [c.model_dump() for c in spec.constraints],
        "solve_result": result.model_dump(),
    }
    explanation = text_call(
        system=_SYSTEM,
        user=json.dumps(payload, indent=2),
        usage=usage,
    )
    return ExplainedSolution(solve_result=result, explanation=explanation)
