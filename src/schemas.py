"""Pydantic data contracts for OptiAgent.

Every agent boundary in the pipeline reads/writes only these types, which is
what makes stage-by-stage evaluation possible.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ProblemIntake(BaseModel):
    raw_text: str
    problem_family: Literal["diet", "transportation", "facility_location", "unknown"]
    entities: dict = Field(default_factory=dict)
    missing_info: list[str] = Field(default_factory=list)


class RetrievedContext(BaseModel):
    chunks: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class DecisionVariable(BaseModel):
    name: str
    var_type: Literal["continuous", "binary", "integer"] = "continuous"
    lower_bound: float | None = 0.0
    upper_bound: float | None = None


class Constraint(BaseModel):
    name: str
    expression: str  # linear expression over declared variables, e.g. "2*x1 + 3*x2"
    sense: Literal["<=", ">=", "=="]
    rhs: float


class OptimizationSpec(BaseModel):
    problem_family: str
    objective_sense: Literal["minimize", "maximize"]
    objective_expression: str
    variables: list[DecisionVariable]
    constraints: list[Constraint]
    raw_llm_notes: str = ""


class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)


class SolveResult(BaseModel):
    status: str  # "Optimal", "Infeasible", "Unbounded", "Not Solved", "Undefined"
    objective_value: float | None = None
    variable_values: dict[str, float] = Field(default_factory=dict)
    binding_constraints: list[str] = Field(default_factory=list)


class ExplainedSolution(BaseModel):
    solve_result: SolveResult
    explanation: str
