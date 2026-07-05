import pytest
from pydantic import ValidationError

from src.schemas import (
    Constraint,
    DecisionVariable,
    ExplainedSolution,
    OptimizationSpec,
    ProblemIntake,
    RetrievedContext,
    SolveResult,
)


def test_problem_intake_defaults():
    intake = ProblemIntake(raw_text="hi", problem_family="diet")
    assert intake.entities == {}
    assert intake.missing_info == []


def test_problem_intake_rejects_bad_family():
    with pytest.raises(ValidationError):
        ProblemIntake(raw_text="hi", problem_family="scheduling")


def test_retrieved_context_empty_by_default():
    ctx = RetrievedContext()
    assert ctx.chunks == [] and ctx.scores == [] and ctx.source_ids == []


def test_decision_variable_defaults():
    var = DecisionVariable(name="x")
    assert var.var_type == "continuous"
    assert var.lower_bound == 0.0
    assert var.upper_bound is None


def test_constraint_rejects_bad_sense():
    with pytest.raises(ValidationError):
        Constraint(name="c", expression="x", sense="<", rhs=1.0)


def test_optimization_spec_round_trip():
    spec = OptimizationSpec(
        problem_family="diet",
        objective_sense="minimize",
        objective_expression="2*x + 3*y",
        variables=[DecisionVariable(name="x"), DecisionVariable(name="y")],
        constraints=[Constraint(name="c1", expression="x + y", sense=">=", rhs=1)],
        raw_llm_notes="test",
    )
    restored = OptimizationSpec.model_validate(spec.model_dump())
    assert restored == spec


def test_solve_result_and_explained_solution():
    result = SolveResult(status="Optimal", objective_value=1.5,
                         variable_values={"x": 1.0}, binding_constraints=["c1"])
    explained = ExplainedSolution(solve_result=result, explanation="Buy 1 unit of x.")
    assert explained.solve_result.objective_value == 1.5


def test_tool_schema_generation():
    # The modeling agent embeds this schema in the Gemini JSON-output request.
    schema = OptimizationSpec.model_json_schema()
    assert "properties" in schema
    assert set(schema["required"]) >= {
        "problem_family", "objective_sense", "objective_expression",
        "variables", "constraints",
    }
