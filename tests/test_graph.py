"""LangGraph wiring tests with the LLM agents stubbed out — verifies the
clarify path, the validator retry loop, graceful failure after max attempts,
and the happy path, all offline."""

import pytest

import src.graph as graph_module
from src.schemas import (
    Constraint,
    DecisionVariable,
    ExplainedSolution,
    OptimizationSpec,
    ProblemIntake,
    RetrievedContext,
)

GOOD_SPEC = OptimizationSpec(
    problem_family="diet",
    objective_sense="minimize",
    objective_expression="2*x",
    variables=[DecisionVariable(name="x")],
    constraints=[Constraint(name="c", expression="x", sense=">=", rhs=3)],
)

BAD_SPEC = GOOD_SPEC.model_copy(
    update={"objective_expression": "2*undeclared_var"}
)


@pytest.fixture(autouse=True)
def stub_agents(monkeypatch):
    monkeypatch.setattr(
        graph_module, "retrieve",
        lambda intake, rag_enabled=True: RetrievedContext(),
    )
    monkeypatch.setattr(
        graph_module, "explain",
        lambda spec, result, context, usage=None: ExplainedSolution(
            solve_result=result, explanation="stub explanation"
        ),
    )
    # Each test builds a fresh graph (module caches the compiled graph).
    monkeypatch.setattr(graph_module, "_compiled", None)


def make_intake(missing=()):
    return ProblemIntake(
        raw_text="q", problem_family="diet", missing_info=list(missing)
    )


def test_happy_path(monkeypatch):
    monkeypatch.setattr(graph_module, "run_intake",
                        lambda text, usage=None: make_intake())
    monkeypatch.setattr(
        graph_module, "build_spec",
        lambda intake, ctx, previous_errors=None, usage=None: GOOD_SPEC,
    )
    state = graph_module.run_pipeline("q")
    assert state["solve_result"].status == "Optimal"
    assert state["solve_result"].objective_value == pytest.approx(6.0)
    assert state["explained"].explanation == "stub explanation"
    assert state["modeling_attempts"] == 1
    assert not state.get("error")


def test_retry_loop_recovers_after_bad_spec(monkeypatch):
    monkeypatch.setattr(graph_module, "run_intake",
                        lambda text, usage=None: make_intake())
    calls = []

    def flaky_build(intake, ctx, previous_errors=None, usage=None):
        calls.append(previous_errors)
        return BAD_SPEC if len(calls) == 1 else GOOD_SPEC

    monkeypatch.setattr(graph_module, "build_spec", flaky_build)
    state = graph_module.run_pipeline("q")
    assert state["modeling_attempts"] == 2
    # The retry got the validator's errors as repair context.
    assert calls[0] is None
    assert calls[1] and any("undeclared_var" in e for e in calls[1])
    assert state["solve_result"].status == "Optimal"


def test_graceful_failure_after_max_attempts(monkeypatch):
    monkeypatch.setattr(graph_module, "run_intake",
                        lambda text, usage=None: make_intake())
    monkeypatch.setattr(
        graph_module, "build_spec",
        lambda intake, ctx, previous_errors=None, usage=None: BAD_SPEC,
    )
    state = graph_module.run_pipeline("q")
    assert state["modeling_attempts"] == 3
    assert "error" in state and "3 attempts" in state["error"]
    assert "solve_result" not in state


def test_interactive_clarify_short_circuits(monkeypatch):
    monkeypatch.setattr(
        graph_module, "run_intake",
        lambda text, usage=None: make_intake(missing=["no demand quantities"]),
    )
    monkeypatch.setattr(
        graph_module, "build_spec",
        lambda *a, **k: pytest.fail("modeling must not run before clarification"),
    )
    state = graph_module.run_pipeline("q", interactive=True)
    assert "no demand quantities" in state["clarification_request"]
    assert "spec" not in state


def test_batch_mode_proceeds_despite_missing_info(monkeypatch):
    monkeypatch.setattr(
        graph_module, "run_intake",
        lambda text, usage=None: make_intake(missing=["no demand quantities"]),
    )
    monkeypatch.setattr(
        graph_module, "build_spec",
        lambda intake, ctx, previous_errors=None, usage=None: GOOD_SPEC,
    )
    state = graph_module.run_pipeline("q", interactive=False)
    assert not state.get("clarification_request")
    assert state["solve_result"].status == "Optimal"
