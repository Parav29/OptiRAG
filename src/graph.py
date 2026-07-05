"""LangGraph orchestrator: Intake -> (clarify?) -> Retriever -> Modeling ->
Validator -> (retry loop, max 3 modeling attempts) -> Solver -> Explainer.

Conditional edges:
  * after intake: in interactive mode, if critical info is missing we stop and
    return a clarification request instead of guessing; in batch/eval mode we
    log the gaps and proceed with standard defaults (documented in the eval
    report).
  * after validation: pass -> solve; fail -> back to modeling with the error
    list (up to MAX_MODELING_ATTEMPTS total), then graceful failure.
"""

import logging
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from src import config
from src.agents.explainer import explain
from src.agents.intake import run_intake
from src.agents.modeling import build_spec
from src.agents.retriever import retrieve
from src.llm import Usage
from src.schemas import (
    ExplainedSolution,
    OptimizationSpec,
    ProblemIntake,
    RetrievedContext,
    SolveResult,
    ValidationResult,
)
from src.solver import solve_spec
from src.validator import validate_spec

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    raw_text: str
    interactive: bool
    rag_enabled: bool
    usage: Usage

    intake: ProblemIntake
    clarification_request: str | None
    context: RetrievedContext
    spec: OptimizationSpec
    validation: ValidationResult
    modeling_attempts: int
    validation_retries: int  # retries actually consumed (attempts - 1)
    solve_result: SolveResult
    explained: ExplainedSolution
    error: str | None


def _intake_node(state: PipelineState) -> PipelineState:
    intake = run_intake(state["raw_text"], usage=state["usage"])
    update: PipelineState = {"intake": intake, "clarification_request": None}
    if intake.missing_info:
        if state.get("interactive", False):
            update["clarification_request"] = (
                "I need a bit more information before I can model this:\n- "
                + "\n- ".join(intake.missing_info)
            )
        else:
            logger.info(
                "Batch mode: proceeding with defaults despite missing info: %s",
                intake.missing_info,
            )
    return update


def _after_intake(state: PipelineState) -> Literal["clarify", "retrieve"]:
    return "clarify" if state.get("clarification_request") else "retrieve"


def _clarify_node(state: PipelineState) -> PipelineState:
    # Terminal in a single run: the caller surfaces the question to the user
    # and starts a new run with the enriched problem text.
    return {}


def _retrieve_node(state: PipelineState) -> PipelineState:
    context = retrieve(state["intake"], rag_enabled=state.get("rag_enabled", True))
    return {"context": context}


def _modeling_node(state: PipelineState) -> PipelineState:
    previous_errors = None
    validation = state.get("validation")
    if validation is not None and not validation.is_valid:
        previous_errors = validation.errors
    spec = build_spec(
        state["intake"],
        state["context"],
        previous_errors=previous_errors,
        usage=state["usage"],
    )
    return {"spec": spec, "modeling_attempts": state.get("modeling_attempts", 0) + 1}


def _validate_node(state: PipelineState) -> PipelineState:
    validation = validate_spec(state["spec"])
    update: PipelineState = {"validation": validation}
    if not validation.is_valid:
        update["validation_retries"] = state.get("modeling_attempts", 1)
        logger.info("Validation failed (attempt %s): %s",
                    state.get("modeling_attempts"), validation.errors)
    return update


def _after_validation(state: PipelineState) -> Literal["solve", "retry", "fail"]:
    if state["validation"].is_valid:
        return "solve"
    if state.get("modeling_attempts", 0) < config.MAX_MODELING_ATTEMPTS:
        return "retry"
    return "fail"


def _fail_node(state: PipelineState) -> PipelineState:
    errors = state["validation"].errors if "validation" in state else []
    return {
        "error": (
            f"Could not produce a valid optimization spec after "
            f"{config.MAX_MODELING_ATTEMPTS} attempts. Last validation errors: "
            + "; ".join(errors)
        )
    }


def _solve_node(state: PipelineState) -> PipelineState:
    return {"solve_result": solve_spec(state["spec"])}


def _explain_node(state: PipelineState) -> PipelineState:
    explained = explain(
        state["spec"], state["solve_result"], state["context"], usage=state["usage"]
    )
    return {"explained": explained}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("intake", _intake_node)
    graph.add_node("clarify", _clarify_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("modeling", _modeling_node)
    graph.add_node("validate", _validate_node)
    graph.add_node("fail", _fail_node)
    graph.add_node("solve", _solve_node)
    graph.add_node("explain", _explain_node)

    graph.set_entry_point("intake")
    graph.add_conditional_edges(
        "intake", _after_intake, {"clarify": "clarify", "retrieve": "retrieve"}
    )
    graph.add_edge("clarify", END)
    graph.add_edge("retrieve", "modeling")
    graph.add_edge("modeling", "validate")
    graph.add_conditional_edges(
        "validate",
        _after_validation,
        {"solve": "solve", "retry": "modeling", "fail": "fail"},
    )
    graph.add_edge("fail", END)
    graph.add_edge("solve", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


_compiled = None


def run_pipeline(
    raw_text: str, rag_enabled: bool = True, interactive: bool = False
) -> PipelineState:
    """Run the full pipeline for one problem and return the final state."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    initial: PipelineState = {
        "raw_text": raw_text,
        "interactive": interactive,
        "rag_enabled": rag_enabled,
        "usage": Usage(),
        "modeling_attempts": 0,
        "validation_retries": 0,
    }
    return _compiled.invoke(initial)
