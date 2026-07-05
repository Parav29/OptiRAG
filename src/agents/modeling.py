"""Modeling Agent: (ProblemIntake, RetrievedContext) -> OptimizationSpec.

Structured output is obtained via Gemini JSON-schema-constrained output with
the Pydantic schema — no ad-hoc free-text parsing. On validation failure
the orchestrator calls this again with the validator's error list, which is
appended to the prompt as explicit repair instructions.
"""

from src.llm import Usage, structured_call
from src.schemas import OptimizationSpec, ProblemIntake, RetrievedContext

_SYSTEM = """You are an expert operations-research modeler. Convert the user's
natural-language problem into a precise linear/mixed-integer optimization spec.

Hard rules for the spec you emit:
1. Expressions (objective and constraint left-hand sides) must be EXPANDED
   linear arithmetic over the declared variable names using only + - * / and
   parentheses, e.g. "2.5*x_corn + 1.2*x_soy". No sum(), no indexing, no
   comparison operators inside `expression` (the sense/rhs fields carry those).
2. Every variable used in any expression MUST appear in `variables`, with the
   correct var_type ("binary" for open/close decisions, "continuous" for
   quantities) and sensible bounds (non-negative quantities: lower_bound 0).
3. Use short, valid Python identifiers for variable names (e.g. ship_A_1,
   open_dallas, x_beef).
4. Numbers must come from the problem statement or the reference notes. Do NOT
   invent data, constraints, or entities that are not in the problem.
5. If reference notes are provided, ground your variable/constraint patterns
   in them. In `raw_llm_notes`, briefly state which reference patterns you
   used, and EXPLICITLY FLAG any place where you extrapolated beyond the
   provided notes or made an assumption not stated in the problem.
6. If the problem is underspecified, make the most standard textbook
   assumption, state it in `raw_llm_notes`, and proceed."""


def build_spec(
    intake: ProblemIntake,
    context: RetrievedContext,
    previous_errors: list[str] | None = None,
    usage: Usage | None = None,
) -> OptimizationSpec:
    parts = [f"Problem statement:\n{intake.raw_text}"]
    parts.append(f"Detected problem family: {intake.problem_family}")
    if intake.entities:
        parts.append(f"Extracted entities: {intake.entities}")
    if intake.missing_info:
        parts.append(
            "Known missing information (proceed with standard defaults and note "
            f"them): {intake.missing_info}"
        )
    if context.chunks:
        notes = "\n\n---\n\n".join(
            f"[{sid}]\n{chunk}"
            for sid, chunk in zip(context.source_ids, context.chunks)
        )
        parts.append(f"Reference formulation notes (retrieved):\n{notes}")
    else:
        parts.append(
            "No reference notes are available; rely on standard formulations."
        )
    if previous_errors:
        parts.append(
            "Your previous spec FAILED validation with these errors — fix every "
            "one of them in this attempt:\n- " + "\n- ".join(previous_errors)
        )

    return structured_call(
        system=_SYSTEM,
        user="\n\n".join(parts),
        output_model=OptimizationSpec,
        tool_name="emit_optimization_spec",
        tool_description=(
            "Emit the structured optimization spec for the user's problem."
        ),
        usage=usage,
    )
