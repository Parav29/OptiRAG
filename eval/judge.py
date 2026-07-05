"""LLM-as-judge scoring, separate from the pipeline's own model calls.

Two judges:
  * hallucination judge — strict rubric comparing the generated
    OptimizationSpec against the ORIGINAL user prompt.
  * explanation judge — 1-5 scores for correctness / clarity / completeness.

Every verdict carries a rationale; a score without a rationale is not
trustworthy and is not reported.
"""

from pydantic import BaseModel, Field

from src import config
from src.llm import Usage, structured_call
from src.schemas import ExplainedSolution, OptimizationSpec


class HallucinationVerdict(BaseModel):
    hallucinated: bool
    invented_items: list[str] = Field(default_factory=list)
    rationale: str


class ExplanationScore(BaseModel):
    correctness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    rationale: str


_HALLUCINATION_SYSTEM = """You are a strict auditor of optimization model
formulations. You compare a generated model spec against the ORIGINAL problem
statement and decide whether the spec invents things that are not there.

Mark hallucinated = true ONLY if the spec contains:
- a constraint with data (coefficients/rhs) that appears nowhere in the
  problem statement and is not a universally standard structural device, or
- decision variables for entities (foods, routes, facilities, customers) the
  problem never mentions, or
- objective cost coefficients contradicting or absent from the statement.

The following are NOT hallucinations (do not penalize):
- structural constraints implied by the problem type: non-negativity, batch
  totals, big-M linking rows using stated capacities or total stated demand,
  assignment row/column-sum-to-one conventions;
- unit conversions or percentages of stated quantities;
- reasonable standard defaults for information the problem explicitly leaves
  open, IF they use no invented numbers beyond the default itself.

List each genuinely invented constraint/entity/number in invented_items and
explain your reasoning concisely in rationale."""


def judge_hallucination(
    prompt: str, spec: OptimizationSpec, usage: Usage | None = None
) -> HallucinationVerdict:
    user = (
        f"ORIGINAL PROBLEM STATEMENT:\n{prompt}\n\n"
        f"GENERATED SPEC:\n{spec.model_dump_json(indent=2)}"
    )
    return structured_call(
        system=_HALLUCINATION_SYSTEM,
        user=user,
        output_model=HallucinationVerdict,
        tool_name="record_hallucination_verdict",
        tool_description="Record whether the spec hallucinates beyond the prompt.",
        usage=usage,
        model=config.JUDGE_MODEL,
    )


_EXPLANATION_SYSTEM = """You grade explanations of optimization results on a
1-5 scale for each criterion (5 = excellent, 1 = unacceptable):

- correctness: every number and claim in the explanation matches the provided
  solve result; misstating the objective value, decisions, or status caps
  this at 2.
- clarity: a non-expert could act on it; jargon-free, well organized.
- completeness: states the optimal decision AND the objective value AND
  discusses the binding constraints (bottlenecks); missing binding-constraint
  discussion caps this at 3.

Provide a concise rationale citing specific phrases; scores without concrete
justification are invalid."""


def judge_explanation(
    explained: ExplainedSolution, usage: Usage | None = None
) -> ExplanationScore:
    user = (
        f"SOLVE RESULT (ground truth for grading):\n"
        f"{explained.solve_result.model_dump_json(indent=2)}\n\n"
        f"EXPLANATION TO GRADE:\n{explained.explanation}"
    )
    return structured_call(
        system=_EXPLANATION_SYSTEM,
        user=user,
        output_model=ExplanationScore,
        tool_name="record_explanation_score",
        tool_description="Record the 1-5 rubric scores for the explanation.",
        usage=usage,
        model=config.JUDGE_MODEL,
    )
