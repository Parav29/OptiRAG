"""Intake / Clarifier Agent: classify the problem family, extract entities,
flag missing information.

Primary path is an LLM few-shot classification via tool-calling. A keyword
fallback exists purely as an offline safety net (used when the API is
unreachable or for deterministic tests) — it is NOT the primary classifier.
"""

import re

from src.llm import Usage, structured_call
from src.schemas import ProblemIntake

_SYSTEM = """You are the intake agent of an optimization-modeling assistant.
You classify a natural-language planning problem into exactly one family and
extract the key quantitative entities.

Families:
- "diet": choose quantities of foods/ingredients to minimize cost subject to
  nutritional or composition requirements (a.k.a. blending problems).
- "transportation": ship goods from supply nodes to demand nodes minimizing
  shipping cost subject to supply capacities and demand requirements
  (includes assignment problems).
- "facility_location": decide which facilities/warehouses to OPEN (a yes/no
  decision with a fixed opening cost) and how to serve customers from the
  opened ones, minimizing fixed + variable cost.
- "unknown": none of the above.

Few-shot examples:
1. "I need the cheapest mix of corn and soybean meal so the feed has at least
   20% protein and 5% fat." -> diet
2. "Two plants ship widgets to three stores; plant A can make 100 units..."
   -> transportation
3. "We can open warehouses in Dallas or Reno ($50k each to open) and must
   serve 4 regions..." -> facility_location
4. "Schedule nurses across shifts to minimize overtime." -> unknown

Extract entities as a flat dict of the numbers/names that matter (costs,
capacities, demands, requirements). List in missing_info anything CRITICAL
that prevents building a complete model (e.g. "no demand quantities given").
Do NOT list nice-to-haves; an empty list means the problem is solvable as
stated."""


def run_intake(raw_text: str, usage: Usage | None = None) -> ProblemIntake:
    intake = structured_call(
        system=_SYSTEM,
        user=raw_text,
        output_model=ProblemIntake,
        tool_name="record_intake",
        tool_description="Record the classified and parsed problem intake.",
        usage=usage,
    )
    # The LLM sometimes echoes a paraphrase; keep the user's original text.
    intake.raw_text = raw_text
    return intake


_KEYWORDS = {
    "facility_location": [
        r"\bopen(ing)?\b.*\b(facilit|warehouse|plant|depot|site)",
        r"\bfixed[- ](cost|charge)\b",
        r"\bwhich\b.*\b(facilit|warehouse|plant|site)s?\b.*\bopen\b",
    ],
    "transportation": [
        r"\bship(ping|ment)?s?\b",
        r"\bsupply\b.*\bdemand\b",
        r"\btransport(ation)?\b",
        r"\bassign(ment)?\b.*\b(worker|task|job|machine)",
    ],
    "diet": [
        r"\bdiet\b",
        r"\bnutri(ent|tion)",
        r"\bblend(ing)?\b",
        r"\b(protein|calorie|vitamin|fat|fiber)s?\b",
        r"\b(feed|food|ingredient)s?\b.*\bcost\b",
    ],
}


def keyword_fallback_intake(raw_text: str) -> ProblemIntake:
    """Deterministic offline classifier (safety net only)."""
    text = raw_text.lower()
    for family, patterns in _KEYWORDS.items():
        if any(re.search(p, text) for p in patterns):
            return ProblemIntake(
                raw_text=raw_text,
                problem_family=family,  # type: ignore[arg-type]
                entities={},
                missing_info=[],
            )
    return ProblemIntake(raw_text=raw_text, problem_family="unknown")
