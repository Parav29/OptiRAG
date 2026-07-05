# BUILD PROMPT FOR CLAUDE CODE — "OptiAgent"
### Natural-Language → Optimization-Model Multi-Agent System with RAG Grounding and an Evaluation Harness

Paste this entire document as your first message to Claude Code (or save it as `CLAUDE.md` in an empty project folder and open Claude Code there). It is written as a complete spec — architecture, contracts, file layout, build phases, and acceptance criteria — so Claude Code can execute it with minimal back-and-forth.

---

## 0. One-line project description

Build **OptiAgent**: a multi-agent LLM system that takes a natural-language description of a planning/optimization problem, retrieves relevant formulation knowledge via RAG (hybrid dense+BM25 retrieval with reranking), constructs a validated structured optimization spec, solves it with a real solver (PuLP), and returns the solution with a generated natural-language explanation. The project must ship with a **quantitative evaluation harness** that measures formulation accuracy, solve correctness, and hallucination rate — including an ablation comparing performance **with vs. without the RAG layer**, since that comparison is the headline result.

---

## 1. Scope: the three problem domains to support

Keep the *optimization* side deliberately narrow so the *agent/RAG/eval* side (the actual point of this project) gets full attention. Support exactly these three classical OR problem families:

1. **Diet / blending problem** (LP) — minimize cost subject to nutrient/ingredient constraints.
2. **Transportation / assignment problem** (LP or MILP) — minimize shipping cost from supply nodes to demand nodes subject to capacity/demand constraints.
3. **Fixed-charge facility location** (MILP) — choose which facilities to open (binary) and how to serve demand from them, minimizing fixed + variable cost.

Do not add more domains until all three work end-to-end with a passing eval suite. A stretch domain (single-vehicle routing or a small multi-echelon inventory problem, tying back to the user's prior IEOR coursework) may be added only after Phase 6 is complete and accepted.

---

## 2. High-level architecture

```
                         ┌─────────────────────────────────────────┐
   User NL problem       │              LangGraph Orchestrator       │
   ──────────────────►   │                                           │
                         │   ┌────────────┐                          │
                         │   │  Intake /  │  parses problem type,    │
                         │   │  Clarifier │  extracts entities,      │
                         │   │   Agent    │  flags missing info      │
                         │   └─────┬──────┘                          │
                         │         │ (loops back to user if           │
                         │         │  critical info missing)          │
                         │         ▼                                 │
                         │   ┌────────────┐      ┌──────────────┐    │
                         │   │  Retriever │─────►│  Knowledge   │    │
                         │   │  (Hybrid   │      │  Base (KB)   │    │
                         │   │  RAG)      │◄─────│  Qdrant +    │    │
                         │   └─────┬──────┘      │  BM25 index  │    │
                         │         │             └──────────────┘    │
                         │         ▼                                 │
                         │   ┌────────────┐                          │
                         │   │  Modeling  │  emits structured spec   │
                         │   │   Agent    │  (Pydantic, tool-call)   │
                         │   └─────┬──────┘                          │
                         │         ▼                                 │
                         │   ┌────────────┐   fail → retry loop      │
                         │   │  Validator │───────┐ (max 3 tries)    │
                         │   └─────┬──────┘       │                  │
                         │         │ pass         │                  │
                         │         ▼              │                  │
                         │   ┌────────────┐        │                  │
                         │   │   Solver   │◄───────┘                  │
                         │   │   Tool     │  builds PuLP model,       │
                         │   │  (PuLP)    │  solves, returns result  │
                         │   └─────┬──────┘                          │
                         │         ▼                                 │
                         │   ┌────────────┐                          │
                         │   │ Explainer  │  NL rationale + binding  │
                         │   │   Agent    │  constraints commentary  │
                         │   └─────┬──────┘                          │
                         └─────────┼─────────────────────────────────┘
                                   ▼
                     Solution + explanation returned to user
                     (via FastAPI JSON, and a minimal chat UI)

   Offline, separate from the live path:
   ┌───────────────────────────────────────────────────────┐
   │  Evaluation Harness — runs the full pipeline against  │
   │  25–30 benchmark problems with known ground-truth     │
   │  optima; runs it twice per problem (RAG ON / RAG OFF) │
   │  to produce the ablation numbers.                      │
   └───────────────────────────────────────────────────────┘
```

---

## 3. Tech stack (pin these; do not substitute without flagging it to the user first)

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | **LangGraph** | Explicit graph with conditional edges for the retry/clarify loops |
| LLM | **Anthropic API (Claude)**, via `anthropic` Python SDK, tool use for structured output | Make the model name a config value, not hardcoded |
| Structured output | **Pydantic v2** models + Anthropic tool-calling JSON schemas | Every agent boundary has a typed contract — see §4 |
| Vector DB | **Qdrant** (via `docker-compose`, local, no cloud dependency) | Simple to self-host; avoids API keys for a demo |
| Embeddings | **`sentence-transformers`**, model `BAAI/bge-small-en-v1.5` (or similar small local model) | Free, local, reproducible — no external embedding API dependency |
| Sparse retrieval | **`rank_bm25`** (BM25Okapi) | Combine with dense via weighted score fusion or RRF (Reciprocal Rank Fusion) |
| Reranker | **`sentence-transformers` CrossEncoder**, model `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranks top-K fused candidates before passing to the modeling agent |
| Solver | **PuLP** with the bundled **CBC** backend | No external solver license needed |
| Serving | **FastAPI** | One `/solve` endpoint (full pipeline) and one `/eval` endpoint (triggers eval run) |
| Demo UI | **Streamlit** (single-file, minimal) | Not the focus — keep it under ~150 lines |
| Containerization | **Dockerfile + docker-compose.yml** (app + Qdrant) | Must run with a single `docker-compose up` |
| Testing | **pytest** | Unit tests per agent/tool, plus the eval harness as an integration-style test |
| Config/secrets | `.env` + `python-dotenv`, **never commit API keys** | Provide `.env.example` |

---

## 4. Data contracts (Pydantic schemas) — define these first, in `src/schemas.py`

```python
class ProblemIntake(BaseModel):
    raw_text: str
    problem_family: Literal["diet", "transportation", "facility_location", "unknown"]
    entities: dict            # extracted numbers/names, family-specific
    missing_info: list[str]   # empty if nothing critical is missing

class RetrievedContext(BaseModel):
    chunks: list[str]
    scores: list[float]
    source_ids: list[str]

class DecisionVariable(BaseModel):
    name: str
    var_type: Literal["continuous", "binary", "integer"]
    lower_bound: float | None
    upper_bound: float | None

class Constraint(BaseModel):
    name: str
    expression: str      # human-readable, e.g. "sum(x_i for i in facilities) <= budget"
    sense: Literal["<=", ">=", "=="]
    rhs: float

class OptimizationSpec(BaseModel):
    problem_family: str
    objective_sense: Literal["minimize", "maximize"]
    objective_expression: str
    variables: list[DecisionVariable]
    constraints: list[Constraint]
    raw_llm_notes: str        # the model's own reasoning trace, kept for audit/eval

class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str]

class SolveResult(BaseModel):
    status: str               # e.g. "Optimal", "Infeasible", "Unbounded"
    objective_value: float | None
    variable_values: dict[str, float]
    binding_constraints: list[str]

class ExplainedSolution(BaseModel):
    solve_result: SolveResult
    explanation: str
```

Every agent reads/writes only these types. This is what makes the eval harness possible — you can score each stage independently.

---

## 5. Agent-by-agent build spec

### 5.1 Intake / Clarifier Agent
- Input: raw user text.
- Output: `ProblemIntake`.
- Classifies problem family using the LLM with a small few-shot prompt (do NOT hardcode keyword matching only — the point is LLM-based classification, though a keyword fallback for the offline eval harness is fine as a safety net).
- If `missing_info` is non-empty AND the pipeline is running in interactive mode, the orchestrator loops back and asks the user; in eval/batch mode, missing info is instead logged and the agent proceeds with reasonable defaults (documented in the eval report).

### 5.2 Retriever (Hybrid RAG)
- Input: `ProblemIntake`.
- Output: `RetrievedContext` (top 5 chunks after rerank).
- Pipeline: BM25 top-20 ∪ dense top-20 → Reciprocal Rank Fusion → CrossEncoder rerank → top 5.
- Must support a **`rag_enabled: bool` flag** that, when `False`, returns an empty `RetrievedContext` — this flag is what drives the ablation study in the eval harness. This is not optional; the whole headline result depends on this switch existing cleanly.

### 5.3 Knowledge Base content (`data/kb/`)
- Write **original** markdown notes (not copied from any textbook) — one file per problem family — covering: standard decision-variable setup, standard constraint patterns, a worked toy example, and common modeling pitfalls (e.g., forgetting non-negativity, off-by-one in facility indices).
- Aim for 8–12 chunks per family after splitting (chunk size ~200–300 tokens, no overlap needed given the short docs).
- Include at least one "worked example with full formulation" per family — this is the single highest-value chunk for retrieval.

### 5.4 Modeling Agent
- Input: `ProblemIntake` + `RetrievedContext`.
- Output: `OptimizationSpec` (via Anthropic tool-calling with the Pydantic schema as the tool's input schema — do not parse free text).
- Prompt must explicitly instruct the model to ground variable/constraint choices in the retrieved chunks when available, and to flag in `raw_llm_notes` when it is extrapolating beyond retrieved context.

### 5.5 Validator
- Pure Python, no LLM call. Checks: every variable referenced in constraints/objective is declared; bounds are internally consistent (lower ≤ upper); at least one constraint exists; objective expression references at least one declared variable.
- On failure: returns `ValidationResult(is_valid=False, errors=[...])`, which the orchestrator feeds back to the Modeling Agent as additional context for a retry (max 3 attempts total, then fail gracefully with a clear error to the user/eval log).

### 5.6 Solver Tool
- Pure Python. Translates `OptimizationSpec` into a `pulp.LpProblem` programmatically (you will need a small expression parser/evaluator for `expression` strings — use `sympy` or a constrained `eval` with a whitelisted namespace of the declared variables; **do not use raw `eval` on unsanitized LLM output** — validate that only declared variable names and arithmetic operators appear before evaluating).
- Solves with CBC, returns `SolveResult`. Also computes binding constraints (constraints tight at the optimum) for the Explainer Agent to use.

### 5.7 Explainer Agent
- Input: `OptimizationSpec` + `SolveResult` + `RetrievedContext`.
- Output: `ExplainedSolution`.
- Must explain: what the optimal solution is in plain English, which constraints are binding and why that matters, and (for LP problems) a one-sentence intuition drawn from shadow-price/dual-value if easily available from PuLP.

---

## 6. Evaluation harness — this is the most important deliverable

Location: `eval/`. This must be runnable as `python -m eval.run_eval` and produce `eval/report.md` + `eval/results.json`.

### 6.1 Benchmark set
- Build **25–30 hand-crafted problems**, roughly evenly split across the three families, each with:
  - The natural-language prompt (as a user would type it).
  - A ground-truth `OptimizationSpec` written by hand.
  - A ground-truth optimal objective value, computed by directly solving the hand-written spec with PuLP (this is your independent ground truth — never derive it from the agent's own output).
  - Include 3–4 deliberately ambiguous/underspecified prompts per family to test the Clarifier and the failure-handling path.

### 6.2 Metrics to compute, per problem and aggregated
1. **Solve correctness**: does the agent's final objective value match ground truth within a relative tolerance (e.g. 1%)? Report as a %.
2. **Feasibility correctness**: does the agent's solution status (Optimal/Infeasible) match the ground-truth status?
3. **Formulation hallucination rate**: flag a run as hallucinated if the `OptimizationSpec` references constraints/entities not present in the original NL prompt (check via a separate LLM-as-judge call comparing spec against prompt, with a strict rubric — see 6.4) OR if the Validator required more than 1 retry to pass.
4. **Latency** (wall-clock per problem) and **token usage** (input+output tokens, from the Anthropic API response usage field).
5. **Explanation quality**: LLM-as-judge score (see 6.4).

### 6.3 The ablation (this produces the resume-bullet number)
- Run the **entire benchmark set twice**: once with `rag_enabled=True`, once with `rag_enabled=False`. Everything else identical (same model, same temperature/seed if settable, same benchmark problems).
- Report hallucination rate and solve correctness for both conditions side by side. This before/after comparison is the single most important artifact — it is the direct evidence behind the "cut hallucinations from X% to Y%" claim, so this must be a real, reproducible measurement, not an estimate.

### 6.4 LLM-as-judge for explanation quality
- Use a **separate** Claude call (different, more explicit rubric prompt) to score each `ExplainedSolution.explanation` on a 1–5 scale across: correctness (does it match the actual solve result), clarity, and completeness (does it mention binding constraints).
- Log the judge's rationale alongside the score for auditability — a judge score with no rationale is not trustworthy and should not be presented as-is.
- Report mean judge score per family and overall.

### 6.5 Output format
- `eval/report.md`: a human-readable markdown report — table of per-family metrics, the RAG-ablation comparison table, 2–3 example transcripts (one success, one failure/retry case, one hallucination case if any occurred), and a short "known limitations" section.
- `eval/results.json`: full machine-readable results for every problem × condition, for reproducibility.

---

## 7. File structure to create

```
optiagent/
├── src/
│   ├── schemas.py              # §4 Pydantic contracts
│   ├── agents/
│   │   ├── intake.py
│   │   ├── retriever.py
│   │   ├── modeling.py
│   │   ├── explainer.py
│   ├── validator.py
│   ├── solver.py                # PuLP construction + safe expression evaluation
│   ├── kb_index.py               # builds/loads Qdrant + BM25 indices from data/kb/
│   ├── graph.py                  # LangGraph orchestrator wiring all agents together
│   ├── api.py                     # FastAPI app: /solve, /eval endpoints
├── data/
│   └── kb/
│       ├── diet.md
│       ├── transportation.md
│       └── facility_location.md
├── eval/
│   ├── benchmark_problems.json   # the 25-30 problems + ground truth
│   ├── run_eval.py
│   ├── judge.py                   # LLM-as-judge logic
│   ├── report.md                  # generated
│   └── results.json               # generated
├── ui/
│   └── streamlit_app.py
├── tests/
│   ├── test_schemas.py
│   ├── test_validator.py
│   ├── test_solver.py
│   ├── test_retriever.py
├── docker-compose.yml             # app + qdrant services
├── Dockerfile
├── requirements.txt
├── .env.example
├── README.md                      # architecture diagram, setup steps, sample run, eval summary
└── CLAUDE.md                      # this file, kept in-repo for future sessions
```

---

## 8. Build phases (build and get sign-off in this order; do not skip ahead)

1. **Phase 1 — Schemas & scaffolding.** Create `schemas.py`, repo structure, `requirements.txt`, Docker skeleton. No agent logic yet. Write `tests/test_schemas.py`.
2. **Phase 2 — Solver in isolation.** Implement `solver.py` and the safe expression evaluator. Write 3 hand-built `OptimizationSpec` examples (one per family) and prove `solver.py` solves them correctly against a known answer. This de-risks the hardest engineering piece early.
3. **Phase 3 — KB + Retriever.** Write the KB markdown files, build the Qdrant + BM25 indices, implement hybrid retrieval + reranking, implement the `rag_enabled` toggle. Test retrieval quality manually against 5 sample queries before moving on.
4. **Phase 4 — Modeling Agent + Validator + retry loop.** Wire tool-calling, the validator, and the retry loop. Test against the same 3 hand-built examples from Phase 2 — the agent's generated spec should solve to the same objective value.
5. **Phase 5 — Explainer Agent + LangGraph wiring.** Assemble the full graph (Intake → Retriever → Modeling → Validator → Solver → Explainer), including the clarify-loop and retry-loop conditional edges.
6. **Phase 6 — FastAPI + Streamlit UI + Docker Compose.** Expose `/solve`; verify the whole stack runs via `docker-compose up` end to end on a fresh machine/container.
7. **Phase 7 — Evaluation harness.** Build the 25–30 benchmark problems with independently-verified ground truth, implement `run_eval.py` and `judge.py`, run the full RAG-on/RAG-off ablation, generate `report.md`.
8. **Phase 8 — README + polish.** Write the README with the architecture diagram, a `docker-compose up` quickstart, 2–3 example transcripts, and the headline eval numbers pulled from `eval/report.md`. Run `pytest` clean.

**After each phase, stop and summarize what was built, what was tested, and any deviations from this spec, before proceeding to the next phase.**

---

## 9. Acceptance criteria (do not declare the project "done" until all are true)

- [ ] All three problem families solve correctly on at least 3 example problems each, verified against independently computed ground truth.
- [ ] `rag_enabled=False` measurably changes retrieval (empty context) and the eval harness reports different hallucination/accuracy numbers for RAG-on vs RAG-off — with real numbers, not placeholders.
- [ ] The Validator retry loop is demonstrably exercised at least once in the benchmark run (i.e., at least one benchmark problem triggers a retry) — if none do, add a harder benchmark problem until one does, so the retry path isn't dead code.
- [ ] `eval/report.md` exists with the full metrics table and the RAG ablation table, generated from an actual run, not fabricated.
- [ ] `docker-compose up` brings up the full stack (app + Qdrant) from a clean checkout with no manual steps beyond copying `.env.example` to `.env` and filling in the Anthropic API key.
- [ ] `pytest` passes with no failing tests.
- [ ] README contains an architecture diagram (ASCII is fine, reuse/adapt the one in §2), a quickstart, and the headline eval numbers.
- [ ] No API keys or secrets committed anywhere in the repo; `.env` is gitignored.

---

## 10. Explicit non-goals (do not build these; flag if tempted to scope-creep)

- No cloud deployment (Kubernetes, AWS, etc.) — Docker Compose locally is sufficient.
- No user authentication/multi-tenancy.
- No support for nonlinear or stochastic optimization — LP/MILP only.
- No fine-tuning of any model.
- No frontend polish beyond a functional Streamlit page — this is not a design showcase.

---

## 11. First message to send

Start by proposing the Phase 1 file scaffolding and the exact contents of `schemas.py`, then stop and wait for confirmation before writing any agent logic.
