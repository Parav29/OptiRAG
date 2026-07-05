# OptiAgent

Natural-language → optimization-model multi-agent system with RAG grounding
and a quantitative evaluation harness.

Describe a planning problem in plain English ("blend corn and soy into the
cheapest feed with 30% protein…"); OptiAgent classifies it, retrieves
formulation knowledge (hybrid dense + BM25 with reranking), emits a validated
structured optimization spec via Claude tool-calling, solves it with PuLP/CBC,
and explains the solution — including which constraints are binding.

Supported problem families (deliberately narrow so the agent/RAG/eval side
gets full attention):

1. **Diet / blending** (LP)
2. **Transportation / assignment** (LP or MILP)
3. **Fixed-charge facility location** (MILP)

## Architecture

```
                         ┌─────────────────────────────────────────┐
   User NL problem       │            LangGraph Orchestrator        │
   ──────────────────►   │                                          │
                         │   ┌────────────┐                         │
                         │   │  Intake /  │  classifies family,     │
                         │   │  Clarifier │  extracts entities,     │
                         │   │   Agent    │  flags missing info     │
                         │   └─────┬──────┘                         │
                         │         │ (interactive mode: returns a   │
                         │         │  clarification question)       │
                         │         ▼                                │
                         │   ┌────────────┐      ┌──────────────┐   │
                         │   │  Retriever │─────►│  Knowledge   │   │
                         │   │  (Hybrid   │      │  Base (KB)   │   │
                         │   │  RAG)      │◄─────│  Qdrant +    │   │
                         │   └─────┬──────┘      │  BM25 index  │   │
                         │         │             └──────────────┘   │
                         │         ▼                                │
                         │   ┌────────────┐                         │
                         │   │  Modeling  │  emits structured spec  │
                         │   │   Agent    │  (Pydantic, tool-call)  │
                         │   └─────┬──────┘                         │
                         │         ▼                                │
                         │   ┌────────────┐   fail → retry loop     │
                         │   │  Validator │───────┐ (max 3 tries)   │
                         │   └─────┬──────┘       │                 │
                         │         │ pass         │                 │
                         │         ▼              │                 │
                         │   ┌────────────┐       │                 │
                         │   │   Solver   │◄──────┘                 │
                         │   │ (PuLP/CBC) │  safe expression parse, │
                         │   └─────┬──────┘  binding constraints    │
                         │         ▼                                │
                         │   ┌────────────┐                         │
                         │   │ Explainer  │  NL rationale + binding │
                         │   │   Agent    │  constraints commentary │
                         │   └─────┬──────┘                         │
                         └─────────┼────────────────────────────────┘
                                   ▼
                     FastAPI /solve  +  minimal Streamlit chat UI

   Offline: eval/ runs the full pipeline against 27 benchmark problems with
   independently solved ground truth, twice (RAG ON / RAG OFF), producing the
   headline hallucination/accuracy ablation.
```

## Quickstart

```bash
cp .env.example .env        # fill in ANTHROPIC_API_KEY
docker-compose up --build
```

- API: `http://localhost:8000` (`POST /solve`, `POST /eval`, `GET /health`)
- UI: `http://localhost:8501`

Local dev without Docker (uses an in-memory Qdrant automatically):

```bash
pip install -r requirements.txt
cp .env.example .env         # fill in ANTHROPIC_API_KEY
uvicorn src.api:app --reload # terminal 1
streamlit run ui/streamlit_app.py  # terminal 2
```

### Example request

```bash
curl -s localhost:8000/solve -H 'content-type: application/json' -d '{
  "problem": "Plant A can ship 100 units, plant B 80. Stores 1/2/3 need 60/70/40. Costs: A→1 $4, A→2 $6, A→3 $9, B→1 $5, B→2 $3, B→3 $2. Minimize shipping cost.",
  "rag_enabled": true
}'
```

Returns the generated spec, the CBC solve result (objective `620.0`, binding
constraints), and a plain-English explanation.

## Evaluation harness (the headline deliverable)

```bash
python -m eval.make_benchmark   # regenerate + re-verify ground truths
python -m eval.run_eval         # full run: 27 problems × RAG on/off
```

- 27 hand-crafted problems (9 per family): 18 with independently solved
  ground-truth optima, 3 deliberately infeasible, 6 deliberately
  ambiguous/underspecified (clarifier tests).
- Ground truth comes from solving the **hand-written** specs directly with
  PuLP — never from the agent's own output.
- Metrics per problem × condition: solve correctness (1% rel. tolerance),
  feasibility-status correctness, hallucination rate (strict LLM-as-judge with
  logged rationale, OR >1 validator retry), validator retries, latency, token
  usage, and 1–5 explanation quality (LLM-as-judge, rationale logged).
- Outputs: `eval/report.md` (human-readable, incl. the RAG-on/RAG-off ablation
  table and example transcripts) and `eval/results.json` (full per-run
  records).

> **Status:** the harness is fully implemented; run it with your
> `ANTHROPIC_API_KEY` to generate `eval/report.md` — headline numbers belong
> here once produced by a real run (never fabricate them).

## Tests

```bash
pytest        # 48 tests: schemas, validator, safe expression parser,
              # solver vs known optima, hybrid retrieval, LangGraph wiring
```

The three hand-built reference problems solve to independently computed
optima: diet ≈ `$0.6231/kg`, transportation `$620`, facility location `$1300`.

## Design notes

- **Safe expression handling:** LLM-emitted expressions are parsed with an
  AST whitelist (numbers, declared variable names, `+ - * /`, parentheses) —
  never `eval`. Nonlinear or unknown-name expressions are rejected and the
  error list is fed back to the Modeling Agent for a retry (max 3 attempts).
- **`rag_enabled` ablation switch:** `retrieve(..., rag_enabled=False)`
  returns an empty context; the eval harness flips this flag to produce the
  with/without-RAG comparison.
- **Graceful degradation:** if the embedding/reranker models can't be loaded
  (offline), retrieval degrades to BM25-only with a logged warning; if
  `QDRANT_URL` is unset, an in-memory Qdrant is used.

## Non-goals

No cloud deployment, no auth/multi-tenancy, no nonlinear/stochastic models,
no fine-tuning, no UI polish beyond a functional Streamlit page.
