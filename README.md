# OptiAgent

Natural-language → optimization-model multi-agent system with RAG grounding
and a quantitative evaluation harness.

Describe a planning problem in plain English ("blend corn and soy into the
cheapest feed with 30% protein…"); OptiAgent classifies it, retrieves
formulation knowledge (hybrid dense + BM25 with reranking), emits a validated
structured optimization spec via an LLM (Google Gemini, OpenAI-compatible),
solves it with PuLP/CBC,
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
                         │   │   Agent    │  (Pydantic JSON output) │
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
cp .env.example .env        # fill in GEMINI_API_KEY
docker-compose up --build
```

- API: `http://localhost:8000` (`POST /solve`, `POST /eval`, `GET /health`)
- UI: `http://localhost:8501`

Local dev without Docker (uses an in-memory Qdrant automatically):

```bash
pip install -r requirements.txt
cp .env.example .env         # fill in GEMINI_API_KEY
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

### Live results so far

The full 27×2 ablation is gated by the Gemini free tier's **20 requests/day
per model** cap (confirmed directly from the API's quota error, not
estimated) — at ~5 calls per pipeline run, that's a handful of problems per
day. A first live pass ran one problem per family end-to-end against the real
Gemini API (see `eval/results.json` / `eval/report.md` for the raw data):

| Problem | Family | Solve correct | Objective (agent vs. ground truth) | Hallucinated | Explanation score |
|---|---|---|---|---|---|
| diet_01 | Diet (LP) | ✅ | 0.6231 vs 0.6231 | No | 5.0 / 5 |
| transport_01 | Transportation (LP) | ✅ | 620.0 vs 620.0 | No | 5.0 / 5 |
| facility_01 | Facility location (MILP) | ✅ | 1300.0 vs 1300.0 | No | 4.67 / 5 |

**3/3 solved exactly to independently-computed ground truth, 0 hallucinations,
mean explanation quality 4.89/5 (LLM-as-judge)** — across all three problem
domains, with zero validator retries needed.

The RAG-on/RAG-off ablation table itself is **not yet populated**: the RAG-off
condition hit the daily quota wall immediately after RAG-on finished (visible
in the log as clean 429s). The harness is resumable and will fill in RAG-off
— and the remaining 24 benchmark problems — automatically across subsequent
daily runs, or in one pass once billing is enabled on the API key. No numbers
are fabricated or extrapolated; this table will only grow with real runs.

## Tests

```bash
pytest        # 48 tests: schemas, validator, safe expression parser,
              # solver vs known optima, hybrid retrieval, LangGraph wiring
```

The three hand-built reference problems solve to independently computed
optima: diet ≈ `$0.6231/kg`, transportation `$620`, facility location `$1300`.

## Design notes

- **LLM provider:** Google Gemini via its OpenAI-compatible endpoint
  (`generativelanguage.googleapis.com/v1beta/openai/`) and the `openai` SDK —
  pipeline and judge both on `gemini-2.5-flash` by default (the build spec in
  `CLAUDE.md` originally pinned Anthropic/Claude; the provider was changed
  deliberately at the user's request — briefly to NVIDIA NIM, then back to
  Gemini, which is the reachable endpoint here). The provider lives entirely
  behind `src/llm.py` — every agent talks to `structured_call` / `text_call`,
  so pointing at any other OpenAI-compatible endpoint is just a `LLM_BASE_URL`
  + key change. Structured output embeds the Pydantic JSON Schema in the
  prompt, requests JSON mode, and defensively extracts + validates the
  returned JSON. Gemini 2.5's "thinking" is disabled (`reasoning_effort=none`)
  so hidden reasoning doesn't consume the token budget and leave the answer
  empty. `LLM_MODEL` / `JUDGE_MODEL` are config values.
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
