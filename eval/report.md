# OptiAgent Evaluation Report

Benchmark: 3 problems × 2 conditions (RAG on / RAG off), identical model and settings.

## Headline: RAG ablation

| Metric | RAG ON | RAG OFF |
|---|---|---|
| Solve correctness (obj within 1% of ground truth) | 100.0% | —% |
| Feasibility status correctness | 100.0% | —% |
| Hallucination rate | 0.0% | —% |
| Runs needing validator retries | 0.0% | —% |
| Mean explanation score (1-5) | 4.89 | — |
| Mean latency (s) | 64.93 | 239.6 |
| Mean tokens in / out | 4201 / 1156 | — / — |

## Per-family metrics (RAG ON)

| Family | Solve correct | Feasibility correct | Hallucination | Expl. score | Clarifier flag rate (ambiguous) |
|---|---|---|---|---|---|
| diet | 100.0% | 100.0% | 0.0% | 5.0 | —% |
| transportation | 100.0% | 100.0% | 0.0% | 5.0 | —% |
| facility_location | 100.0% | 100.0% | 0.0% | 4.67 | —% |

## Validator retry loop

Total validator-triggered modeling retries: 0 (RAG on), 0 (RAG off).

## Example transcripts

### Success case

**Problem `diet_01`** (RAG on), pipeline status: completed.

- Agent objective: 0.623076924 vs ground truth 0.623076924

> To minimize cost, we should use approximately 46% corn and 54% soy. This mix will cost about $0.62 per batch.
> 
> The two main limitations we're hitting are:
> 
> *   **Batch Size:** We are making exactly one batch, as required.
> *   **Minimum Protein:** We are meeting the minimum protein requirement exactly. Any less protein would not meet the nutritional needs.
> 
> The maximum fiber constraint is not limiting us, meaning we could potentially use less fiber if needed, but it doesn't impact the current opt

### Retry / failure case

**Problem `diet_01`** (RAG off), pipeline status: crashed.

- Error: ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash-lite\nPlease retry in 58.909413862s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-lite'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '58s'}]}}

### Hallucination case

_No qualifying run in this evaluation._

## Known limitations

- Ambiguous problems are scored on clarifier behavior and hallucination only; batch mode proceeds with documented standard defaults instead of asking the user.
- The hallucination metric combines an LLM judge with a retry-count heuristic; judge verdicts carry rationales in results.json and can be audited.
- Only LP/MILP with expanded linear expressions are supported; the expression grammar rejects nonlinear or indexed forms by design.
- If dense embeddings are unavailable at runtime, retrieval degrades to BM25-only and logs a warning; results then measure the degraded retriever.
