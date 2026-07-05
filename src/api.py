"""FastAPI app exposing the OptiAgent pipeline.

POST /solve  -> run the full pipeline on one natural-language problem
POST /eval   -> trigger a (potentially long) benchmark evaluation run
GET  /health -> liveness probe
"""

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from src.graph import run_pipeline

app = FastAPI(title="OptiAgent", version="0.1.0")


class SolveRequest(BaseModel):
    problem: str
    rag_enabled: bool = True
    interactive: bool = False


class SolveResponse(BaseModel):
    status: str  # "solved" | "clarification_needed" | "failed"
    clarification_request: str | None = None
    error: str | None = None
    problem_family: str | None = None
    spec: dict | None = None
    solve_result: dict | None = None
    explanation: str | None = None
    retrieved_sources: list[str] = []
    input_tokens: int = 0
    output_tokens: int = 0


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> SolveResponse:
    state = run_pipeline(
        request.problem,
        rag_enabled=request.rag_enabled,
        interactive=request.interactive,
    )
    usage = state["usage"]
    common = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }
    if state.get("clarification_request"):
        return SolveResponse(
            status="clarification_needed",
            clarification_request=state["clarification_request"],
            problem_family=state["intake"].problem_family,
            **common,
        )
    if state.get("error"):
        return SolveResponse(status="failed", error=state["error"], **common)
    return SolveResponse(
        status="solved",
        problem_family=state["intake"].problem_family,
        spec=state["spec"].model_dump(),
        solve_result=state["solve_result"].model_dump(),
        explanation=state["explained"].explanation,
        retrieved_sources=state["context"].source_ids,
        **common,
    )


@app.post("/eval")
def trigger_eval(background_tasks: BackgroundTasks) -> dict:
    from eval.run_eval import main as run_eval_main

    background_tasks.add_task(run_eval_main)
    return {
        "status": "started",
        "detail": "Evaluation running in background; results land in eval/report.md "
        "and eval/results.json.",
    }
