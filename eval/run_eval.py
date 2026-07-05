"""Evaluation harness: run the full pipeline over the benchmark set, twice
(RAG on / RAG off), score every stage, and emit report.md + results.json.

Run: python -m eval.run_eval [--problems eval/benchmark_problems.json]
                             [--conditions on off] [--limit N]
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from statistics import mean

from eval.judge import judge_explanation, judge_hallucination
from src.graph import run_pipeline
from src.llm import Usage
from src.schemas import OptimizationSpec

EVAL_DIR = Path(__file__).parent
REL_TOL = 0.01  # 1% relative tolerance on the objective value

FAMILIES = ("diet", "transportation", "facility_location")


def evaluate_problem(problem: dict, rag_enabled: bool) -> dict:
    record = {
        "id": problem["id"],
        "family": problem["family"],
        "ambiguous": problem["ambiguous"],
        "rag_enabled": rag_enabled,
        "ground_truth_status": problem["ground_truth_status"],
        "ground_truth_objective": problem["ground_truth_objective"],
    }
    judge_usage = Usage()
    start = time.monotonic()
    try:
        state = run_pipeline(problem["prompt"], rag_enabled=rag_enabled,
                             interactive=False)
    except Exception as exc:  # noqa: BLE001 - a crash is an eval datapoint
        record.update({
            "pipeline_status": "crashed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
            "latency_s": round(time.monotonic() - start, 2),
        })
        return record
    record["latency_s"] = round(time.monotonic() - start, 2)

    usage: Usage = state["usage"]
    record["input_tokens"] = usage.input_tokens
    record["output_tokens"] = usage.output_tokens
    intake = state.get("intake")
    record["clarifier_flagged_missing_info"] = bool(
        intake and intake.missing_info
    )
    record["missing_info"] = list(intake.missing_info) if intake else []
    record["validation_retries"] = max(0, state.get("modeling_attempts", 1) - 1)
    record["retrieved_sources"] = (
        state["context"].source_ids if state.get("context") else []
    )

    if state.get("error"):
        record["pipeline_status"] = "failed_validation"
        record["error"] = state["error"]
        record["spec"] = state["spec"].model_dump() if state.get("spec") else None
        return record

    record["pipeline_status"] = "completed"
    spec: OptimizationSpec = state["spec"]
    result = state["solve_result"]
    record["spec"] = spec.model_dump()
    record["agent_status"] = result.status
    record["agent_objective"] = result.objective_value
    record["explanation"] = state["explained"].explanation

    # --- correctness metrics (only where ground truth exists) ---------------
    gt_status = problem["ground_truth_status"]
    gt_obj = problem["ground_truth_objective"]
    if gt_status is not None:
        record["feasibility_correct"] = result.status == gt_status
    if gt_obj is not None:
        if result.objective_value is None:
            record["solve_correct"] = False
        else:
            record["solve_correct"] = (
                abs(result.objective_value - gt_obj)
                <= REL_TOL * max(1e-9, abs(gt_obj))
            )

    # --- LLM-as-judge --------------------------------------------------------
    # Judging is isolated: a failure here (e.g. a quota 429 mid-run) must not
    # discard the successful pipeline result. The record is still saved with
    # judge fields marked incomplete so the run stays resumable and the report
    # simply omits it from judge-based aggregates.
    try:
        verdict = judge_hallucination(problem["prompt"], spec, usage=judge_usage)
        record["judge_hallucination"] = verdict.model_dump()
        # Hallucinated if the judge flags invented content OR the validator
        # needed more than 1 retry (spec §6.2 metric 3).
        record["hallucinated"] = bool(
            verdict.hallucinated or record["validation_retries"] > 1
        )

        score = judge_explanation(state["explained"], usage=judge_usage)
        record["judge_explanation"] = score.model_dump()
        record["explanation_score"] = round(
            (score.correctness + score.clarity + score.completeness) / 3, 2
        )
        record["judge_tokens"] = judge_usage.input_tokens + judge_usage.output_tokens
    except Exception as exc:  # noqa: BLE001 - judging is best-effort
        record["judge_error"] = f"{type(exc).__name__}: {exc}"
    return record


# --------------------------- aggregation ------------------------------------

def _rate(records: list[dict], key: str) -> float | None:
    values = [r[key] for r in records if key in r]
    return round(100 * sum(values) / len(values), 1) if values else None


def _mean(records: list[dict], key: str) -> float | None:
    values = [r[key] for r in records if r.get(key) is not None]
    return round(mean(values), 2) if values else None


def aggregate(records: list[dict]) -> dict:
    completed = [r for r in records if r.get("pipeline_status") == "completed"]
    return {
        "n_problems": len(records),
        "n_completed": len(completed),
        "n_failed": len(records) - len(completed),
        "solve_correct_pct": _rate(records, "solve_correct"),
        "feasibility_correct_pct": _rate(records, "feasibility_correct"),
        "hallucination_rate_pct": _rate(completed, "hallucinated"),
        "runs_with_retries_pct": _rate(
            [{**r, "retried": r.get("validation_retries", 0) > 0}
             for r in records if "validation_retries" in r], "retried"),
        "total_validation_retries": sum(
            r.get("validation_retries", 0) for r in records),
        "clarifier_flag_rate_on_ambiguous_pct": _rate(
            [{**r, "f": r.get("clarifier_flagged_missing_info", False)}
             for r in records if r["ambiguous"]], "f"),
        "mean_explanation_score": _mean(completed, "explanation_score"),
        "mean_latency_s": _mean(records, "latency_s"),
        "mean_input_tokens": _mean(records, "input_tokens"),
        "mean_output_tokens": _mean(records, "output_tokens"),
    }


# --------------------------- report -----------------------------------------

def _fmt(value) -> str:
    return "—" if value is None else str(value)


def build_report(records: list[dict]) -> str:
    on = [r for r in records if r["rag_enabled"]]
    off = [r for r in records if not r["rag_enabled"]]
    agg_on, agg_off = aggregate(on), aggregate(off)

    lines = ["# OptiAgent Evaluation Report", ""]
    lines.append(f"Benchmark: {len(on)} problems × 2 conditions "
                 "(RAG on / RAG off), identical model and settings.")
    lines.append("")

    lines.append("## Headline: RAG ablation")
    lines.append("")
    lines.append("| Metric | RAG ON | RAG OFF |")
    lines.append("|---|---|---|")
    for label, key in [
        ("Solve correctness (obj within 1% of ground truth)", "solve_correct_pct"),
        ("Feasibility status correctness", "feasibility_correct_pct"),
        ("Hallucination rate", "hallucination_rate_pct"),
        ("Runs needing validator retries", "runs_with_retries_pct"),
        ("Mean explanation score (1-5)", "mean_explanation_score"),
        ("Mean latency (s)", "mean_latency_s"),
        ("Mean tokens in / out", None),
    ]:
        if key is None:
            lines.append(
                f"| {label} | {_fmt(agg_on['mean_input_tokens'])} / "
                f"{_fmt(agg_on['mean_output_tokens'])} | "
                f"{_fmt(agg_off['mean_input_tokens'])} / "
                f"{_fmt(agg_off['mean_output_tokens'])} |")
        else:
            suffix = "%" if key.endswith("pct") else ""
            lines.append(f"| {label} | {_fmt(agg_on[key])}{suffix} | "
                         f"{_fmt(agg_off[key])}{suffix} |")
    lines.append("")

    lines.append("## Per-family metrics (RAG ON)")
    lines.append("")
    lines.append("| Family | Solve correct | Feasibility correct | "
                 "Hallucination | Expl. score | Clarifier flag rate "
                 "(ambiguous) |")
    lines.append("|---|---|---|---|---|---|")
    for family in FAMILIES:
        rows = [r for r in on if r["family"] == family]
        agg = aggregate(rows)
        lines.append(
            f"| {family} | {_fmt(agg['solve_correct_pct'])}% | "
            f"{_fmt(agg['feasibility_correct_pct'])}% | "
            f"{_fmt(agg['hallucination_rate_pct'])}% | "
            f"{_fmt(agg['mean_explanation_score'])} | "
            f"{_fmt(agg['clarifier_flag_rate_on_ambiguous_pct'])}% |")
    lines.append("")

    lines.append("## Validator retry loop")
    lines.append("")
    lines.append(
        f"Total validator-triggered modeling retries: "
        f"{agg_on['total_validation_retries']} (RAG on), "
        f"{agg_off['total_validation_retries']} (RAG off).")
    retried = [r for r in records if r.get("validation_retries", 0) > 0]
    for r in retried[:5]:
        lines.append(f"- `{r['id']}` (RAG {'on' if r['rag_enabled'] else 'off'}): "
                     f"{r['validation_retries']} retr"
                     f"{'y' if r['validation_retries'] == 1 else 'ies'}")
    lines.append("")

    lines.append("## Example transcripts")
    lines.append("")
    _add_transcript(lines, "Success case", next(
        (r for r in on if r.get("solve_correct")), None))
    _add_transcript(lines, "Retry / failure case", next(
        (r for r in records if r.get("validation_retries", 0) > 0
         or r.get("pipeline_status") != "completed"), None))
    _add_transcript(lines, "Hallucination case", next(
        (r for r in records if r.get("hallucinated")), None))

    lines.append("## Known limitations")
    lines.append("")
    lines.extend([
        "- Ambiguous problems are scored on clarifier behavior and "
        "hallucination only; batch mode proceeds with documented standard "
        "defaults instead of asking the user.",
        "- The hallucination metric combines an LLM judge with a retry-count "
        "heuristic; judge verdicts carry rationales in results.json and can "
        "be audited.",
        "- Only LP/MILP with expanded linear expressions are supported; the "
        "expression grammar rejects nonlinear or indexed forms by design.",
        "- If dense embeddings are unavailable at runtime, retrieval "
        "degrades to BM25-only and logs a warning; results then measure the "
        "degraded retriever.",
    ])
    lines.append("")
    return "\n".join(lines)


def _add_transcript(lines: list[str], title: str, record: dict | None) -> None:
    lines.append(f"### {title}")
    lines.append("")
    if record is None:
        lines.append("_No qualifying run in this evaluation._")
        lines.append("")
        return
    lines.append(f"**Problem `{record['id']}`** "
                 f"(RAG {'on' if record['rag_enabled'] else 'off'}), "
                 f"pipeline status: {record.get('pipeline_status')}.")
    lines.append("")
    if record.get("agent_objective") is not None:
        lines.append(
            f"- Agent objective: {record['agent_objective']} vs ground truth "
            f"{record.get('ground_truth_objective')}")
    if record.get("validation_retries"):
        lines.append(f"- Validator retries: {record['validation_retries']}")
    if record.get("judge_hallucination", {}).get("hallucinated"):
        lines.append(
            f"- Judge verdict: {record['judge_hallucination']['rationale']}")
    if record.get("error"):
        lines.append(f"- Error: {record['error']}")
    if record.get("explanation"):
        excerpt = record["explanation"][:500]
        lines.append("")
        lines.append("> " + excerpt.replace("\n", "\n> "))
    lines.append("")


# --------------------------- entry point ------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default=str(EVAL_DIR / "benchmark_problems.json"))
    parser.add_argument("--conditions", nargs="+", default=["on", "off"],
                        choices=["on", "off"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N problems (smoke runs).")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing results.json and start over.")
    args = parser.parse_args(argv)

    problems = json.loads(Path(args.problems).read_text())
    if args.limit:
        problems = problems[: args.limit]

    results_path = EVAL_DIR / "results.json"
    report_path = EVAL_DIR / "report.md"

    # Resume support: reuse any prior successful run for a (problem, condition)
    # so an interrupted run (e.g. daily quota 429) continues without re-spending
    # calls. Records that previously "crashed" are retried.
    records: list[dict] = []
    done: dict[tuple[str, bool], dict] = {}
    if results_path.exists() and not args.fresh:
        for rec in json.loads(results_path.read_text()):
            if rec.get("pipeline_status") != "crashed":
                done[(rec["id"], rec["rag_enabled"])] = rec
        if done:
            print(f"Resuming: {len(done)} prior results kept "
                  "(use --fresh to ignore them).")

    def flush() -> None:
        results_path.write_text(json.dumps(records, indent=2))

    for condition in args.conditions:
        rag_enabled = condition == "on"
        print(f"=== Condition: RAG {'ON' if rag_enabled else 'OFF'} ===")
        for problem in problems:
            key = (problem["id"], rag_enabled)
            if key in done:
                records.append(done[key])
                print(f"  {problem['id']} ... (cached)")
                continue
            print(f"  {problem['id']} ...", end=" ", flush=True)
            try:
                record = evaluate_problem(problem, rag_enabled)
            except KeyboardInterrupt:
                print("interrupted — writing partial results.")
                flush()
                raise
            except Exception as exc:  # noqa: BLE001 - never let one problem kill the run
                record = {
                    "id": problem["id"], "family": problem["family"],
                    "ambiguous": problem["ambiguous"], "rag_enabled": rag_enabled,
                    "pipeline_status": "crashed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)
            flush()  # incremental save after every problem
            print(record.get("pipeline_status"),
                  f"(solve_correct={record.get('solve_correct', 'n/a')})")

    flush()
    report = build_report(records)
    report_path.write_text(report)
    print(f"\nWrote {results_path} and {report_path}")


if __name__ == "__main__":
    main()
