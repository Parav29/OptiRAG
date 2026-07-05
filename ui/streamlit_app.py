"""Minimal Streamlit demo UI for OptiAgent. Talks to the FastAPI /solve endpoint."""

import os

import httpx
import streamlit as st

API_URL = os.getenv("OPTIAGENT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="OptiAgent", page_icon="📈")
st.title("OptiAgent — NL → Optimization Model")
st.caption(
    "Describe a diet/blending, transportation, or facility-location problem "
    "in plain English. OptiAgent formulates it, solves it with CBC, and "
    "explains the answer."
)

rag_enabled = st.sidebar.toggle("RAG grounding", value=True)
st.sidebar.markdown(
    "Turning RAG off skips knowledge-base retrieval — the same switch the "
    "eval harness uses for the ablation study."
)

example = (
    "A feed mill blends corn ($0.30/kg) and soybean meal ($0.90/kg). The blend "
    "must contain at least 30% protein and at most 5% fiber. Corn is 9% protein "
    "and 2% fiber; soybean meal is 48% protein and 6% fiber. Find the cheapest "
    "mix per kg of blend."
)
problem = st.text_area("Problem description", value=example, height=180)

if st.button("Solve", type="primary") and problem.strip():
    with st.spinner("Running the agent pipeline..."):
        try:
            response = httpx.post(
                f"{API_URL}/solve",
                json={"problem": problem, "rag_enabled": rag_enabled,
                      "interactive": True},
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    if data["status"] == "clarification_needed":
        st.warning(data["clarification_request"])
    elif data["status"] == "failed":
        st.error(data["error"])
    else:
        result = data["solve_result"]
        st.subheader(f"Status: {result['status']}")
        if result["objective_value"] is not None:
            st.metric("Objective value", f"{result['objective_value']:,.4f}")
        st.markdown("### Explanation")
        st.write(data["explanation"])
        st.markdown("### Decision variables")
        st.json(result["variable_values"])
        with st.expander("Optimization spec (generated)"):
            st.json(data["spec"])
        with st.expander("Retrieved KB sources"):
            st.write(data["retrieved_sources"] or "RAG disabled — no retrieval.")
        st.caption(
            f"Tokens: {data['input_tokens']} in / {data['output_tokens']} out"
        )
