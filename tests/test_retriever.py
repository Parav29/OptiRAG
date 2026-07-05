import pytest

from src.agents.retriever import _rrf, retrieve
from src.kb_index import KBIndex, load_chunks
from src.schemas import ProblemIntake, RetrievedContext


@pytest.fixture(scope="module")
def index() -> KBIndex:
    return KBIndex()


def test_kb_chunking_yields_expected_density():
    chunks = load_chunks()
    families = {"diet.md", "transportation.md", "facility_location.md"}
    seen = {c.source_id.split("#")[0] for c in chunks}
    assert families <= seen
    for family_file in families:
        count = sum(1 for c in chunks if c.source_id.startswith(family_file))
        assert 6 <= count <= 12, f"{family_file} has {count} chunks"


def test_rag_disabled_returns_empty_context(index):
    intake = ProblemIntake(raw_text="cheapest feed blend with protein minimum",
                           problem_family="diet")
    ctx = retrieve(intake, rag_enabled=False, index=index)
    assert ctx == RetrievedContext()
    assert ctx.chunks == []


def test_retrieval_returns_top_5(index):
    intake = ProblemIntake(
        raw_text="ship widgets from two plants to three stores at least cost",
        problem_family="transportation",
    )
    ctx = retrieve(intake, rag_enabled=True, index=index)
    assert len(ctx.chunks) == 5
    assert len(ctx.scores) == 5
    assert len(ctx.source_ids) == 5


def test_retrieval_surfaces_matching_family(index):
    intake = ProblemIntake(
        raw_text="which warehouses should we open given fixed opening costs "
                 "and demand in three regions",
        problem_family="facility_location",
    )
    ctx = retrieve(intake, rag_enabled=True, index=index)
    top_sources = " ".join(ctx.source_ids[:3])
    assert "facility_location.md" in top_sources


def test_worked_example_chunk_retrievable(index):
    intake = ProblemIntake(
        raw_text="blend corn and soybean meal for cheapest feed with 30% protein",
        problem_family="diet",
    )
    ctx = retrieve(intake, rag_enabled=True, index=index)
    assert any("diet.md" in sid for sid in ctx.source_ids)


def test_rrf_fuses_rankings():
    fused = _rrf([[0, 1, 2], [2, 0, 3]], k=60)
    # 0 appears high in both lists -> best fused score
    assert max(fused, key=fused.get) == 0
    assert set(fused) == {0, 1, 2, 3}


def test_bm25_search_ranks_relevant_family_first(index):
    hits = index.bm25_search("binary variable fixed cost open facility big-M", 5)
    top_chunk = index.chunks[hits[0][0]]
    assert "facility_location.md" in top_chunk.source_id
