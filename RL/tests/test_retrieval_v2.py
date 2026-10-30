"""Deterministic tables, HTML filings, value facts, GraphRAG search modes, multi-dataset routing and EDGAR selection."""
import json
from unittest.mock import patch

import numpy as np
import pytest


FINQA_DOC = """SOURCE: VRTX/2003/page_71.pdf

under the agreement , novartis agreed to pay the company an up-front payment .

TABLE HEADERS: common stock under stock and option plans | 21829

TABLE ROW common stock under the vertex purchase plan | 21829: 249

TABLE ROW total | 21829: 22203

the company also has 401 ( k ) plans ."""


def test_finqa_table_rows_become_facts_including_totals():
    from app import table_parser
    entities, triples, prose = table_parser.detect_and_extract_tables(FINQA_DOC, "vrtx")
    values = {(t.subject, t.object) for t in triples if t.predicate != "reports"}
    # The header row is data in this FinQA table, and later rows name columns by its values.
    assert ("VRTX common stock under stock and option plans", "21829") in values
    assert ("VRTX total", "22203") in values
    assert all(t.predicate in ("reports", "value") for t in triples)
    assert "TABLE ROW" not in prose and "novartis" in prose and "401" in prose
    chunks = table_parser.table_chunk_texts(FINQA_DOC, "vrtx")
    assert {t.chunk_id for t in triples} == {c["chunk_id"] for c in chunks}
    assert "total | value: 22203" in chunks[0]["text"]


def test_labeled_tables_qualify_generic_rows_and_skip_scale_for_rates():
    from app import table_parser
    doc = """COMPANY: Acme Air, Inc. (ACM)

Results ($ in millions, except per share)

TABLE HEADERS: ($ in millions, except per share) | 3Q26 | 3Q25
TABLE SECTION Operating expense:
TABLE ROW Fuel | 3Q26: 4,351 ; 3Q25: 2,570
TABLE ROW Total | 3Q26: 18,732 ; 3Q25: 14,989
TABLE ROW Operating margin | 3Q26: 7.2% ; 3Q25: 10.1%"""
    entities, triples, _ = table_parser.detect_and_extract_tables(doc, "acm")
    facts = {(t.subject, t.predicate): t for t in triples}
    assert facts[("Acme Air, Inc. Operating expense - Total", "3Q26")].object == "18,732"
    assert facts[("Acme Air, Inc. Fuel", "3Q25")].evidence.endswith("(in millions)")
    assert "millions" not in facts[("Acme Air, Inc. Operating margin", "3Q26")].evidence
    assert ("Acme Air, Inc.", "reports") in {(t.subject, t.predicate) for t in triples}


def test_whitespace_tables_still_parse():
    from app import table_parser
    doc = "Five9, Inc. results\n                        Q2 2025      Q2 2024\nStock-based compensation   41,859       43,632\n"
    _, triples, prose = table_parser.detect_and_extract_tables(doc, "f9")
    assert {(t.predicate, t.object) for t in triples if t.predicate != "reports"} == {("Q2 2025", "41,859"), ("Q2 2024", "43,632")}
    assert "41,859" not in prose


def test_html_filing_tables_keep_column_meaning_across_split_cells():
    from app.html_text import html_to_text
    html = """<html><body><p>Acme reports results.</p>
    <table>
      <tr><td></td><td colspan="3">Three Months Ended</td></tr>
      <tr><td>($ in millions)</td><td colspan="2">2026</td><td>2025</td></tr>
      <tr><td>Revenue</td><td>$</td><td>1,234</td><td>1,100</td></tr>
      <tr><td>Net loss</td><td>$</td><td>(56</td><td>)</td></tr>
      <tr><td>Operating expense:</td><td></td><td></td><td></td></tr>
      <tr><td>Total</td><td></td><td>900</td><td>850</td></tr>
    </table><p>Outlook remains strong.</p></body></html>"""
    text = html_to_text(html)
    assert "TABLE ROW Revenue | Three Months Ended 2026: $1,234 ; Three Months Ended 2025: 1,100" in text
    assert "TABLE ROW Net loss | Three Months Ended 2026: $(56)" in text
    assert "TABLE ROW Operating expense - Total" in text
    assert text.index("Acme reports") < text.index("TABLE") < text.index("Outlook")


def test_values_are_node_facts_and_periods_never_merge():
    from app.graph_store import GraphStore, is_literal
    from app.models import Entity, ExtractionResult, Triple
    store = GraphStore()
    narrative = ExtractionResult(doc_id="d1", chunk_id="d1::chunk0", entities=[
        Entity(name="Acme net revenue Q2 2025", type="Metric"), Entity(name="Acme net revenue Q2 2024", type="Metric")],
        triples=[Triple(subject="Acme net revenue Q2 2025", predicate="has value", object="$108.8 million", doc_id="d1", chunk_id="d1::chunk0", evidence=""),
                 Triple(subject="Acme net revenue Q2 2024", predicate="has value", object="$100.0 million", doc_id="d1", chunk_id="d1::chunk0", evidence=""),
                 Triple(subject="Acme net revenue Q2 2025", predicate="changed by", object="up", doc_id="d1", chunk_id="d1::chunk0", evidence="")])
    with patch("app.text_embeddings.embed_texts", side_effect=RuntimeError("offline")):
        store.ingest_document_chunks("d1", [{"chunk_id": "d1::chunk0", "doc_id": "d1", "text": "..."}], [narrative])
    assert store.graph.has_node("Acme net revenue Q2 2025") and store.graph.has_node("Acme net revenue Q2 2024")
    assert not store.graph.has_node("$108.8 million") and not store.graph.has_node("up")
    values = store.graph.nodes["Acme net revenue Q2 2025"]["values"]
    assert {v["value"] for v in values} == {"$108.8 million", "up"}
    assert store.chunk_ids_of("Acme net revenue Q2 2025") == {"d1::chunk0"}
    assert is_literal("(1,234)") and is_literal("Q3 2026") and not is_literal("3M Company Inc")


def test_table_metrics_match_exactly_but_companies_link_to_narrative_names():
    from app.graph_store import GraphStore
    from app.models import Entity, ExtractionResult, Triple
    store = GraphStore()
    table = ExtractionResult(doc_id="d1", chunk_id="d1::tables", entities=[
        Entity(name="Delta Air Lines, Inc.", type="Company"), Entity(name="Delta Air Lines, Inc. Operating income", type="Metric"),
        Entity(name="Delta Air Lines, Inc. Operating incomes", type="Metric")],
        triples=[Triple(subject="Delta Air Lines, Inc.", predicate="reports", object="Delta Air Lines, Inc. Operating income", doc_id="d1", chunk_id="d1::table0", evidence="")])
    narrative = ExtractionResult(doc_id="d2", chunk_id="d2::chunk0", entities=[Entity(name="Delta Air Lines", type="Company")], triples=[])
    with patch("app.text_embeddings.embed_texts", side_effect=RuntimeError("offline")):
        store.ingest_document_chunks("d1", [], [table])
        store.ingest_document_chunks("d2", [], [narrative])
    assert store.graph.has_node("Delta Air Lines, Inc. Operating incomes")  # near-identical table names stay distinct
    assert not store.graph.has_node("Delta Air Lines")  # the narrative name resolved to the table's company
    assert store.graph.nodes["Delta Air Lines, Inc."]["source_docs"] == ["d1", "d2"]


def _graph_store():
    from app.graph_store import GraphStore
    store = GraphStore()
    store.graph.add_node("Acme", type="Company", description="", source_docs=["a"], values=[])
    store.graph.add_node("Acme revenue", type="Metric", description="", source_docs=["a"],
                         values=[{"predicate": "Q3 2026", "value": "1,234", "doc_id": "a", "chunk_id": "a::table0", "evidence": "Revenue Q3 2026: 1,234"}])
    store.graph.add_edge("Acme", "Acme revenue", predicate="reports", doc_id="a", chunk_id="a::table0", evidence="")
    store.chunks = {"a::table0": {"doc_id": "a", "text": "TABLE ROW Revenue | Q3 2026: 1,234"},
                    "a::chunk0": {"doc_id": "a", "text": "Acme grew on strong demand."}}
    return store


def test_local_search_uses_value_facts_and_reuses_classification():
    from app import query_engine
    store = _graph_store()
    with patch.object(query_engine.GraphStore, "load", return_value=store), \
         patch.object(query_engine, "_fuzzy_match_nodes", return_value=["Acme"]), \
         patch.object(query_engine.rgcn, "nearest_nodes", return_value=[]), \
         patch.object(query_engine, "classify_query", side_effect=AssertionError("classification must be reused")), \
         patch.object(query_engine, "generate_answer", return_value="1,234") as generate:
        response = query_engine.answer_question("Acme revenue in Q3 2026?", classification=("multi_hop", ["Acme"]))
    assert response.mode_used == "local"  # multi-hop questions search the entity neighborhood
    context = generate.call_args.args[1]
    assert "Acme revenue | Q3 2026: 1,234" in context and "TABLE ROW Revenue" in context


def test_global_search_grounds_summaries_with_source_text():
    from app import query_engine
    from app.models import CommunitySummary
    store = _graph_store()
    summary = CommunitySummary(community_id=0, level=0, members=["Acme", "Acme revenue"], title="Acme growth", summary="Acme grew.")
    with patch.object(query_engine.GraphStore, "load", return_value=store), \
         patch.object(query_engine, "load_summaries", return_value=[summary]), \
         patch.object(query_engine, "generate_answer", return_value="Demand") as generate:
        response = query_engine.answer_question("What themes drove growth?", classification=("global", []))
    context = generate.call_args.args[1]
    assert response.mode_used == "global"
    assert "COMMUNITY SUMMARIES" in context and "SOURCE TEXT" in context and "1,234" in context
    assert {c.chunk_id for c in response.citations} >= {"a::table0", "community_0"}


def _rows(dataset, tasks, latency=None):
    from app import benchmark, engines
    meta = {**benchmark.metadata(), "dataset": dataset, "split": "train", "dataset_fingerprint": dataset + "-fp", "metric": "m"}
    rows = []
    for task, category in tasks:
        row = {"task": task, "benchmark_category": category, "category": "local", "context": [0] * 8,
               "metadata": meta, **{"reward_" + e: .5 for e in engines.ENGINE_NAMES}}
        if latency:
            row["telemetry"] = {e: {"latency_ms": latency[e]} for e in engines.ENGINE_NAMES}
        rows.append(row)
    return rows


def _write(folder, rows):
    folder.mkdir(parents=True)
    (folder / "query_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "summary.json").write_text('{"complete": true}')
    return folder / "query_results.jsonl"


def test_router_training_weights_datasets_equally(tmp_path):
    from train_shared_router import load_weighted
    text = [("text", c) for c in ("local", "global", "multi_hop", "conflict")]
    finqa = _write(tmp_path / "finqa", _rows("shared_finqa", [("numeric", "local")] * 6 + text))
    edgar = _write(tmp_path / "edgar", _rows("edgar_xdoc", text * 3))
    rows = load_weighted([finqa, edgar])
    total = sum(r["sample_weight"] for r in rows)
    assert sum(r["sample_weight"] for r in rows if r["metadata"]["dataset"] == "edgar_xdoc") == pytest.approx(total / 2)
    numeric = sum(r["sample_weight"] for r in rows if r["task"] == "numeric")
    assert numeric == pytest.approx(total / 4)  # half of FinQA's half


def test_router_training_rejects_mixed_models_across_datasets(tmp_path):
    from train_shared_router import load_weighted
    text = [("text", c) for c in ("local", "global", "multi_hop", "conflict")]
    other = _rows("edgar_xdoc", text)
    for row in other:
        row["metadata"] = {**row["metadata"], "answer_model": "another-model"}
    with pytest.raises(ValueError, match="different models"):
        load_weighted([_write(tmp_path / "a", _rows("shared_finqa", text)), _write(tmp_path / "b", other)])


def test_cost_aware_reward_prefers_faster_engine_when_quality_ties():
    from train_shared_router import effective_reward
    row = {"reward_graphrag": .8, "reward_hybrid_rag": .8,
           "telemetry": {"graphrag": {"latency_ms": 45000}, "hybrid_rag": {"latency_ms": 9000}}}
    assert effective_reward(row, "hybrid_rag", .2) > effective_reward(row, "graphrag", .2)
    assert effective_reward(row, "graphrag", 0) == .8
    with pytest.raises(ValueError, match="telemetry"):
        effective_reward({"reward_graphrag": 1}, "graphrag", .2)


def test_policy_from_other_dataset_versions_is_rejected_but_new_datasets_transfer():
    from run_public_benchmark import validate_policy
    from app import benchmark, engines
    from app.query_bandit import LinUCBBandit
    meta = benchmark.metadata()
    policy = LinUCBBandit(8, list(engines.ENGINE_NAMES))
    policy.metadata = {**meta, "split": "train", "datasets": {"shared_finqa": "fp1"}}
    validate_policy(policy, {"dataset": "edgar_xdoc", "fingerprint": "x"}, meta)  # unseen dataset: transfer evaluation
    with pytest.raises(ValueError, match="different version"):
        validate_policy(policy, {"dataset": "shared_finqa", "fingerprint": "changed"}, meta)


def test_edgar_selects_press_release_exhibit_and_renders_header(tmp_path):
    import prepare_edgar
    submissions = {"name": "COSTCO WHOLESALE CORP /NEW", "filings": {"recent": {
        "form": ["8-K", "10-Q", "8-K", "8-K"], "items": ["2.02,9.01", "", "5.02", "2.02"],
        "accessionNumber": ["0001-26-000002", "x", "y", "0001-26-000001"],
        "filingDate": ["2026-09-24", "2026-08-01", "2026-07-01", "2026-05-28"]}}}
    index = ('<tr><td>1</td><td>8-K</td><td><a href="/Archives/x/main.htm">main.htm</a></td><td>8-K</td></tr>'
             '<tr><td>2</td><td>Release</td><td><a href="/Archives/x/release.htm">release.htm</a></td><td>EX-99</td></tr>')

    class FakeEdgar:
        def get(self, url, cache_name=None):
            return json.dumps(submissions).encode() if url.endswith(".json") else index.encode()
    releases = prepare_edgar.earnings_releases(FakeEdgar(), "COST", {"COST": {"cik_str": 909832, "title": "x"}})
    assert [r["filed"] for r in releases] == ["2026-09-24", "2026-05-28"]
    assert releases[0]["url"] == "https://www.sec.gov/Archives/x/release.htm"
    assert releases[0]["company"] == "Costco Wholesale Corp"
    text = prepare_edgar.render(releases[0], "<p>Costco reported net sales growth.</p>")
    assert text.startswith("COMPANY: Costco Wholesale Corp (COST)") and "net sales growth" in text
    from app.table_parser import guess_primary_entity
    assert guess_primary_entity(text) == "Costco Wholesale Corp"


def test_truncated_answer_retries_with_double_budget_then_scores_as_non_answer():
    from unittest.mock import patch
    from app import answering, config
    from app.llm_client import OutputTruncatedError
    budgets = []
    def truncate_then_answer(model, system, user, max_tokens, temperature):
        budgets.append(max_tokens)
        if len(budgets) == 1:
            raise OutputTruncatedError("length")
        return "Revenue was $10 million."
    with patch.object(answering, "chat", side_effect=truncate_then_answer):
        assert answering.generate_answer("Revenue?", "ctx") == "Revenue was $10 million."
    assert budgets == [config.ANSWER_MAX_TOKENS, config.ANSWER_MAX_TOKENS * 2]
    with patch.object(answering, "chat", side_effect=OutputTruncatedError("length")):
        assert answering.generate_answer("Revenue?", "ctx") == answering.TRUNCATED_ANSWER


def test_looping_extraction_splits_the_passage_instead_of_failing():
    from unittest.mock import patch
    from app import extraction
    from app.llm_client import OutputTruncatedError
    text = ("Acme revenue grew 10% in Q2 2026. " * 20) + "\n\n" + ("Acme margin fell to 20% in Q2 2026. " * 20)
    seen = []
    def reply(model, system, user, max_tokens):
        seen.append(user)
        if user == text:
            raise OutputTruncatedError("loop")
        return {"entities": [{"name": "Acme", "type": "Company"}], "triples": []}
    with patch.object(extraction, "chat_json", side_effect=reply):
        data = extraction._extract_json(text, "c0")
    assert len(data["entities"]) == 2 and len(seen) == 3
    assert seen[1] + " " + seen[2] != text and "revenue" in seen[1] and "margin" in seen[2]
    with patch.object(extraction, "chat_json", side_effect=OutputTruncatedError("loop")):
        assert extraction._extract_json("short passage", "c1") == {"entities": [], "triples": []}


def test_embedding_cache_reuses_vectors_in_input_order():
    from unittest.mock import patch
    import numpy as np
    from app import text_embeddings
    calls = []
    def embedder(texts):
        calls.append(list(texts))
        return [np.full(3, float(len(t))) for t in texts]
    with patch.object(text_embeddings, "get_embedder", return_value=embedder), patch.dict(text_embeddings._cache, clear=True):
        first = text_embeddings.embed_texts(["a", "bb"])
        second = text_embeddings.embed_texts(["bb", "ccc", "a"])
    assert calls == [["a", "bb"], ["ccc"]]
    assert first[:, 0].tolist() == [1, 2] and second[:, 0].tolist() == [2, 3, 1]
