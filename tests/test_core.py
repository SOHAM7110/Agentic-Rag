"""Core tests. Run: `python -m tests.test_core` (or pytest). Hermetic: builds its own tiny corpora, needs no keys/network."""
import tests.helpers as H           # noqa: F401  (forces dependency-free backends; must be imported first)
import json
from agent import tools as T
from agent.minigraph import StateGraph, END
from eval import metrics as M
from retrieval.fusion import rrf
from retrieval.text import split_sentences, tokenize


# ------------------------------------------------------------------ text / retrieval utilities
def test_sentence_split_keeps_rupee_amounts():
    assert split_sentences("The outlay is Rs. 840 crore. It began in 2015.") == ["The outlay is Rs. 840 crore.", "It began in 2015."]


def test_number_normalisation():
    assert tokenize("Rs. 2,50,000") == ["250000"]


def test_rrf_prefers_consensus():
    fused = rrf([[1, 2, 3], [3, 2, 9]])
    assert fused[0][0] in (2, 3) and {i for i, _ in fused} == {1, 2, 3, 9}


def test_calculator_is_safe():
    assert T.calculator("36000/3").data["value"] == 12000
    assert not T.calculator("__import__('os').system('echo hi')").ok
    assert not T.calculator("1/0").ok


def test_minigraph_loop_and_limit():
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"n": s.get("n", 0) + 1}); g.set_entry_point("a")
    g.add_conditional_edges("a", lambda s: "again" if s["n"] < 3 else "stop", {"again": "a", "stop": END})
    assert g.compile().invoke({})["n"] == 3
    g2 = StateGraph(dict); g2.add_node("a", lambda s: {}); g2.set_entry_point("a"); g2.add_edge("a", "a")
    try: g2.compile().invoke({}, {"recursion_limit": 5}); assert False
    except RuntimeError: pass


# ------------------------------------------------------------------ parsing / chunking (regressions found on real RBI PDFs)
def test_numbered_paragraphs_are_not_tables():
    r = H.regulatory_retriever()
    assert not any(c["text"].lstrip().startswith(("(1) |", "(2) |", "(3) |", "(4) |")) for c in r.chunks), "paragraph markers became table cells"
    para = next(c for c in r.chunks if "single counterparty" in c["text"])
    flat = " ".join(para["text"].split())
    assert para["kind"] == "prose" and "20 percent of the eligible capital base" in flat, (para["kind"], para["text"][:400])
    assert "not less than three years" in " ".join(c["text"] for c in r.chunks)        # sentence is not cut mid-way


def test_marker_line_is_merged_with_the_text_beside_it():
    """PyMuPDF layout: '(4)' and its paragraph text arrive as two separate lines at the same height."""
    from ingest.parse import Line, merge_marker_lines
    items = [(100.0, 72.0, Line("(4)", 10.5, False, 1)), (100.2, 108.0, Line("The tenure of the IO shall be a fixed term of", 10.5, False, 1)),
             (114.0, 108.0, Line("not less than three years.", 10.5, False, 1)), (130.0, 72.0, Line("(5)", 10.5, False, 1)),
             (130.0, 108.0, Line("The IO shall not exceed five years.", 10.5, False, 1))]
    out = [l.text for _, _, l in merge_marker_lines(items)]
    assert out == ["(4) The tenure of the IO shall be a fixed term of", "not less than three years.", "(5) The IO shall not exceed five years."], out
    assert [l.text for _, _, l in merge_marker_lines([(10.0, 72.0, Line("(a)", 10, False, 1))])] == ["(a)"]       # nothing beside it: unchanged


def test_cells_returned_as_separate_lines_are_rebuilt_into_table_rows():
    """PyMuPDF layout: each table cell is its own line; cells of one row differ by a fraction of a point in y and arrive in any order."""
    from ingest.parse import Line, merge_same_row
    L = lambda s: Line(s, 10.0, False, 1)
    items = [(200.2, 300.0, L("40 bps")), (200.0, 72.0, L("More than 30 per cent and up to 50 per cent")),
             (150.0, 72.0, L("Up to 15 per cent")), (150.1, 300.0, L("Nil")),
             (100.0, 72.0, L("A heading line"))]
    out = [l.text for _, _, l in merge_same_row(items)]
    assert out == ["A heading line", "Up to 15 per cent | Nil", "More than 30 per cent and up to 50 per cent | 40 bps"], out
    items2 = [(100.0, 72.0, L("(4)")), (100.2, 108.0, L("The tenure is a fixed term."))]
    assert [l.text for _, _, l in merge_same_row(items2)] == ["(4) The tenure is a fixed term."]
    assert [l.text for _, _, l in merge_same_row([(10.0, 72.0, L("Only line"))])] == ["Only line"]


def test_table_keeps_lead_in_and_wrapped_cells():
    r = H.regulatory_retriever()
    t = next(c for c in r.chunks if "40 bps" in c["text"])
    assert t["kind"] == "table" and t["text"].startswith("A bank shall make incremental provisions on unhedged"), (t["kind"], t["text"][:400])
    assert "More than 75 per cent" in t["text"] and "80 bps" in t["text"], t["text"][:600]              # whole table in one chunk


def test_retrieval_finds_the_right_sibling_document():
    r = H.regulatory_retriever()
    hits = r.search("single counterparty exposure limit", "hybrid_rerank", k=3, docs=["rbi_beta_lending_directions_2025"])
    assert hits and all(h.chunk["doc_id"] == "rbi_beta_lending_directions_2025" for h in hits)
    assert "15 percent" in hits[0].chunk["text"]


# ------------------------------------------------------------------ metrics
def test_retrieval_metrics():
    gold = [{"doc": "d1", "pages": [2]}, {"doc": "d2", "pages": [1]}]
    hits = [{"doc_id": "x", "page": 1}, {"doc_id": "d1", "page": 2}, {"doc_id": "d2", "page": 1}]
    s = M.retrieval_scores(hits, gold, ks=(1, 3))
    assert s["hit@1"] == 0 and s["hit@3"] == 1 and s["recall@3"] == 1 and s["mrr"] == 0.5


def test_refusal_scores():
    rows = [{"type": "unanswerable", "refused": True}, {"type": "unanswerable", "refused": False},
            {"type": "single", "refused": True}, {"type": "single", "refused": False}]
    r = M.refusal_scores(rows)
    assert r["refusal_precision"] == 0.5 and r["refusal_recall"] == 0.5 and r["false_refusals"] == 1


def test_wilson_and_sign_test():
    lo, hi = M.wilson(22, 26); assert 0.65 < lo < 0.72 and 0.93 < hi < 0.97
    assert M.sign_test_p(1, 0) == 1.0 and M.sign_test_p(0, 0) == 1.0
    assert M.sign_test_p(10, 0) < 0.01 and abs(M.sign_test_p(5, 5) - 1.0) < 1e-9


def test_partial_run_compares_only_common_questions():
    import eval.run_eval as R
    rows = []
    for sysname, ids in [("naive_rag", range(6)), ("hybrid_rerank_rag", range(6)), ("agent", range(3))]:
        for i in ids:
            rows.append({"system": sysname, "id": f"q{i}", "type": "single", "question": "", "gold_answer": "", "answer": "", "refused": False,
                         "correct": 1.0, "judge_reason": "", "faithfulness": 1.0, "citation_hit": 1.0, "citation_precision": 1.0,
                         "cited_contains_answer": 1.0, "tokens": 10, "latency_ms": 5, "steps": 2, "loops": 1, "tool_calls": [],
                         "terminated": True, "degraded": False, "n_citations": 1})
    common = set.intersection(*[{r["id"] for r in rows if r["system"] == s} for s in ("naive_rag", "hybrid_rerank_rag", "agent")])
    per = [r for r in rows if r["id"] in common]
    assert len(common) == 3 and len(per) == 9
    text = " ".join(R.findings(R.summarize(per), per))
    assert "3 questions that every system completed" in text and "numerical**: no questions" in text


def test_stratified_sampling_and_balanced_hand_grade_sample():
    from eval.run_eval import stratify
    from eval.hand_grade import pick_sample
    gold = [{"id": f"{t}{i}", "type": t} for t in ("single", "multihop", "numerical", "ambiguous", "unanswerable") for i in range(20)]
    s = stratify(gold, {"single": 3, "multihop": 4, "numerical": 2})
    assert [g["type"] for g in s].count("multihop") == 4 and len(s) == 9 and stratify(gold, {"single": 3, "multihop": 4, "numerical": 2}) == s
    rows = [{"id": g["id"], "type": g["type"], "system": sy, "correct": 1.0} for g in gold for sy in ("agent", "hybrid_rerank_rag", "naive_rag")]
    smp = pick_sample(rows)
    assert len(smp) == 20 and {r["system"] for r in smp} == {"agent", "hybrid_rerank_rag"}
    assert sum(r["type"] != "unanswerable" for r in smp) == 16 and len({(r["id"], r["system"]) for r in smp}) == 20


# ------------------------------------------------------------------ offline (rule-based) agent on the synthetic demo corpus
def test_offline_agent_end_to_end():
    from agent.graph import Agent
    from llm.base import OfflineLLM
    a = Agent(H.demo_retriever(), OfflineLLM())
    r = a.run("What is the helpline number for Kisan Samriddhi Yojana?")
    assert r["terminated"] and r["steps"] >= 4
    nf = a.run("What is the benefit amount under Pradhan Sagar Rakshak Yojana?")
    assert nf["refused"] and nf["loops"] <= 3
    cmp_ = a.run("Which has a higher outlay, Kisan Samriddhi Yojana or Saur Urja Kisan Yojana?")
    assert "higher" in cmp_["answer"] and len(cmp_["citations"]) == 2
    calc = a.run("What is the combined outlay of Kisan Samriddhi Yojana and Saur Urja Kisan Yojana?")
    assert any(t["tool"] == "calculator" and t["ok"] for t in calc["tool_calls"])


# ------------------------------------------------------------------ the shipped golden set
def test_golden_set_schema():
    from config import GOLDEN_PATH
    rows = [json.loads(l) for l in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert 60 <= len(rows) <= 100 and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert {"id", "type", "question", "gold_answer", "answer_keys", "gold_sources"} <= set(r)
        if r["type"] != "unanswerable": assert all(s["pages"] for s in r["gold_sources"])
        else: assert r["gold_sources"] == []


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
