"""Run with `pytest -q` or `python -m tests.test_core` (no pytest needed)."""
import json
from agent import tools as T
from agent.minigraph import StateGraph, END
from eval import metrics as M
from retrieval.fusion import rrf
from retrieval.text import split_sentences, tokenize


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


def test_agent_end_to_end():
    from api.service import service
    a = service.query("What is the helpline number for Kisan Samriddhi Yojana?")
    assert a["terminated"] and a["steps"] >= 4 and a["citations"]
    nf = service.query("What is the benefit amount under Pradhan Sagar Rakshak Yojana?")
    assert nf["refused"] and nf["loops"] <= 3               # unknown scheme -> refuse, loop bounded
    cmp_ = service.query("Which has a higher outlay, Kisan Samriddhi Yojana or Saur Urja Kisan Yojana?")
    assert "higher" in cmp_["answer"] and len(cmp_["citations"]) == 2
    calc = service.query("What is the combined outlay of Kisan Samriddhi Yojana and Saur Urja Kisan Yojana?")
    assert any(t["tool"] == "calculator" and t["ok"] for t in calc["tool_calls"])


def test_golden_set_schema():
    from config import GOLDEN_PATH
    rows = [json.loads(l) for l in GOLDEN_PATH.read_text().splitlines()]
    assert 60 <= len(rows) <= 100
    for r in rows:
        assert {"id", "type", "question", "gold_answer", "answer_keys", "gold_sources"} <= set(r)
        if r["type"] != "unanswerable": assert all(s["pages"] for s in r["gold_sources"])


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
