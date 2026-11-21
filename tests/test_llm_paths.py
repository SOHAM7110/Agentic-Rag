"""Exercises the real-LLM code paths of the agent with a scripted fake model (no network, no API key).
Checks structure and robustness (never crashes, always terminates, degrades on LLM failure) - NOT answer quality.
Run: python -m tests.test_llm_paths   (uses whatever index is in data/index or $INDEX_DIR)"""
import tests.helpers as H   # noqa: F401  (forces dependency-free backends; must be first)
import json
from agent import prompts as P
from agent.graph import Agent
from llm.base import LLM, LLMError, LLMQuotaError, LLMResult


class FakeLLM(LLM):
    name = "fake"; offline = False

    def __init__(self, answer="The limit is 20 percent [1].", fail=False, subs=None, qtype="single", expr="0.05*500"):
        self.answer, self.fail, self.subs, self.qtype, self.expr = answer, fail, subs or [], qtype, expr
        self.calls = []

    def generate(self, system, user, json_mode=False, max_tokens=600):
        if self.fail: raise LLMError("rate limited")
        self.calls.append(system[:20])
        if system == P.PLAN_SYSTEM: out = {"type": self.qtype, "sub_questions": self.subs}
        elif system == P.GRADE_SYSTEM: out = {"sufficient": True, "reason": "ok"}
        elif system == P.REWRITE_SYSTEM: out = {"query": user.split("Question:")[-1][:80]}
        elif system == P.CALC_SYSTEM: out = {"expression": self.expr}
        elif system == P.VERIFY_SYSTEM:
            n = sum(1 for l in user.splitlines() if l[:3].strip().rstrip(".").isdigit() and "(cited" in l)
            out = {"results": [{"id": i, "supported": True} for i in range(1, n + 1)]}
        else: return LLMResult(self.answer, 500, 30)
        return LLMResult(json.dumps(out), 200, 20)


def _agent(llm):
    return Agent(H.regulatory_retriever(), llm)


def test_plain_question_with_citations():
    r = _agent(FakeLLM()).run("What is the single counterparty exposure limit?")
    assert not r["refused"] and r["citations"] and r["terminated"]
    assert [t["node"] for t in r["trace"]][:3] == ["plan", "retrieve", "grade"] and r["trace"][-1]["node"] == "verify"


def test_decomposition_runs_one_retrieval_per_sub_question():
    r = _agent(FakeLLM(qtype="compare", subs=["single counterparty limit", "group exposure limit"])).run("Compare the single and group limits")
    retr = [t for t in r["trace"] if t["node"] == "retrieve"][0]
    assert len(retr["data"]["queries"]) == 2


def test_numeric_uses_calculator():
    llm = FakeLLM(qtype="numeric", answer="Retention is 25 [1].", expr="0.05*500")
    r = _agent(llm).run("A bank has eligible capital of Rs 10000 crore. What is its maximum exposure to a single counterparty?")
    assert any(c["tool"] == "calculator" and c["ok"] for c in r["tool_calls"]) and not r["refused"]


def test_fullwidth_and_passage_style_citations_are_parsed():
    from agent import heuristics as H
    assert H.parse_cites("A \u30101\u3011 and B [Passage 2] and C [1, 3] and D \uff3b4\uff3d.") == [1, 2, 3, 4]
    assert H.strip_cites("Limit is 20 percent \u30101\u3011.") == "Limit is 20 percent."
    r = _agent(FakeLLM(answer="The limit is 20 percent \u30101\u3011.")).run("What is the single counterparty exposure limit?")
    assert not r["refused"] and r["citations"]


def test_uncited_answer_is_attributed_then_verified():
    r = _agent(FakeLLM(answer="The single counterparty exposure limit is a percent of the eligible capital base.")).run("What is the single counterparty exposure limit?")
    assert not r["refused"] and r["citations"]
    assert any("attributed" in t["summary"] for t in r["trace"] if t["node"] == "answer")


def test_model_can_refuse():
    r = _agent(FakeLLM(answer="INSUFFICIENT_EVIDENCE")).run("What is the crypto exposure limit?")
    assert r["refused"] and not r["citations"]


def test_llm_failure_degrades_instead_of_crashing():
    r = _agent(FakeLLM(fail=True)).run("What is the single counterparty exposure limit?")
    assert r["terminated"] and r["degraded"] and r["answer"]


def test_quota_error_stops_the_run_instead_of_degrading():
    class Quota(FakeLLM):
        def generate(self, *a, **k): raise LLMQuotaError("daily limit")
    try: _agent(Quota()).run("What is the single counterparty exposure limit?"); assert False, "should have raised"
    except LLMQuotaError: pass


def test_provider_aborts_on_long_429_and_retries_short_429():
    import http.server, os, threading, time
    mode = {"retry": "5000", "hits": 0}
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"])); mode["hits"] += 1
            if mode["retry"] and mode["hits"] <= (1 if mode["retry"] == "1" else 99):
                self.send_response(429); self.send_header("retry-after", mode["retry"]); self.end_headers(); self.wfile.write(b"{}"); return
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "OK"}}], "usage": {}}).encode())
        def log_message(self, *a): pass
    import http.server as hs
    srv = hs.HTTPServer(("127.0.0.1", 8766), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["GROQ_API_KEY"] = "x"
    import llm.providers as PR
    PR.PROVIDERS["groq"]["url"] = "http://127.0.0.1:8766"
    l = PR.OpenAICompatLLM("groq", "m", use_cache=False)
    t0 = time.time()
    try: l.generate("s", "u"); assert False
    except LLMQuotaError: assert time.time() - t0 < 3, "must abort immediately, not sleep"
    mode.update(retry="1", hits=0)
    assert l.generate("s", "u").text == "OK" and mode["hits"] == 2          # slept ~1s, then succeeded
    srv.shutdown()


def test_token_and_fallback_accounting_survives_state_copies():
    llm = FakeLLM()
    r = _agent(llm).run("What is the single counterparty exposure limit?")
    assert r["tokens"] >= 150 * len(llm.calls), (r["tokens"], len(llm.calls))        # every LLM call is counted, not just plan/retrieve
    bad = _agent(FakeLLM(fail=True)).run("What is the single counterparty exposure limit?")
    assert bad["degraded"] and "rate limited" in bad["llm_error"]


def test_hybrid_baseline_parses_fullwidth_citations():
    from agent.baselines import HybridRAG
    r = HybridRAG(H.regulatory_retriever(), FakeLLM(answer="The limit is 20 percent \u30101\u3011.")).run("What is the single counterparty exposure limit?")
    assert r["citations"] and "\u3010" not in r["answer"]


def test_entity_linking_separates_near_duplicate_documents():
    from agent.heuristics import link_entities
    docs = {k: {"title": v} for k, v in {
        "cb": "Rbi Commercial Banks Internal Ombudsman Directions 2026", "pb": "Rbi Payments Banks Internal Ombudsman Directions 2026",
        "sfb": "Rbi Small Finance Banks Internal Ombudsman Directions 2026", "cic": "Rbi Credit Information Companies Internal Ombudsman Directions 2026",
        "ir": "Rbi Commercial Banks Interest Rates On Advances Directions 2025", "kyc": "Rbi Commercial Banks Know Your Customer Directions 2025",
        "td": "Rbi Commercial Banks Transfer And Distribution Of Credit Risk Directions 2025"}.items()}
    assert set(link_entities("Is the minimum tenure for an Internal Ombudsman the same for payments banks and small finance banks?", docs)) == {"pb", "sfb"}
    assert set(link_entities("By how many days does escalation differ between Credit Information Companies and commercial banks under the Internal Ombudsman Directions?", docs)) == {"cic", "cb"}
    assert link_entities("How often must a bank review and publish its MCLR?", docs) == []


def test_grader_and_answerer_see_the_same_passages():
    seen = {}
    class Spy(FakeLLM):
        def generate(self, system, user, json_mode=False, max_tokens=600):
            if system == P.GRADE_SYSTEM: seen["grade"] = user.count("\n[")
            if system == P.ANSWER_SYSTEM: seen["answer"] = user.count("\n[")
            return super().generate(system, user, json_mode, max_tokens)
    _agent(Spy(qtype="compare", subs=["single counterparty exposure limit", "group of connected counterparties exposure limit"])).run("Compare the single and group limits")
    assert seen["grade"] == seen["answer"] and seen["answer"] >= 1, seen


def test_retry_stops_when_a_rewrite_finds_nothing_new():
    class NeverSufficient(FakeLLM):
        def generate(self, system, user, json_mode=False, max_tokens=600):
            if system == P.GRADE_SYSTEM: return LLMResult(json.dumps({"sufficient": False, "reason": "no minimum stated"}), 100, 10)
            return super().generate(system, user, json_mode, max_tokens)
    r = _agent(NeverSufficient()).run("What is the single counterparty exposure limit?")
    assert r["refused"] and r["loops"] == 2 and "no new evidence" in r["answer"]      # tiny corpus: loop 2 adds nothing -> stop early


def test_rewrite_is_told_what_is_missing_and_never_mentions_scheme_terms():
    seen = {}
    class Spy(FakeLLM):
        def generate(self, system, user, json_mode=False, max_tokens=600):
            if system == P.REWRITE_SYSTEM: seen["user"] = user
            if system == P.GRADE_SYSTEM: return LLMResult(json.dumps({"sufficient": False, "reason": "does not state the review frequency"}), 100, 10)
            return super().generate(system, user, json_mode, max_tokens)
    _agent(Spy()).run("How often must the risk categorisation be reviewed?")
    assert "does not state the review frequency" in seen["user"]
    for scheme_word in ("programme outlay", "benefit amount", "nodal ministry", "minimum age"):
        assert scheme_word not in P.REWRITE_SYSTEM


def test_rewrite_never_repeats_a_query_and_stops_when_it_cannot_change_it():
    class Same(FakeLLM):
        reason = "does not state the review frequency"
        def generate(self, system, user, json_mode=False, max_tokens=600):
            if system == P.GRADE_SYSTEM: return LLMResult(json.dumps({"sufficient": False, "reason": self.reason}), 100, 10)
            if system == P.REWRITE_SYSTEM:      # a model that just echoes the previous query
                prev = [l for l in user.splitlines() if l.startswith("Previous query:")][0].split(":", 1)[1].strip()
                return LLMResult(json.dumps({"query": prev}), 100, 10)
            return super().generate(system, user, json_mode, max_tokens)
    r = _agent(Same()).run("How often must the risk categorisation be reviewed?")
    qs = [t["data"]["queries"][0] for t in r["trace"] if t["node"] == "retrieve"]
    assert len(set(qs)) == len(qs), qs                          # no query is ever searched twice
    class Mute(Same): reason = ""
    r2 = _agent(Mute()).run("How often must the risk categorisation be reviewed?")
    assert r2["refused"] and r2["loops"] == 1 and "no new query" in r2["answer"], (r2["loops"], r2["answer"])


def test_control_model_does_the_control_steps_and_main_model_only_answers():
    main, ctl = FakeLLM(), FakeLLM()
    r = Agent(H.regulatory_retriever(), main, ctl).run("What is the single counterparty exposure limit?")
    assert not r["refused"] and len(main.calls) == 1 and len(ctl.calls) >= 3, (main.calls, ctl.calls)


def test_dense_grounded_baseline_uses_dense_retrieval_and_the_grounded_prompt():
    from agent.baselines import DenseGroundedRAG
    r = DenseGroundedRAG(H.regulatory_retriever(), FakeLLM(answer="INSUFFICIENT_EVIDENCE")).run("What is the crypto limit?")
    assert r["system"] == "dense_grounded_rag" and r["refused"] and "dense" in r["trace"][0]["summary"]


def test_loops_are_bounded():
    class NeverSufficient(FakeLLM):
        def generate(self, system, user, json_mode=False, max_tokens=600):
            if system == P.GRADE_SYSTEM: return LLMResult(json.dumps({"sufficient": False, "reason": "no"}), 100, 10)
            return super().generate(system, user, json_mode, max_tokens)
    r = _agent(NeverSufficient()).run("What is the single counterparty exposure limit?")
    assert r["loops"] <= 3 and r["terminated"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
