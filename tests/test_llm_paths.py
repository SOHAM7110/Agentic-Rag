"""Exercises the real-LLM code paths of the agent with a scripted fake model (no network, no API key).
Checks structure and robustness (never crashes, always terminates, degrades on LLM failure) - NOT answer quality.
Run: python -m tests.test_llm_paths   (uses whatever index is in data/index or $INDEX_DIR)"""
import json
from agent import prompts as P
from agent.graph import Agent
from llm.base import LLM, LLMError, LLMResult


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
    from retrieval.retriever import Retriever
    return Agent(Retriever(), llm)


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
    r = _agent(llm).run("A bank securitises Rs 500 crore of 18-month loans. What retention?")
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
