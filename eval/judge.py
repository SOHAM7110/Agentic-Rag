"""LLM-as-judge with a written rubric (prompts in agent/prompts.py). With JUDGE_PROVIDER=offline a rule-based judge
is used so the harness runs anywhere; use a *different* model than the generator for real numbers."""
from __future__ import annotations
import json, re
from agent import heuristics as H, prompts as P
from agent.graph import parse_json
from llm.base import LLMError
from retrieval.retriever import ctx_text
from .metrics import has_key, norm


class Judge:
    def __init__(self, llm):
        self.llm = llm
        self.name = llm.name

    # ---------------------------------------------------------------- correctness
    def correctness(self, item: dict, res: dict) -> dict:
        if item["type"] == "unanswerable":                    # gold behaviour is a refusal: no LLM needed
            ok = res["refused"]
            return {"score": 1.0 if ok else 0.0, "reason": "refused (correct)" if ok else "answered an unanswerable question"}
        if res["refused"]:
            return {"score": 0.0, "reason": "refused an answerable question"}
        if not self.llm.offline:
            user = f"Question: {item['question']}\nGold answer: {item['gold_answer']}\nCandidate answer: {res['answer']}"
            try:
                out = parse_json(self.llm.generate(P.JUDGE_CORRECT_SYSTEM, user, json_mode=True, max_tokens=200).text)
                if out is not None: return {"score": float(out.get("score", 0)), "reason": out.get("reason", "")}
            except LLMError:
                pass                                           # fall through to rules
        return self._rule(item, res["answer"])

    @staticmethod
    def _rule(item: dict, ans: str) -> dict:
        a = ans.strip()
        if item.get("answer_prefix") and not re.match(r"\W*" + item["answer_prefix"] + r"\b", a.lower()):
            return {"score": 0.0, "reason": f"should start with '{item['answer_prefix']}'"}
        if item.get("winner"):                                 # 'which is higher' -> winner must be named in the verdict sentence
            sents = [s for s in re.split(r"(?<=[.;])\s+", a) if re.search(r"\b(higher|larger|greater|bigger|more|earlier|later|lower|smaller|less)\b", s, re.I)]
            if not sents: return {"score": 0.0, "reason": "no verdict sentence"}
            s = norm(sents[0]); w, l = s.find(item["winner"]), s.find(item["loser"])
            ok = w >= 0 and (l < 0 or w < l)
            return {"score": 1.0 if ok else 0.0, "reason": "winner named first" if ok else "wrong or missing winner"}
        missing = [k for k in item["answer_keys"] if not has_key(a, k)]
        return {"score": 0.0 if missing else 1.0, "reason": f"missing {missing}" if missing else "all key facts present"}

    # ---------------------------------------------------------------- faithfulness
    def faithfulness(self, res: dict, chunks_by_id: dict) -> dict:
        if res["refused"]:
            return {"score": None, "reason": "refusal (no claims)"}
        ctx = [ctx_text(chunks_by_id[i]) for i in res["retrieved"] if i in chunks_by_id]
        if not self.llm.offline:
            user = f"Context:\n" + "\n".join(f"- {c}" for c in ctx) + f"\n\nAnswer: {res['answer']}"
            try:
                out = parse_json(self.llm.generate(P.JUDGE_FAITH_SYSTEM, user, json_mode=True, max_tokens=100).text)
                if out and out.get("total"): return {"score": min(1.0, out["supported"] / out["total"]), "reason": "llm"}
            except LLMError:
                pass
        claims = [s for s in re.split(r"(?<=[.;!?])\s+", H.strip_cites(res["answer"])) if s.strip()]
        if not claims: return {"score": 0.0, "reason": "empty"}
        sup = 0
        for c in claims:
            if H.claim_support(c, ctx) >= 0.7: sup += 1
            elif self._computed_ok(c, ctx): sup += 1
        return {"score": sup / len(claims), "reason": "rules"}

    @staticmethod
    def _computed_ok(claim: str, ctx: list[str]) -> bool:
        """A calculated number is faithful if it is derivable from numbers in the context: accept claims whose
        'computed as a/b' expression has all operands in the context."""
        m = re.search(r"computed as ([0-9+\-*/. ()]+)\)", claim)
        if not m: return False
        nums = re.findall(r"\d+(?:\.\d+)?", m.group(1))
        text = norm(" ".join(ctx))
        return all(re.search(r"(?<!\d)" + re.escape(n) + r"(?!\d)", text) for n in nums)
