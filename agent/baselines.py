"""Baselines for the three-way comparison: (1) naive RAG = dense top-k + answer; (2) hybrid+rerank single shot
with a grounded/refusal prompt. Same result schema as Agent.run so one evaluator scores all three."""
from __future__ import annotations
import re, time
from llm.base import LLMError
from retrieval.text import approx_tokens, tokenize
from . import heuristics as H, prompts as P

NAIVE_SYSTEM = "Answer the question using the context passages. Be concise."


class _Base:
    system = "base"

    def __init__(self, retriever, llm):
        self.r, self.llm = retriever, llm

    def _finish(self, q, t0, answer, refused, passages, cites_n, tokens, trace, degraded=False):
        cites = []
        for n in sorted(set(cites_n)):
            if 0 < n <= len(passages):
                ch = passages[n - 1]
                cites.append({"n": n, "doc": ch["doc_id"], "doc_title": ch["doc_title"], "section": ch["section"],
                              "page": ch["page"], "page_end": ch["page_end"], "chunk_id": ch["id"], "text": ch["text"][:400]})
        return {"system": self.system, "question": q, "answer": answer, "refused": refused, "citations": cites,
                "trace": trace, "tokens": tokens, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "steps": len(trace), "loops": 1, "tool_calls": [], "terminated": True, "degraded": degraded,
                "retrieved": [c["id"] for c in passages]}


class NaiveRAG(_Base):
    system = "naive_rag"
    MODE, K = "dense", 3

    def run(self, q: str) -> dict:
        t0 = time.perf_counter()
        hits = self.r.search(q, self.MODE, k=self.K)
        passages = [h.chunk for h in hits]
        trace = [{"step": 1, "node": "retrieve", "summary": f"{self.MODE} top-{self.K}", "ms": 0, "tokens": 0, "data": {"hits": [c["id"] for c in passages]}}]
        prompt = P.format_passages(passages)
        tokens = approx_tokens(NAIVE_SYSTEM + q + prompt); degraded = False
        text, cites = None, []
        if not self.llm.offline:
            try:
                res = self.llm.generate(NAIVE_SYSTEM, f"Question: {q}\n\nContext:\n{prompt}")
                text = res.text.strip(); tokens = res.in_tokens + res.out_tokens
                cites = [int(n) for n in re.findall(r"\[(\d+)\]", text)] or list(range(1, len(passages) + 1))
            except LLMError:
                degraded = True
        if text is None:
            best = H.best_sentences(q, passages, self.r, set(), 3)
            if best:
                text = " ".join(s for _, s, _ in best); cites = [passages.index(ch) + 1 for _, _, ch in best]
            else:
                text = "No answer."
        tokens += approx_tokens(text)
        trace.append({"step": 2, "node": "answer", "summary": "answer (no refusal logic)", "ms": 0, "tokens": tokens, "data": {"answer": text}})
        return self._finish(q, t0, H.strip_cites(text), False, passages, cites, tokens, trace, degraded)


class HybridRAG(_Base):
    system = "hybrid_rerank_rag"
    K = 5
    COVER_MIN = 0.45   # fixed a-priori heuristic for the offline backend (chosen from the coverage distribution; see README)

    def run(self, q: str) -> dict:
        t0 = time.perf_counter()
        hits = self.r.search(q, "hybrid_rerank", k=self.K)
        passages = [h.chunk for h in hits]
        trace = [{"step": 1, "node": "retrieve", "summary": f"hybrid + rerank top-{self.K}", "ms": 0, "tokens": 0, "data": {"hits": [c["id"] for c in passages]}}]
        prompt = P.format_passages(passages)
        tokens = approx_tokens(P.ANSWER_SYSTEM + q + prompt); degraded = False
        qt = tokenize(q)
        best_cov = max((self.r.coverage(qt, c) for c in passages), default=0.0)
        text, refused, cites = None, False, []
        if not self.llm.offline:
            try:
                res = self.llm.generate(P.ANSWER_SYSTEM, f"Question: {q}\n\nPassages:\n{prompt}")
                tokens = res.in_tokens + res.out_tokens; text = res.text.strip()
                if "INSUFFICIENT_EVIDENCE" in text: refused = True
                else: cites = [int(n) for n in re.findall(r"\[(\d+)\]", text)]
            except LLMError:
                degraded = True; text = None
        if text is None:
            if best_cov < self.COVER_MIN: refused = True
            else:
                best = H.best_sentences(q, passages, self.r, set(), 3)
                if best: text = " ".join(s for _, s, _ in best); cites = [passages.index(ch) + 1 for _, _, ch in best]
                else: refused = True
        if refused:
            text = "I can't answer this from the provided documents."; cites = []
        tokens += approx_tokens(text)
        trace.append({"step": 2, "node": "answer", "summary": "grounded answer" + (" (refused)" if refused else ""), "ms": 0, "tokens": tokens, "data": {"coverage": round(best_cov, 2)}})
        return self._finish(q, t0, H.strip_cites(text) if not refused else text, refused, passages, cites, tokens, trace, degraded)
