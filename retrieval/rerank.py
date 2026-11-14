"""Rerankers. Preferred: cross-encoder (ms-marco MiniLM). Fallback: a lexical reranker using IDF-weighted
term coverage + bigram overlap + heading overlap (so the pipeline and the ablation run with no model download)."""
from __future__ import annotations
from config import RERANKER, RERANK_MODEL
from .text import tokenize


class CrossEncoderReranker:
    def __init__(self, model: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self.name = f"cross-encoder:{model}"
        self.m = CrossEncoder(model)

    def rerank(self, query: str, cands: list[tuple[int, str]]) -> list[tuple[int, float]]:
        if not cands: return []
        sc = self.m.predict([(query, t) for _, t in cands])
        return sorted(((i, float(s)) for (i, _), s in zip(cands, sc)), key=lambda x: -x[1])


class LexicalReranker:
    name = "lexical-rerank"

    def __init__(self, idf):
        self.idf = idf

    def rerank(self, query: str, cands: list[tuple[int, str]]) -> list[tuple[int, float]]:
        q = tokenize(query)
        if not q: return [(i, 0.0) for i, _ in cands]
        qset = set(q)
        qbi = set(zip(q, q[1:]))
        tot = sum(self.idf(t) for t in qset) or 1.0
        out = []
        for i, text in cands:
            toks = tokenize(text)
            tset = set(toks)
            cov = sum(self.idf(t) for t in qset if t in tset) / tot
            bi = len(qbi & set(zip(toks, toks[1:]))) / (len(qbi) or 1)
            # density: shorter chunks that contain the terms are more focused
            dens = 1.0 / (1.0 + len(toks) / 120.0)
            out.append((i, 0.75 * cov + 0.2 * bi + 0.05 * dens))
        return sorted(out, key=lambda x: -x[1])


def make_reranker(idf):
    if RERANKER in ("auto", "cross"):
        try:
            return CrossEncoderReranker()
        except Exception as e:
            if RERANKER == "cross": raise
            print(f"[rerank] cross-encoder unavailable ({type(e).__name__}); using lexical reranker")
    return LexicalReranker(idf)
