"""Hybrid retriever: dense + BM25 -> RRF fusion -> cross-encoder rerank. One class, four modes (for the ablation)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from config import INDEX_DIR, CANDIDATES, TOP_K
from .bm25 import BM25
from .embed import load_embedder
from .fusion import rrf
from .rerank import make_reranker
from .store import make_store

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    idx: int
    score: float
    chunk: dict

    def cite(self) -> dict:
        c = self.chunk
        return {"chunk_id": c["id"], "doc": c["doc_id"], "doc_title": c["doc_title"], "section": c["section"], "page": c["page"]}


def ctx_text(c: dict) -> str:
    """Text actually indexed: chunk text prefixed with its document title and section."""
    return f"{c['doc_title']} | {c['section']} | {c['text']}"


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        d = Path(index_dir)
        self.chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text().splitlines() if l.strip()]
        self.facts = json.loads((d / "facts.json").read_text())
        self.docs = json.loads((d / "docs.json").read_text())
        self.embedder = load_embedder(d)
        self.store = make_store(d); self.store.load()
        self.bm25 = BM25([ctx_text(c) for c in self.chunks])
        self.reranker = make_reranker(self.bm25.idf)
        self.doc_index: dict[str, np.ndarray] = {}
        ids = np.array([c["doc_id"] for c in self.chunks])
        for did in self.docs:
            self.doc_index[did] = ids == did
        self._qcache: dict[str, np.ndarray] = {}

    def info(self) -> dict:
        return {"chunks": len(self.chunks), "docs": len(self.docs), "embedder": self.embedder.name,
                "store": self.store.name, "reranker": self.reranker.name}

    def _mask(self, doc_ids):
        if not doc_ids: return None
        m = np.zeros(len(self.chunks), dtype=bool)
        for d in doc_ids: m |= self.doc_index[d]
        return m

    def _qvec(self, q: str):
        if q not in self._qcache:
            self._qcache[q] = self.embedder.encode([q], query=True)[0]
        return self._qcache[q]

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = TOP_K, docs: list[str] | None = None) -> list[Hit]:
        mask = self._mask(docs)
        n = max(CANDIDATES, k)
        if mode == "dense":
            res = self.store.search(self._qvec(query), k, mask)
        elif mode == "bm25":
            res = self.bm25.search(query, k, mask)
        else:
            dense = self.store.search(self._qvec(query), n, mask)
            sparse = self.bm25.search(query, n, mask)
            fused = rrf([[i for i, _ in dense], [i for i, _ in sparse]])[:n]
            if mode == "hybrid":
                res = fused[:k]
            else:
                cands = [(i, ctx_text(self.chunks[i])) for i, _ in fused]
                res = self.reranker.rerank(query, cands)[:k]
        return [Hit(i, s, self.chunks[i]) for i, s in res]

    # --- helpers used by the agent -------------------------------------------------
    def coverage(self, query_tokens: list[str], chunk: dict, ignore: set[str] = frozenset()) -> float:
        """IDF-weighted fraction of query tokens present in the chunk (ignoring e.g. entity-name tokens)."""
        from .text import tokenize
        toks = set(tokenize(ctx_text(chunk)))
        q = [t for t in set(query_tokens) if t not in ignore]
        tot = sum(self.bm25.idf(t) for t in q)
        if tot == 0: return 1.0
        return sum(self.bm25.idf(t) for t in q if t in toks) / tot
