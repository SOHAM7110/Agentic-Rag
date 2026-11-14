"""Small, dependency-free BM25 (Okapi) with an inverted index and optional document mask."""
from __future__ import annotations
import math
from collections import Counter, defaultdict
import numpy as np
from .text import tokenize


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(t) for t in texts]
        self.N = len(texts)
        self.dl = np.array([len(t) for t in self.toks], dtype="float32")
        self.avgdl = float(self.dl.mean()) if self.N else 1.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, ts in enumerate(self.toks):
            for t, c in Counter(ts).items():
                self.post[t].append((i, c))
        self.idf_map = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}
        self.max_idf = math.log(1 + (self.N + 0.5) / 0.5)

    def idf(self, tok: str) -> float:
        return self.idf_map.get(tok, self.max_idf)   # unseen token = maximally informative

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.N, dtype="float32")
        for t in set(tokenize(query)):
            for i, tf in self.post.get(t, ()):
                s[i] += self.idf_map[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s

    def search(self, query: str, k: int, mask=None):
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, -1.0)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]
