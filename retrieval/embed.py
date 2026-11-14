"""Dense embedders. Preferred: sentence-transformers (bge-small). Fallback (no model download needed):
LSA = TF-IDF + truncated SVD fitted on the corpus. Both return L2-normalised vectors."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from config import EMBEDDER, EMBED_MODEL
from .text import tokenize

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


class STEmbedder:
    def __init__(self, model: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.name = f"st:{model}"
        self.model = SentenceTransformer(model)

    def fit(self, texts): pass

    def encode(self, texts, query: bool = False):
        if query and "bge" in self.model_name.lower():
            texts = [BGE_PREFIX + t for t in texts]
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype="float32")

    def save(self, d: Path): (d / "embedder.pkl").write_bytes(pickle.dumps({"kind": "st", "model": self.model_name}))


class LSAEmbedder:
    name = "lsa(tfidf+svd)"

    def __init__(self, dims: int = 160):
        self.dims, self.vec, self.svd = dims, None, None

    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        k = max(2, min(self.dims, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)

    def encode(self, texts, query: bool = False):
        Z = self.svd.transform(self.vec.transform(list(texts)))
        Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return Z.astype("float32")

    def save(self, d: Path): (d / "embedder.pkl").write_bytes(pickle.dumps({"kind": "lsa", "obj": self}))


def make_embedder():
    if EMBEDDER in ("auto", "st"):
        try:
            return STEmbedder()
        except Exception as e:  # not installed / no network
            if EMBEDDER == "st":
                raise
            print(f"[embed] sentence-transformers unavailable ({type(e).__name__}); using LSA fallback")
    return LSAEmbedder()


def load_embedder(d: Path):
    meta = pickle.loads((d / "embedder.pkl").read_bytes())
    return STEmbedder(meta["model"]) if meta["kind"] == "st" else meta["obj"]
