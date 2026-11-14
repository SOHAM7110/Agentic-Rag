"""Vector stores: ChromaDB (preferred) or an in-process numpy store (zero-dependency fallback)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from config import VECTOR_STORE


class NumpyStore:
    name = "numpy"

    def __init__(self, path: Path): self.path, self.V = Path(path), None

    def build(self, chunks, vectors):
        self.V = vectors; np.save(self.path / "vectors.npy", vectors)

    def load(self): self.V = np.load(self.path / "vectors.npy")

    def search(self, qvec, k, mask=None):
        s = self.V @ qvec
        if mask is not None:
            s = np.where(mask, s, -1e9)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx if s[i] > -1e8]


class ChromaStore:
    name = "chroma"

    def __init__(self, path: Path):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(Path(path) / "chroma"))
        self.col = None
        self.doc_ids = None

    def build(self, chunks, vectors):
        try: self.client.delete_collection("chunks")
        except Exception: pass
        self.col = self.client.create_collection("chunks", metadata={"hnsw:space": "cosine"})
        for s in range(0, len(chunks), 500):
            part = chunks[s:s + 500]
            self.col.add(ids=[str(s + i) for i in range(len(part))], embeddings=vectors[s:s + 500].tolist(),
                         documents=[c["text"] for c in part],
                         metadatas=[{"doc_id": c["doc_id"], "page": c["page"], "section": c["section"]} for c in part])

    def load(self): self.col = self.client.get_collection("chunks")

    def search(self, qvec, k, mask=None):
        n = self.col.count()
        res = self.col.query(query_embeddings=[qvec.tolist()], n_results=n if mask is not None else min(k, n))
        out = []
        for i, d in zip(res["ids"][0], res["distances"][0]):
            if mask is None or mask[int(i)]:
                out.append((int(i), 1.0 - float(d)))
        return out[:k]


def make_store(path: Path):
    if VECTOR_STORE in ("auto", "chroma"):
        try:
            return ChromaStore(path)
        except Exception as e:
            if VECTOR_STORE == "chroma": raise
            print(f"[store] chromadb unavailable ({type(e).__name__}); using numpy store")
    return NumpyStore(path)
