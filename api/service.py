"""Framework-independent service layer used by both the FastAPI app and the Streamlit UI.
Adds: lazy init, LRU cache (protects free-tier rate limits), JSONL request logging, graceful degradation."""
from __future__ import annotations
import json, shutil, threading, time
from collections import OrderedDict
from pathlib import Path
from config import LOG_PATH, PDF_DIR, INDEX_DIR, LLM_PROVIDER
from llm import get_llm, get_control_llm

SYSTEMS = ("agent", "hybrid_rerank_rag", "naive_rag")


class Service:
    def __init__(self, cache_size: int = 256):
        self._lock = threading.Lock()
        self.retriever = self.llm = None
        self.systems: dict = {}
        self.cache: OrderedDict = OrderedDict(); self.cache_size = cache_size
        self.started = time.time()

    def load(self):
        from retrieval.retriever import Retriever
        from agent.graph import Agent
        from agent.baselines import NaiveRAG, HybridRAG
        if not (INDEX_DIR / "chunks.jsonl").exists():          # first boot (e.g. fresh container): build the index
            from ingest.build_index import build
            build()
        self.retriever = Retriever(); self.llm = get_llm()
        self.systems = {"agent": Agent(self.retriever, self.llm, get_control_llm()), "hybrid_rerank_rag": HybridRAG(self.retriever, self.llm),
                        "naive_rag": NaiveRAG(self.retriever, self.llm)}
        self.cache.clear()

    def ensure(self):
        if self.retriever is None:
            with self._lock:
                if self.retriever is None: self.load()

    def health(self) -> dict:
        self.ensure()
        return {"status": "ok", "uptime_s": round(time.time() - self.started), "llm": self.llm.name, **self.retriever.info()}

    def query(self, question: str, system: str = "agent") -> dict:
        self.ensure()
        system = system if system in SYSTEMS else "agent"
        key = (system, question.strip().lower())
        if key in self.cache:
            self.cache.move_to_end(key)
            return dict(self.cache[key], cached=True)
        res = self.systems[system].run(question.strip())
        res["cached"] = False
        self._log(res)
        self.cache[key] = res
        if len(self.cache) > self.cache_size: self.cache.popitem(last=False)
        return res

    def ingest(self, files: list[tuple[str, bytes]]) -> dict:
        """Save uploaded PDFs, rebuild the index, hot-reload."""
        from ingest.build_index import build
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in files:
            (PDF_DIR / Path(name).name).write_bytes(data)
        with self._lock:
            info = build(); self.load()
        return info

    def _log(self, res: dict):
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "system": res["system"], "question": res["question"], "answer": res["answer"],
               "refused": res["refused"], "latency_ms": res["latency_ms"], "tokens": res["tokens"], "steps": res["steps"],
               "n_citations": len(res["citations"]), "degraded": res["degraded"]}
        with open(LOG_PATH, "a") as f: f.write(json.dumps(rec) + "\n")


service = Service()
