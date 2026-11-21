"""python scripts/latency_bench.py [N=8]       COLD, UNTHROTTLED latency per system (spends tokens: no cache, no pause between calls).
Use these numbers - not the evaluation table's latency rows - when you quote latency."""
import os, sys
os.environ["LLM_NO_CACHE"] = "1"; os.environ["LLM_MIN_INTERVAL"] = "0"          # must be set before config is imported
sys.path.insert(0, ".")
import json, statistics
from config import GOLDEN_PATH
from llm import get_llm, get_control_llm
from retrieval.retriever import Retriever
from agent.graph import Agent
from agent.baselines import HybridRAG, NaiveRAG
from eval.run_eval import stratify

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
gold = [json.loads(l) for l in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
qs = stratify(gold, {"single": n // 4 or 1, "multihop": n // 4 or 1, "numerical": n // 4 or 1, "ambiguous": n // 4 or 1})
r, llm = Retriever(), get_llm()
for name, s in [("naive_rag", NaiveRAG(r, llm)), ("hybrid_rerank_rag", HybridRAG(r, llm)), ("agent", Agent(r, llm, get_control_llm()))]:
    ms, tok = [], []
    for g in qs:
        res = s.run(g["question"]); ms.append(res["latency_ms"]); tok.append(res["tokens"])
    ms.sort(); p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]
    print(f"{name:<20} n={len(ms)} p50={statistics.median(ms):8.0f} ms  p95={p95:8.0f} ms  mean tokens={statistics.mean(tok):7.0f}")
