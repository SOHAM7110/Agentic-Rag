"""python scripts\\check_linking.py [ids]      (free: no LLM, no API quota)
For every golden question that NAMES documents, shows which documents the agent links, the topic-only query it will use inside each
document, and whether the gold page lands in the top-3 of that per-document search - with YOUR real embedder and reranker."""
import json, sys
sys.path.insert(0, ".")
from config import GOLDEN_PATH
from retrieval.retriever import Retriever
from agent.heuristics import link_entities, strip_entity_words
from eval import metrics as M

want = set(sys.argv[1:])
r = Retriever(); print(r.info())
gold = [json.loads(l) for l in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
tot = ok = linked = 0
for g in gold:
    ents = link_entities(g["question"], r.docs)
    if want and g["id"] not in want: continue
    if not ents: continue
    linked += 1
    q = strip_entity_words(g["question"], ents, r.docs)
    print(f"\n[{g['id']}] {g['question'][:100]}\n   linked: {[e[-48:] for e in ents]}\n   per-document query: {q}")
    for e in ents:
        gs = [s for s in g["gold_sources"] if s["doc"] == e]
        hits = r.search(q, "hybrid_rerank", k=3, docs=[e])
        flags = [M.relevant(h.chunk, gs) is not None for h in hits] if gs else None
        if gs: tot += 1; ok += any(flags)
        print(f"   in ...{e[-34:]}: gold page in top-3 = {any(flags) if gs else 'n/a (not a gold doc!)'} {flags or ''}")
print(f"\nquestions with linked documents: {linked} | (question, doc) pairs with gold page in top-3: {ok}/{tot}")
