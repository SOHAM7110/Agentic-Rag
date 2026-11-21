"""python scripts/diagnose.py [N=6] [--ids r2_001,r1_004]   (run from the project root)
Runs the full agent on a few golden questions and prints WHY each answer/refusal happened:
every graph step (plan type, sub-questions, coverage, grade verdict + reason), tool calls, LLM fallbacks and errors."""
import json, sys
sys.path.insert(0, ".")
from config import GOLDEN_PATH
from llm import get_llm, get_control_llm
from retrieval.retriever import Retriever
from agent.graph import Agent
from eval.run_eval import spread

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ids = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--ids=")), None)
gold = [json.loads(l) for l in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [g for g in gold if g["id"] in ids] if ids else spread(gold, int(args[0]) if args else 6)
agent = Agent(Retriever(), get_llm(), get_control_llm())
print("LLM:", agent.llm.name, "| facts table populated:", agent.has_facts)
for g in rows:
    r = agent.run(g["question"])
    print("\n" + "=" * 100)
    print(f"[{g['id']}] ({g['type']}) {g['question']}")
    print(f"GOLD: {g['gold_answer'][:160]}")
    print(f"GOT : {r['answer'][:260]}")
    print(f"refused={r['refused']} loops={r['loops']} degraded={r['degraded']} tokens={r['tokens']} citations={[(c['doc'][:28], c['page']) for c in r['citations']]}")
    if r.get("llm_error"): print("LLM ERROR:", r["llm_error"])
    for t in r["trace"]:
        extra = ""
        if t["node"] == "retrieve": extra = "  coverage=" + str([(c["coverage"], c["aspect_ok"]) for c in t["data"].get("coverage", [])])
        print(f"   {t['step']:>2}. {t['node']:<8} {t['summary'][:170]}{extra}")
        for dc in t.get("data", {}).get("dropped_claims", []): print(f"        DROPPED CLAIM: {dc}")
        if t["node"] == "verify" and t.get("data", {}).get("llm_flags") is not None: print(f"        verifier verdicts: {t['data']['llm_flags']}")
