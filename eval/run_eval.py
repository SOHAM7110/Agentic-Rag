"""python -m eval.run_eval   (or `make eval`)
Runs: (1) retrieval ablation, (2) three-system comparison (naive RAG / hybrid+rerank / full agent) with LLM-judge
correctness + faithfulness, citation accuracy, refusal P/R, agent + ops metrics. Writes results/ and refreshes the README."""
from __future__ import annotations
import json, re, sys, time
from collections import defaultdict
from config import GOLDEN_PATH, RESULTS_DIR, LLM_PROVIDER, LLM_MODEL, JUDGE_PROVIDER, JUDGE_MODEL, ROOT
from llm import get_llm
from retrieval.retriever import Retriever, MODES
from agent.graph import Agent, ENGINE
from agent.baselines import NaiveRAG, HybridRAG
from .judge import Judge
from . import metrics as M

TYPES = ["single", "multihop", "numerical", "unanswerable", "ambiguous"]
SYS_LABEL = {"naive_rag": "Naive RAG (dense top-3)", "hybrid_rerank_rag": "Hybrid + rerank (single shot)", "agent": "Full agent"}
MODE_LABEL = {"dense": "Dense only", "bm25": "BM25 only", "hybrid": "Hybrid (RRF)", "hybrid_rerank": "Hybrid + rerank"}


def load_golden():
    return [json.loads(l) for l in GOLDEN_PATH.read_text().splitlines() if l.strip()]


def retrieval_ablation(r: Retriever, gold):
    rows = defaultdict(list)
    ans = [g for g in gold if g["type"] != "unanswerable"]
    for mode in MODES:
        for g in ans:
            hits = [h.chunk for h in r.search(g["question"], mode, k=10)]
            rows[mode].append(M.retrieval_scores(hits, g["gold_sources"], ks=(1, 3, 5)))
    return {m: {k: M.mean([x[k] for x in v]) for k in v[0]} for m, v in rows.items()}, len(ans)


def run_systems(r, gold, gen_llm, judge, names=None):
    systems = {"naive_rag": NaiveRAG(r, gen_llm), "hybrid_rerank_rag": HybridRAG(r, gen_llm), "agent": Agent(r, gen_llm)}
    systems = {k: v for k, v in systems.items() if not names or k in names}
    by_id = {c["id"]: c for c in r.chunks}
    per_q = []
    for name, s in systems.items():
        for g in gold:
            res = s.run(g["question"])
            corr = judge.correctness(g, res); faith = judge.faithfulness(res, by_id)
            for c in res["citations"]: c["full_text"] = by_id[c["chunk_id"]]["text"]
            cit = M.citation_scores(res["citations"], g["gold_sources"], g["answer_keys"]) if g["type"] != "unanswerable" and not res["refused"] else \
                  {"citation_hit": 0.0, "citation_precision": 0.0, "cited_contains_answer": 0.0}
            per_q.append({"system": name, "id": g["id"], "type": g["type"], "question": g["question"], "gold_answer": g["gold_answer"],
                          "answer": res["answer"], "refused": res["refused"], "correct": corr["score"], "judge_reason": corr["reason"],
                          "faithfulness": faith["score"], **cit, "tokens": res["tokens"], "latency_ms": res["latency_ms"], "steps": res["steps"],
                          "loops": res["loops"], "tool_calls": res["tool_calls"], "terminated": res["terminated"], "degraded": res["degraded"],
                          "llm_error": res.get("llm_error", ""), "trace": [f"{t['node']}: {t['summary'][:160]}" for t in res["trace"]],
                          "n_citations": len(res["citations"])})
        print(f"[eval] {name} done")
    return per_q


def summarize(per_q):
    S = {}
    for name in [s for s in SYS_LABEL if any(r["system"] == s for r in per_q)]:
        rows = [r for r in per_q if r["system"] == name]
        ans = [r for r in rows if r["type"] != "unanswerable"]
        faith = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
        calls = [c for r in rows for c in r["tool_calls"]]
        d = {"n": len(rows), "correctness": M.mean([r["correct"] for r in rows]),
             "correctness_answerable": M.mean([r["correct"] for r in ans]),
             "faithfulness": M.mean(faith),
             "citation_hit": M.mean([r["citation_hit"] for r in ans]),
             "cited_contains_answer": M.mean([r["cited_contains_answer"] for r in ans]),
             **M.refusal_scores(rows),
             "avg_steps": M.mean([r["steps"] for r in rows]), "avg_loops": M.mean([r["loops"] for r in rows]),
             "tool_success": (sum(c["ok"] for c in calls) / len(calls)) if calls else None,
             "termination": M.mean([1.0 if r["terminated"] else 0.0 for r in rows]),
             "p50_ms": M.pct([r["latency_ms"] for r in rows], 0.5), "p95_ms": M.pct([r["latency_ms"] for r in rows], 0.95),
             "tokens": M.mean([r["tokens"] for r in rows]), "degraded": sum(r["degraded"] for r in rows),
             "by_type": {t: M.mean([r["correct"] for r in rows if r["type"] == t]) for t in TYPES},
             "by_type_latency": {t: M.mean([r["latency_ms"] for r in rows if r["type"] == t]) for t in TYPES}}
        S[name] = d
    return S


def findings(S):
    if not all(k in S for k in SYS_LABEL):
        return ["(findings need all three systems - run without --systems)"]
    a, h, n = S["agent"], S["hybrid_rerank_rag"], S["naive_rag"]
    out = []
    p = lambda x: f"{100 * x:.0f}%"
    for t, nice in [("multihop", "multi-hop / comparison"), ("numerical", "numerical"), ("single", "single-fact"),
                    ("ambiguous", "ambiguous-wording"), ("unanswerable", "unanswerable (refusal)")]:
        d = a["by_type"][t] - h["by_type"][t]
        if abs(d) < 0.03: out.append(f"On **{nice}** questions the agent is on par with single-shot hybrid+rerank ({p(a['by_type'][t])} vs {p(h['by_type'][t])}): no measurable gain.")
        elif d > 0: out.append(f"On **{nice}** questions the agent beats single-shot hybrid+rerank by {100 * d:.0f} points ({p(a['by_type'][t])} vs {p(h['by_type'][t])}).")
        else: out.append(f"On **{nice}** questions the agent is *worse* than single-shot hybrid+rerank by {-100 * d:.0f} points ({p(a['by_type'][t])} vs {p(h['by_type'][t])}).")
    out.append(f"Cost of the agent: {a['avg_steps']:.1f} graph steps and {a['tokens']:.0f} (estimated) tokens per query vs {h['tokens']:.0f} for hybrid+rerank "
               f"({a['tokens'] / max(h['tokens'], 1):.1f}x); p95 latency {a['p95_ms']:.0f} ms vs {h['p95_ms']:.0f} ms.")
    out.append(f"Naive RAG never refuses (refusal recall {p(n['refusal_recall'])}); the agent reaches {p(a['refusal_recall'])} recall at {p(a['refusal_precision'])} precision, "
               f"with {a['false_refusals']} false refusals on answerable questions.")
    return out


def tables(abl, n_ans, S, per_q, meta, gold_n):
    pc = lambda x: "-" if x is None else f"{100 * x:.1f}"
    L = []
    if meta["gen"].startswith("offline"):
        L.append("> **Read this first.** This run used the *offline rule-based backend* (no LLM) on a *synthetic* corpus. It proves the pipeline, tools, "
                 "graph, metrics and refusal logic work end to end, but it is **not evidence of answer quality**: the single-shot baselines can only extract "
                 "sentences (they cannot compare or do arithmetic) and the agent's rules were written by the same author as the question templates. "
                 "Re-run with `LLM_PROVIDER=groq JUDGE_PROVIDER=hf make eval` on your real corpus and hand-written golden set for numbers you can quote.\n")
    L.append(f"*{gold_n} golden questions ({n_ans} answerable). Generator: `{meta['gen']}`; judge: `{meta['judge']}`; retrieval: `{meta['retrieval']}`; "
             f"graph engine: `{ENGINE}`.*\n")
    L.append("### Retrieval ablation (answerable questions, page-level gold)\n")
    L.append("| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | MRR |\n|---|---|---|---|---|---|---|")
    for m in MODES:
        x = abl[m]; L.append(f"| {MODE_LABEL[m]} | {pc(x['hit@1'])} | {pc(x['hit@3'])} | {pc(x['hit@5'])} | {pc(x['recall@3'])} | {pc(x['recall@5'])} | {x['mrr']:.3f} |")
    L.append("\n### Three-system comparison\n")
    names = list(S)
    L.append("| Metric | " + " | ".join(SYS_LABEL[s] for s in names) + " |\n|---|" + "---|" * len(names))
    rowdefs = [("Correctness, all questions (%)", "correctness", pc), ("Correctness, answerable only (%)", "correctness_answerable", pc),
               ("Faithfulness (%)", "faithfulness", pc), ("Citation hit (gold page cited) (%)", "citation_hit", pc),
               ("Cited text contains answer (%)", "cited_contains_answer", pc), ("Refusal precision (%)", "refusal_precision", pc),
               ("Refusal recall (%)", "refusal_recall", pc), ("False refusals (count)", "false_refusals", lambda x: str(x)),
               ("Avg steps / question", "avg_steps", lambda x: f"{x:.1f}"), ("Tool-call success (%)", "tool_success", pc),
               ("Loop termination (%)", "termination", pc), ("Degraded answers (LLM fell back to rules)", "degraded", lambda x: str(x)), ("Latency p50 (ms)", "p50_ms", lambda x: f"{x:.0f}"),
               ("Latency p95 (ms)", "p95_ms", lambda x: f"{x:.0f}"), ("Tokens / query (est.)", "tokens", lambda x: f"{x:.0f}")]
    for label, key, f in rowdefs:
        vals = [S[s][key] for s in names]
        L.append(f"| {label} | " + " | ".join("-" if v is None else f(v) for v in vals) + " |")
    L.append("\n### Correctness by question type (%)\n")
    L.append("| System | " + " | ".join(TYPES) + " |\n|---|" + "---|" * len(TYPES))
    for s in names:
        L.append(f"| {SYS_LABEL[s]} | " + " | ".join(pc(S[s]['by_type'][t]) if any(r['system'] == s and r['type'] == t for r in per_q) else '-' for t in TYPES) + " |")
    L.append("\n### Findings (generated from the run)\n")
    for f in findings(S): L.append(f"- {f}")
    jf = RESULTS_DIR / "judge_agreement.json"
    if jf.exists():
        j = json.loads(jf.read_text()); L.append(f"\n**Judge vs. human:** {100 * j['agreement']:.0f}% agreement, Cohen's kappa {j['cohen_kappa']} on {j['n']} hand-graded answers.")
    else:
        L.append("\n**Judge vs. human agreement:** not measured yet - run `python -m eval.hand_grade sample`, label 20 answers, then `python -m eval.hand_grade agree`.")
    return "\n".join(L)


def plots(abl, S):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        print("[eval] matplotlib not installed - skipping plots"); return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ks = ["hit@1", "hit@3", "hit@5", "mrr"]; w = 0.2
    for i, m in enumerate(MODES):
        ax.bar([j + i * w for j in range(len(ks))], [abl[m][k] for k in ks], w, label=MODE_LABEL[m])
    ax.set_xticks([j + 1.5 * w for j in range(len(ks))]); ax.set_xticklabels(["Hit@1", "Hit@3", "Hit@5", "MRR"]); ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval ablation"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(RESULTS_DIR / "retrieval_ablation.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.4)); w = 0.26
    for i, s in enumerate(S):
        ax.bar([j + i * w for j in range(len(TYPES))], [S[s]["by_type"][t] for t in TYPES], w, label=SYS_LABEL[s])
    ax.set_xticks([j + w for j in range(len(TYPES))]); ax.set_xticklabels(TYPES, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_title("Correctness by question type"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(RESULTS_DIR / "correctness_by_type.png", dpi=140); plt.close(fig)


def update_readme(md: str):
    p = ROOT / "README.md"
    if not p.exists(): return
    s = p.read_text()
    block = "<!-- RESULTS:START -->\n" + md + "\n<!-- RESULTS:END -->"
    s = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda m: block, s, flags=re.S)
    p.write_text(s)


def spread(gold, n):
    """n questions spread evenly over the whole (type-interleaved) set, deterministic."""
    if not n or n >= len(gold): return gold
    step = len(gold) / n
    return [gold[int(i * step)] for i in range(n)]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrieval-only", action="store_true", help="only the retrieval ablation (no LLM calls at all)")
    ap.add_argument("--systems", default="", help="comma list of: naive_rag,hybrid_rerank_rag,agent (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only N questions spread across the set (saves free-tier quota)")
    a = ap.parse_args()
    t0 = time.time(); RESULTS_DIR.mkdir(exist_ok=True)
    gold = load_golden(); r = Retriever()
    abl, n_ans = retrieval_ablation(r, gold)
    info = r.info()
    ret_meta = f"{info['embedder']} + BM25 + {info['reranker']}; store={info['store']}"
    if a.retrieval_only:
        pc = lambda x: f"{100 * x:.1f}"
        out = [f"*{len(gold)} golden questions ({n_ans} answerable); retrieval: `{ret_meta}`*\n", "| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | MRR |", "|---|---|---|---|---|---|---|"]
        for m in MODES:
            x = abl[m]; out.append(f"| {MODE_LABEL[m]} | {pc(x['hit@1'])} | {pc(x['hit@3'])} | {pc(x['hit@5'])} | {pc(x['recall@3'])} | {pc(x['recall@5'])} | {x['mrr']:.3f} |")
        md = "\n".join(out); (RESULTS_DIR / "retrieval_only.md").write_text(md + "\n"); print(md); return
    gold = spread(gold, a.limit)
    names = [s.strip() for s in a.systems.split(",") if s.strip()]
    gen, jud = get_llm(), get_llm(JUDGE_PROVIDER, JUDGE_MODEL)
    if not gen.offline and gen.name == jud.name:
        print("[eval] WARNING: judge model == generator model; use a different judge to avoid self-preference bias")
    judge = Judge(jud)
    per_q = run_systems(r, gold, gen, judge, names)
    S = summarize(per_q)
    meta = {"gen": gen.name, "judge": jud.name + (" (rule-based)" if jud.offline else ""), "retrieval": ret_meta}
    partial = bool(a.limit or names)
    (RESULTS_DIR / ("results_partial.json" if partial else "results.json")).write_text(json.dumps({"meta": meta, "retrieval_ablation": abl, "systems": S, "n_questions": len(gold)}, indent=1))
    (RESULTS_DIR / "per_question.jsonl").write_text("\n".join(json.dumps(x, default=str) for x in per_q) + "\n")
    md = tables(abl, n_ans, S, per_q, meta, len(gold))
    (RESULTS_DIR / ("results_partial.md" if partial else "results.md")).write_text(md + "\n")
    if not partial:
        plots(abl, S); update_readme(md)
    print(md); print(f"\n[eval] done in {time.time() - t0:.0f}s -> results/")


if __name__ == "__main__":
    main()
