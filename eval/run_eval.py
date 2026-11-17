"""python -m eval.run_eval   (or `make eval`)
Runs: (1) retrieval ablation, (2) three-system comparison (naive RAG / hybrid+rerank / full agent) with LLM-judge
correctness + faithfulness, citation accuracy, refusal P/R, agent + ops metrics. Writes results/ and refreshes the README."""
from __future__ import annotations
import json, re, sys, time
from collections import defaultdict
from config import GOLDEN_PATH, RESULTS_DIR, LLM_PROVIDER, LLM_MODEL, JUDGE_PROVIDER, JUDGE_MODEL, ROOT
from llm import get_llm
from llm.base import LLMQuotaError
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
    """-> (rows, stop_message). On a long rate-limit (LLMQuotaError) it stops cleanly and returns what is finished."""
    systems = {"naive_rag": NaiveRAG(r, gen_llm), "hybrid_rerank_rag": HybridRAG(r, gen_llm), "agent": Agent(r, gen_llm)}
    systems = {k: v for k, v in systems.items() if not names or k in names}
    by_id = {c["id"]: c for c in r.chunks}
    per_q, stop = [], ""
    for name, s in systems.items():
        for i, g in enumerate(gold, 1):
            t0 = time.time()
            try:
                res = s.run(g["question"])
                corr = judge.correctness(g, res); faith = judge.faithfulness(res, by_id)
            except LLMQuotaError as e:
                stop = f"STOPPED at {name} question {i}/{len(gold)} ({g['id']}): {e}"
                print(f"[eval] {stop}", flush=True); break
            for c in res["citations"]: c["full_text"] = by_id[c["chunk_id"]]["text"]
            cit = M.citation_scores(res["citations"], g["gold_sources"], g["answer_keys"]) if g["type"] != "unanswerable" and not res["refused"] else \
                  {"citation_hit": 0.0, "citation_precision": 0.0, "cited_contains_answer": 0.0}
            per_q.append({"system": name, "id": g["id"], "type": g["type"], "question": g["question"], "gold_answer": g["gold_answer"],
                          "answer": res["answer"], "refused": res["refused"], "correct": corr["score"], "judge_reason": corr["reason"],
                          "faithfulness": faith["score"], **cit, "tokens": res["tokens"], "latency_ms": res["latency_ms"], "steps": res["steps"],
                          "loops": res["loops"], "tool_calls": res["tool_calls"], "terminated": res["terminated"], "degraded": res["degraded"],
                          "llm_error": res.get("llm_error", ""), "trace": [f"{t['node']}: {t['summary'][:160]}" for t in res["trace"]],
                          "n_citations": len(res["citations"])})
            print(f"[eval] {name} {i}/{len(gold)} {g['id']:<8} {time.time() - t0:5.1f}s  correct={corr['score']} refused={res['refused']}", flush=True)
            (RESULTS_DIR / "per_question.jsonl").write_text("\n".join(json.dumps(x, default=str) for x in per_q) + "\n")   # saved as we go
        if stop: break
        print(f"[eval] {name} done", flush=True)
    return per_q, stop


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


def findings(S, per_q):
    if not all(k in S for k in SYS_LABEL):
        return ["(findings need all three systems - run without --systems)"]
    a, h, n = S["agent"], S["hybrid_rerank_rag"], S["naive_rag"]
    by = {s: {r["id"]: r for r in per_q if r["system"] == s} for s in SYS_LABEL}
    ids = sorted(set.intersection(*[set(v) for v in by.values()]))
    ok = lambda s, i: by[s][i]["correct"] >= 0.5
    p = lambda x: f"{100 * x:.0f}%"
    out = [f"All comparisons below use the **{len(ids)} questions that every system completed**."]
    for x, y, lx, ly in [("agent", "hybrid_rerank_rag", "agent", "hybrid+rerank"), ("agent", "naive_rag", "agent", "naive RAG"),
                         ("hybrid_rerank_rag", "naive_rag", "hybrid+rerank", "naive RAG")]:
        xo = sum(ok(x, i) and not ok(y, i) for i in ids); yo = sum(ok(y, i) and not ok(x, i) for i in ids)
        pv = M.sign_test_p(xo, yo)
        verdict = "a statistically detectable difference" if pv < 0.05 else "no statistically detectable difference"
        out.append(f"Paired, {lx} vs {ly}: {lx} right & {ly} wrong on {xo} question(s); {ly} right & {lx} wrong on {yo} (exact sign test p={pv:.2f}) - {verdict}.")
    for t, nice in [("multihop", "multi-hop / comparison"), ("numerical", "numerical"), ("single", "single-fact"),
                    ("ambiguous", "tricky-wording"), ("unanswerable", "unanswerable (refusal)")]:
        it = [i for i in ids if by["agent"][i]["type"] == t]
        if not it:
            out.append(f"**{nice}**: no questions of this type were completed by all systems yet."); continue
        ka = sum(ok("agent", i) for i in it); kh = sum(ok("hybrid_rerank_rag", i) for i in it); nn = len(it)
        d = ka - kh
        tag = "within one question - noise" if abs(d) <= 1 else f"{'agent' if d > 0 else 'hybrid+rerank'} ahead by {abs(d)} question(s) - small sample, not tested on its own"
        out.append(f"**{nice}** ({nn} questions): agent {ka}/{nn} vs hybrid+rerank {kh}/{nn} ({tag}).")
    out.append(f"Cost of the agent: {a['avg_steps']:.1f} graph steps and {a['tokens']:.0f} tokens per query vs {h['tokens']:.0f} for hybrid+rerank "
               f"({a['tokens'] / max(h['tokens'], 1):.1f}x); p95 latency {a['p95_ms']:.0f} ms vs {h['p95_ms']:.0f} ms (latency is distorted by caching/throttling).")
    n_unans = sum(1 for i in ids if by["agent"][i]["type"] == "unanswerable")
    if n_unans:
        out.append(f"Refusal ({n_unans} unanswerable questions): naive RAG recall {p(n['refusal_recall'])}; hybrid+rerank recall {p(h['refusal_recall'])} at {p(h['refusal_precision'])} precision "
                   f"({h['false_refusals']} false refusals); the agent recall {p(a['refusal_recall'])} at {p(a['refusal_precision'])} ({a['false_refusals']} false refusals).")
    else:
        out.append(f"No unanswerable questions in this subset, so refusal precision/recall are n/a. False refusals of answerable questions: naive {n['false_refusals']}, "
                   f"hybrid+rerank {h['false_refusals']}, agent {a['false_refusals']}.")
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
        no_unans = not any(r["type"] == "unanswerable" for r in per_q)
        L.append(f"| {label} | " + " | ".join("n/a" if ((key in ("citation_hit", "cited_contains_answer") and s == "naive_rag")
                                                       or (key in ("refusal_precision", "refusal_recall") and no_unans)) else "-" if v is None else f(v)
                                              for s, v in zip(names, vals)) + " |")
    ci = []
    for s in names:
        rs = [r for r in per_q if r["system"] == s]; k = sum(r["correct"] >= 0.5 for r in rs); lo, hi = M.wilson(k, len(rs))
        ci.append(f"{k}/{len(rs)} ({100 * lo:.0f}-{100 * hi:.0f}%)")
    L.append("| Correct / n (95% Wilson interval) | " + " | ".join(ci) + " |")
    L.append("\n*Naive RAG is not asked to cite (its citation metrics are n/a). Hybrid and the agent use the same citation parser and the same post-hoc attribution rule. Latency includes any client-side throttling and excludes cached calls; use a cold run for latency claims.*")
    L.append("\n### Correctness by question type (%)\n")
    L.append("| System | " + " | ".join(TYPES) + " |\n|---|" + "---|" * len(TYPES))
    for s in names:
        cells = []
        for t in TYPES:
            rs = [r for r in per_q if r["system"] == s and r["type"] == t]
            cells.append(f"{100 * sum(r['correct'] >= 0.5 for r in rs) / len(rs):.0f} ({sum(r['correct'] >= 0.5 for r in rs)}/{len(rs)})" if rs else "-")
        L.append(f"| {SYS_LABEL[s]} | " + " | ".join(cells) + " |")
    L.append("\n### Findings (generated from the run)\n")
    for f in findings(S, per_q): L.append(f"- {f}")
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
    ap.add_argument("--types", default="", help="comma list of question types to evaluate (single,multihop,numerical,unanswerable,ambiguous)")
    ap.add_argument("--exclude", default="", help="comma list of question ids to leave out (e.g. the ones used while debugging)")
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
    if a.types:
        want = {s.strip() for s in a.types.split(",") if s.strip()}
        gold = [g for g in gold if g["type"] in want]
    if a.exclude:
        skip = {s.strip() for s in a.exclude.split(",") if s.strip()}
        gold = [g for g in gold if g["id"] not in skip]
    gold = spread(gold, a.limit)
    names = [s.strip() for s in a.systems.split(",") if s.strip()]
    gen, jud = get_llm(), get_llm(JUDGE_PROVIDER, JUDGE_MODEL)
    if not gen.offline and gen.name == jud.name:
        print("[eval] WARNING: judge model == generator model; use a different judge to avoid self-preference bias")
    judge = Judge(jud)
    per_q, stop = run_systems(r, gold, gen, judge, names)
    if not per_q: raise SystemExit(stop or 'nothing evaluated')
    ids_by_sys = [{r["id"] for r in per_q if r["system"] == s} for s in {r["system"] for r in per_q}]
    common = set.intersection(*ids_by_sys) if ids_by_sys else set()
    per_all, per_q = per_q, [r for r in per_q if r["id"] in common]        # fair comparison: only questions finished by every system
    if len(common) < len(gold): stop = stop or f"{len(common)} of {len(gold)} requested questions were completed by every system; the table uses only those."
    S = summarize(per_q)
    meta = {"gen": gen.name, "judge": jud.name + (" (rule-based)" if jud.offline else ""), "retrieval": ret_meta}
    partial = bool(a.limit or names or stop)
    (RESULTS_DIR / ("results_partial.json" if partial else "results.json")).write_text(json.dumps({"meta": meta, "retrieval_ablation": abl, "systems": S, "n_questions": len(gold)}, indent=1))
    (RESULTS_DIR / "per_question.jsonl").write_text("\n".join(json.dumps(x, default=str) for x in per_all) + "\n")
    md = tables(abl, n_ans, S, per_q, meta, len(gold))
    if stop: md = f"> **Run stopped early - results below cover only the questions finished.** {stop}\n\n" + md
    (RESULTS_DIR / ("results_partial.md" if partial else "results.md")).write_text(md + "\n")
    if not partial:
        plots(abl, S); update_readme(md)
    print(md); print(f"\n[eval] done in {time.time() - t0:.0f}s -> results/")


if __name__ == "__main__":
    main()
