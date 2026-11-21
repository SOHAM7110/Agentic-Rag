"""The agent as an explicit state graph:

  plan -> retrieve -> grade --(insufficient & loops<3 & budget left)--> rewrite -> retrieve ...
                       \\--(sufficient or limits hit)--> tools -> answer -> verify -> END

Hard limits: MAX_LOOPS retrieval loops and TOKEN_BUDGET tokens per query, so it always terminates."""
from __future__ import annotations
import json, re, threading, time
from typing import Any, TypedDict
from config import MAX_LOOPS, TOKEN_BUDGET, TOP_K, GRADE_FLOOR, PROMPT_PASSAGES
from ingest.facts import FACT_SPEC
from llm.base import LLMError, LLMQuotaError
from retrieval.retriever import ctx_text
from retrieval.text import approx_tokens, tokenize
from . import heuristics as H, prompts as P, tools as T

try:                                           # real LangGraph if installed, otherwise the bundled mini engine
    from langgraph.graph import StateGraph, END
    ENGINE = "langgraph"
except Exception:                              # pragma: no cover
    from .minigraph import StateGraph, END
    ENGINE = "minigraph"


class State(TypedDict, total=False):
    question: str; plan: dict; queries: list; loops: int; evidence: list; coverages: list; progress: int; tried: list; stalled: bool
    sufficient: bool; reason: str; tool_results: list; answer: str; claims: list
    citations: list; refused: bool; trace: list; tokens: int; degraded: bool; passages: list


def inr(n) -> str:
    if isinstance(n, float): n = round(n, 2)
    if not float(n).is_integer(): return str(n)
    s = str(int(abs(n))); sign = "-" if n < 0 else ""
    if len(s) <= 3: return sign + s
    head, tail, parts = s[:-3], s[-3:], []
    while len(head) > 2: parts.insert(0, head[-2:]); head = head[:-2]
    if head: parts.insert(0, head)
    return sign + ",".join(parts + [tail])


def parse_json(text: str):
    try:
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group()) if m else None
    except Exception:
        return None


class Agent:
    def __init__(self, retriever, llm, control=None):
        """control: optional cheaper model for the control steps (plan, grade, rewrite, calculator, verify); the main `llm` writes the answer."""
        self.r, self.llm = retriever, llm
        self.ctl = control if (control is not None and not llm.offline) else llm
        self.docs = retriever.docs
        # the structured tools only make sense if the facts table was populated at ingestion
        self.has_facts = any(retriever.facts.values())
        self._tl = threading.local()          # per-run counters (tokens, fallbacks): LangGraph passes nodes a COPY of the state,
        self.app = self._build()              # so anything mutated in place there would be lost

    # ------------------------------------------------------------------ plumbing
    def _rs(self) -> dict:
        rs = getattr(self._tl, "rs", None)
        if rs is None: rs = self._tl.rs = {"tokens": 0, "degraded": False, "llm_error": ""}
        return rs

    def _llm(self, st: dict, system: str, user: str, heuristic, json_mode=True):
        """Call the model (or the rule-based heuristic in offline mode). Falls back to the heuristic on any
        failure so a rate-limit or malformed JSON never crashes a query."""
        tok = approx_tokens(system + user)
        if self.llm.offline:
            out = heuristic(); self._rs()["tokens"] += tok + approx_tokens(json.dumps(out, default=str))
            return out
        try:
            r = (self.llm if system == P.ANSWER_SYSTEM else self.ctl).generate(system, user, json_mode=json_mode)
            self._rs()["tokens"] += r.in_tokens + r.out_tokens
            if not json_mode: return r.text
            data = parse_json(r.text)
            if data is not None: return data
            self._rs()["llm_error"] = f"unparseable JSON: {r.text[:200]!r}"
        except LLMQuotaError:
            raise
        except LLMError as e:
            self._rs()["llm_error"] = str(e)[:300]
        self._rs()["degraded"] = True
        return heuristic()

    def _trace(self, st, node, summary, t0, **data):
        tr = list(st.get("trace", []))
        tr.append({"step": len(tr) + 1, "node": node, "summary": summary, "ms": round((time.perf_counter() - t0) * 1000, 1),
                   "tokens": self._rs()["tokens"], "data": data})
        return tr

    def _lim(self, st) -> int:
        """How many evidence chunks the LLM sees - identical for grading and answering, so the grader never approves
        evidence the answerer cannot see. 3 per sub-query when there are several."""
        return 8 if self.llm.offline else max(PROMPT_PASSAGES, 3 * len(st.get("queries", [])))

    def _ev_chunks(self, st) -> list[dict]:
        return [self.r.chunks[e["idx"]] for e in st.get("evidence", [])]

    # ------------------------------------------------------------------ nodes
    def plan(self, st):
        t0 = time.perf_counter(); q = st["question"]
        plan = H.make_plan(q, self.docs)
        if not self.has_facts:                      # no facts table -> no aspect lookups / scheme-specific numeric templates
            plan.update(aspect=None, numeric=None, descriptor=None, tools=[])
            if plan["type"] == "compare" and len(plan["entities"]) < 2: plan["type"] = "single"
        subs: list[str] = []
        if not self.llm.offline:
            out = self._llm(st, P.PLAN_SYSTEM, q, lambda: {"type": plan["type"], "sub_questions": []})
            if out.get("type") in ("single", "compare", "numeric"): plan["type"] = out["type"]
            subs = [s.strip() for s in (out.get("sub_questions") or []) if isinstance(s, str) and s.strip()][:4]
        plan["sub_questions"] = subs
        queries = self._initial_queries(q, plan)
        names = [self.docs[e]["title"] for e in plan["entities"]]
        return {"plan": plan, "queries": queries, "loops": 0, "evidence": [], "tool_results": [],
                "trace": self._trace(st, "plan", f"type={plan['type']} sub_questions={len(subs)} entities={names or 'none'} aspect={plan['aspect']} tools={plan['tools']}"
                                     + (f" descriptor='{plan['descriptor']}'" if plan["descriptor"] else ""), t0, plan=plan, queries=queries)}

    def _initial_queries(self, q, plan):
        ents, asp = plan["entities"], plan["aspect"]
        label = FACT_SPEC[asp][1] if asp else ""
        subs = plan.get("sub_questions") or []
        if len(ents) >= 2:       # named documents: search INSIDE each one (essential when documents are near-duplicates)
            topic = H.strip_entity_words(q, ents, self.docs)
            return [{"q": f"{self.docs[e]['title']} {label}" if label else topic, "doc": e} for e in ents[:4]]
        if len(ents) == 1 and not subs:
            return [{"q": H.strip_entity_words(q, ents, self.docs), "doc": ents[0]}]
        if len(subs) >= 2 or (subs and plan["type"] != "single"):     # LLM decomposition: one retrieval per sub-question
            return [{"q": s, "doc": None} for s in subs]
        if plan["descriptor"]:
            return [{"q": plan["descriptor"], "doc": None, "role": "resolve"}]
        if len(ents) >= 2:
            return [{"q": f"{self.docs[e]['title']} {label or q}", "doc": e} for e in ents]
        if ents:
            return [{"q": q, "doc": ents[0]}]
        return [{"q": q, "doc": None}]

    def retrieve(self, st):
        t0 = time.perf_counter(); plan = dict(st["plan"]); queries = list(st["queries"]); loops = st["loops"] + 1
        note = ""
        # descriptor resolution ("the scheme that insures fishing boats") = first hop of a multi-hop question
        if plan["descriptor"] and not plan["entities"]:
            hit = self.r.search(plan["descriptor"], "hybrid_rerank", k=3)
            if hit:
                did = hit[0].chunk["doc_id"]; plan["entities"] = [did]
                queries = [{"q": st["question"], "doc": did}]
                note = f" | hop 1: descriptor resolved to '{self.docs[did]['title']}'"
        lists, cov = [], []
        for qd in queries:
            docs_f = [qd["doc"]] if qd.get("doc") and not qd.get("drop") else None
            hits = self.r.search(qd["q"], "hybrid_rerank", k=TOP_K if len(queries) == 1 else 3, docs=docs_f)
            lists.append(hits)
            ign = H.entity_tokens(plan["entities"], self.docs) if qd.get("doc") else set()
            qt = H.query_tokens(qd["q"], ign, drop_numbers=not self.llm.offline)
            best = max((self.r.coverage(qt, h.chunk, ign) for h in hits), default=0.0)
            asp_ok = True
            if plan["aspect"] and qd.get("doc"):
                cue = FACT_SPEC[plan["aspect"]][3]
                asp_ok = any(re.search(cue, h.chunk["text"].lower().replace(",", "")) for h in hits)
            cov.append({"q": qd["q"], "doc": qd.get("doc"), "coverage": round(best, 2), "aspect_ok": asp_ok})
        merged, seen = [], set()
        for rank in range(max((len(l) for l in lists), default=0)):          # round-robin so every entity is represented
            for l in lists:
                if rank < len(l) and l[rank].idx not in seen:
                    seen.add(l[rank].idx); merged.append({"idx": l[rank].idx, "score": round(l[rank].score, 3)})
        new_ids = {e["idx"] for e in merged}
        prev = st.get("evidence", []) if loops > 1 else []
        progress = len(new_ids - {e["idx"] for e in prev}) if prev else len(new_ids)
        if prev:                                   # a rewrite should ADD evidence, not replace what was already found
            from itertools import zip_longest
            mixed, seen2 = [], set()
            for a_, b_ in zip_longest(merged, prev):
                for x in (a_, b_):
                    if x is not None and x["idx"] not in seen2:
                        seen2.add(x["idx"]); mixed.append(x)
            merged = mixed
        ev = merged[:8]
        tok = sum(approx_tokens(self.r.chunks[e["idx"]]["text"]) for e in ev)
        self._rs()["tokens"] += tok // 2
        return {"plan": plan, "queries": queries, "loops": loops, "evidence": ev, "coverages": cov, "progress": progress,
                "trace": self._trace(st, "retrieve", f"loop {loops}: {len(queries)} query(ies) -> {len(ev)} chunks (hybrid+rerank){note}", t0,
                                     queries=[q["q"] for q in queries], hits=[self.r.chunks[e["idx"]]["id"] for e in ev], coverage=cov)}

    def grade(self, st):
        t0 = time.perf_counter(); plan = st["plan"]; cov = st["coverages"]
        def heuristic():
            asp, ents = plan["aspect"], plan["entities"]
            # structured table counts as evidence: a known fact is sufficient on its own
            if asp and ents and plan["type"] in ("single", "compare") and all(asp in self.r.facts.get(e, {}) for e in ents):
                return {"sufficient": True, "reason": f"fact '{asp}' present in facts table for {len(ents)} doc(s)"}
            if plan["type"] == "numeric" and plan["numeric"]:
                need = [(d, k) for d in plan["numeric"]["docs"] for k in plan["numeric"]["keys"]]
                if all(k in self.r.facts.get(d, {}) for d, k in need):
                    return {"sufficient": True, "reason": "all numeric inputs present in facts table"}
            bad = [c for c in cov if c["coverage"] < 0.6 or not c["aspect_ok"]]
            if bad:
                b = bad[0]
                why = "attribute not mentioned in the document" if not b["aspect_ok"] else f"low query coverage ({b['coverage']})"
                return {"sufficient": False, "reason": why}
            return {"sufficient": True, "reason": f"coverage ok ({min(c['coverage'] for c in cov)})"}
        h = heuristic()
        if self.llm.offline:
            out = h; self._llm(st, P.GRADE_SYSTEM, st["question"], lambda: h)
        else:
            user = f"Question: {st['question']}\n\nPassages:\n{P.format_passages(self._ev_chunks(st)[:self._lim(st)])}"
            out = self._llm(st, P.GRADE_SYSTEM, user, lambda: h)
            floor = min((c["coverage"] for c in cov), default=0)
            out = {"sufficient": bool(out.get("sufficient")) and floor >= GRADE_FLOOR, "reason": out.get("reason", "")}
        over = self._rs()["tokens"] > TOKEN_BUDGET
        suff = bool(out["sufficient"])
        reason = out["reason"] + (" | token budget exhausted" if over and not suff else "")
        return {"sufficient": suff, "reason": reason,
                "trace": self._trace(st, "grade", f"sufficient={suff} ({reason})", t0, sufficient=suff, reason=reason)}

    def route_after_grade(self, st):
        if not st["sufficient"] and st["loops"] >= 2 and st.get("progress", 1) == 0:
            return "tools"            # the rewritten search found nothing new: more loops would only burn tokens
        if st["sufficient"] or st["loops"] >= MAX_LOOPS or self._rs()["tokens"] > TOKEN_BUDGET:
            return "tools"
        return "rewrite"

    def rewrite(self, st):
        t0 = time.perf_counter(); plan = st["plan"]; attempt = st["loops"]
        norm = lambda q: " ".join(sorted(set(tokenize(q))))
        tried = set(st.get("tried", [])) | {norm(qd["q"]) for qd in st["queries"]}
        new_queries, stalled = [], False
        for qd in st["queries"]:
            if qd.get("doc") is None and len(st["queries"]) > 1:      # LLM sub-question: only the LLM rewrites it
                rw = {"q": qd["q"], "drop_filter": False}
            else:
                rw = H.rewrite_query(qd["q"] if len(st["queries"]) > 1 else st["question"], plan, self.docs, attempt)
            if len(st["queries"]) > 1 and qd.get("doc") is not None:   # keep one query per entity
                label = FACT_SPEC[plan["aspect"]][1] if plan["aspect"] else qd["q"]
                rw["q"] = f"{self.docs[qd['doc']]['title']} {label}" if attempt < 2 else f"{label} {' '.join(FACT_SPEC[plan['aspect']][2][:3]) if plan['aspect'] else ''}"
            if not self.llm.offline:
                out = self._llm(st, P.REWRITE_SYSTEM, f"Question: {st['question']}\nPrevious query: {qd['q']}\nWhat the grader says is still missing: {st.get('reason', '')}", lambda: {"query": rw["q"]})
                rw["q"] = out.get("query") or rw["q"]
            if norm(rw["q"]) in tried:                                  # identical to something already searched: try a keyword variant
                kw = " ".join(H.query_tokens(st["question"], set(), drop_numbers=True))
                alt = f"{kw} {st.get('reason', '')}".strip()
                if norm(alt) not in tried and kw:
                    rw["q"] = alt
                else:
                    stalled = True
            tried.add(norm(rw["q"]))
            new_queries.append({"q": rw["q"], "doc": qd.get("doc"), "drop": rw["drop_filter"]})
        return {"queries": new_queries, "tried": sorted(tried), "stalled": stalled,
                "trace": self._trace(st, "rewrite", f"attempt {attempt}: " + " || ".join(q["q"] for q in new_queries)
                                     + (" (doc filter dropped)" if new_queries[0].get("drop") else "")
                                     + (" | no new query possible -> stop" if stalled else ""), t0, queries=[q["q"] for q in new_queries])}

    def route_after_rewrite(self, st):
        return "tools" if st.get("stalled") else "retrieve"

    def tools(self, st):
        t0 = time.perf_counter(); plan = st["plan"]; res: list[T.ToolResult] = []
        asp, ents = plan["aspect"], plan["entities"]
        if plan["type"] == "compare" and asp and len(ents) >= 2 and all(self.r.facts.get(e) for e in ents):
            res.append(T.compare(self.r, ents, asp))
        elif plan["type"] == "numeric" and plan["numeric"]:
            npn = plan["numeric"]; vals = []
            for i, key in enumerate(npn["keys"]):
                d = npn["docs"][i % len(npn["docs"])] if npn["op"] in ("sum", "diff") else npn["docs"][0]
                r = T.lookup(self.r, d, key); res.append(r); vals.append(r.data.get("value") if r.ok else None)
            if npn["op"] in ("sum", "diff") and len(npn["keys"]) == 1 and len(npn["docs"]) == 2:
                res, vals = [], []
                for d in npn["docs"]:
                    r = T.lookup(self.r, d, npn["keys"][0]); res.append(r); vals.append(r.data.get("value") if r.ok else None)
            if all(v is not None for v in vals) and vals:
                expr = npn["tmpl"].format(*vals)
                if not self.llm.offline:
                    out = self._llm(st, P.CALC_SYSTEM, f"Question: {st['question']}\nFacts: {[r.output for r in res]}", lambda: {"expression": expr})
                    cand = T.calculator(str(out.get("expression", "")))
                    if cand.ok: expr = str(out["expression"])
                res.append(T.calculator(expr))
        elif plan["type"] == "numeric" and not self.llm.offline:
            # no scheme template applies: let the LLM write ONE expression from numbers in the question/passages
            pas = self._passages(st)
            user = f"Question: {st['question']}\n\nPassages:\n{P.format_passages(pas)}"
            out = self._llm(st, P.CALC_SYSTEM, user, lambda: {"expression": ""})
            expr = str(out.get("expression", "")).strip()
            if expr: res.append(T.calculator(expr))
        elif plan["type"] == "single" and asp and ents and self.r.facts.get(ents[0]):
            res.append(T.lookup(self.r, ents[0], asp))
        out = [r.brief() for r in res]
        return {"tool_results": [r.__dict__ for r in res],
                "trace": self._trace(st, "tools", ("; ".join(f"{r.tool}({'ok' if r.ok else 'fail: ' + r.error})" for r in res) or "no tool needed"), t0, calls=out)}

    # ---- answer
    def _passages(self, st) -> list[dict]:
        lim = self._lim(st)
        chunks = self._ev_chunks(st)[:lim]
        ids = {c["id"] for c in chunks}
        for tr in st.get("tool_results", []):          # chunks backing tool outputs must be citable
            for cid in tr["sources"]:
                if cid not in ids:
                    chunks.append(next(c for c in self.r.chunks if c["id"] == cid)); ids.add(cid)
        return chunks[:lim + 2]

    def answer(self, st):
        t0 = time.perf_counter(); plan = st["plan"]
        passages = self._passages(st)
        num = {c["id"]: i for i, c in enumerate(passages, 1)}
        tools_ok = [r for r in st["tool_results"] if r["ok"]]
        refuse = (not st["sufficient"]) or not passages
        claims: list[dict] = []; attributed = 0
        if not refuse:
            if self.llm.offline:
                claims = self._compose_offline(st, passages, num, tools_ok)
            else:
                tool_txt = "\n".join(f"TOOL RESULT {r['tool']}({r['args']}) = {r['output']} {''.join('[' + str(num[s]) + ']' for s in r['sources'] if s in num)}" for r in tools_ok)
                user = f"Question: {st['question']}\n\nPassages:\n{P.format_passages(passages)}\n\n{tool_txt}"
                txt = self._llm(st, P.ANSWER_SYSTEM, user, lambda: "INSUFFICIENT_EVIDENCE", json_mode=False)
                if "INSUFFICIENT_EVIDENCE" in txt: refuse = True
                else:
                    claims = self._claims_from_text(txt)
                    attributed = self._attribute(claims, passages)
            if not claims: refuse = True
        if refuse:
            reason = st.get("reason", "") + (" | rewritten searches found no new evidence" if st.get("loops", 1) > 1 and st.get("progress", 1) == 0 else "") + (" | no new query could be formed" if st.get("stalled") else "")
            text = f"I can't answer this from the provided documents ({reason or 'no supporting evidence found'})."
            return {"answer": text, "claims": [], "refused": True, "passages": [c["id"] for c in passages],
                    "trace": self._trace(st, "answer", "refused: evidence insufficient", t0)}
        text = " ".join(c["text"] + "".join(f"[{n}]" for n in c["cites"]) for c in claims)
        return {"answer": text, "claims": claims, "refused": False, "passages": [c["id"] for c in passages],
                "trace": self._trace(st, "answer", f"{len(claims)} claim(s), {len({n for c in claims for n in c['cites']})} citation(s)" + (f" ({attributed} attributed post-hoc: model gave no [n] markers)" if attributed else ""), t0, answer=text)}

    def _claims_from_text(self, txt: str) -> list[dict]:
        claims = []
        for s in re.split(r"(?<=[.!?\]\uff3d\u3011])\s+(?=[A-Z0-9])", txt.strip()):
            claims.append({"text": H.strip_cites(s), "cites": H.parse_cites(s), "computed": False})
        claims = [c for c in claims if c["text"]]
        allc = sorted({n for c in claims for n in c["cites"]})
        for c in claims:                      # models often cite once per paragraph: inherit rather than drop the claim
            if not c["cites"] and allc: c["cites"] = allc
        return claims

    def _attribute(self, claims: list[dict], passages: list[dict]) -> int:
        """If the model gave no citation at all, attach each claim to the passages that best support it (lexical overlap).
        The verifier still has the last word, so an unsupported claim is dropped, not rescued."""
        n = 0
        for c in claims:
            if c["cites"]: continue
            sc = sorted(((H.content_overlap(c["text"], ctx_text(p)), i) for i, p in enumerate(passages, 1)), reverse=True)
            c["cites"] = sorted(i for s, i in sc[:2] if s >= 0.3); c["attributed"] = bool(c["cites"]); n += bool(c["cites"])
        return n

    def _compose_offline(self, st, passages, num, tools_ok) -> list[dict]:
        q, plan = st["question"], st["plan"]
        cites = lambda srcs: sorted({num[s] for s in srcs if s in num})
        by = {r["tool"]: r for r in tools_ok}
        if "compare" in by:
            r = by["compare"]; rows = r["data"]["rows"]; key = r["data"]["key"]; srcs = r["sources"]
            label = FACT_SPEC[key][1]; ql = q.lower(); out = []
            if "same ministry" in ql and key == "ministry":
                same = len({x["value"] for x in rows}) == 1
                if same: out.append({"text": f"Yes. Both {rows[0]['doc_title']} and {rows[1]['doc_title']} are run by the {rows[0]['value']}.", "inputs": [rows[0]["value"].lower()]})
                else: out.append({"text": f"No. {rows[0]['doc_title']} is run by the {rows[0]['value']}, while {rows[1]['doc_title']} is run by the {rows[1]['value']}.", "inputs": [x["value"].lower() for x in rows]})
            else:
                vals = [x["value"] for x in rows]
                mx = re.search(r"\b(higher|larger|greater|bigger|more|later)\b", ql); mn = re.search(r"\b(lower|smaller|less|earlier|fewer)\b", ql)
                if (mx or mn) and all(isinstance(v, (int, float)) for v in vals) and len(set(vals)) > 1:
                    pick = max(rows, key=lambda x: x["value"]) if (mx and not mn) else min(rows, key=lambda x: x["value"])
                    other = [x for x in rows if x is not pick][0]
                    word = (mx or mn).group(1)
                    out.append({"text": f"{pick['doc_title']} has the {word} {label} ({pick['raw']} vs {other['raw']}).", "inputs": [str(v) for v in vals]})
                else:
                    out.append({"text": "; ".join(f"{x['doc_title']}: {label} is {x['raw']}" for x in rows) + ".", "inputs": [str(v) for v in vals]})
            return [{"text": o["text"], "cites": cites(srcs), "computed": True, "inputs": o["inputs"]} for o in out]
        if "calculator" in by:
            npn = plan["numeric"]; calc = by["calculator"]; val = calc["data"]["value"]
            srcs = [s for r in tools_ok if r["tool"] == "lookup" for s in r["sources"]]
            inputs = [str(r["data"]["value"]) for r in tools_ok if r["tool"] == "lookup"]
            title = " and ".join(self.docs[d]["title"] for d in npn["docs"])
            text = f"{npn['what'].capitalize()} for {title}: {npn.get('unit', '')}{inr(val)}{npn.get('suffix', '')} (computed as {calc['args']['expression']})."
            return [{"text": text, "cites": cites(srcs), "computed": True, "inputs": inputs}]
        if "lookup" in by:
            r = by["lookup"]; d = r["data"]
            return [{"text": f"{d['doc_title']}: {d['label']} is {d['raw']}.", "cites": cites(r["sources"]), "computed": False}]
        ign = H.entity_tokens(plan["entities"], self.docs)
        best = H.best_sentences(q, [self.r.chunks[e["idx"]] for e in st["evidence"][:5]], self.r, ign, 1)
        if not best: return []
        _, sent, ch = best[0]
        return [{"text": sent, "cites": [num[ch["id"]]] if ch["id"] in num else [], "computed": False}]

    # ---- verify
    def verify(self, st):
        t0 = time.perf_counter()
        if st["refused"]:
            return {"verified": True, "trace": self._trace(st, "verify", "nothing to verify (refusal)", t0)}
        ids = st["passages"]; by_id = {c["id"]: c for c in self.r.chunks}
        pas = [ctx_text(by_id[i]) for i in ids]
        kept, dropped = [], []
        llm_flags = None
        if not self.llm.offline:
            tools_txt = "\n".join(f"TOOL RESULT {r['tool']}({r['args']}) = {r['output']}" for r in st.get("tool_results", []) if r["ok"])
            user = f"Question (facts stated in the question are given and may be relied on): {st['question']}\n\nPassages:\n" + "\n".join(f"[{i}] {t}" for i, t in enumerate(pas, 1)) + (f"\n\nTool results (computed from the passages; arithmetic is correct):\n{tools_txt}" if tools_txt else "") + "\n\nClaims:\n" + "\n".join(
                f"{i}. {c['text']} (cited: {c['cites']})" for i, c in enumerate(st["claims"], 1))
            out = self._llm(st, P.VERIFY_SYSTEM, user, lambda: None)
            if out and isinstance(out.get("results"), list):
                llm_flags = {int(r["id"]): bool(r["supported"]) for r in out["results"] if "id" in r}
        for i, c in enumerate(st["claims"], 1):
            cited = [pas[n - 1] for n in c["cites"] if 0 < n <= len(pas)]
            if not cited: ok = False
            elif c.get("computed") and c.get("inputs") is not None:
                ok = H.claim_support(c["text"], [p.lower() for p in cited], [str(x).lower() for x in c["inputs"]]) >= 1.0
            else:
                ok = H.claim_support(c["text"], cited) >= 0.7
            if llm_flags is not None and i in llm_flags: ok = llm_flags[i] and bool(cited)   # LLM verdict wins; claim must still carry a citation
            (kept if ok else dropped).append(c)
        if not self.llm.offline:   # keep token accounting honest for the heuristic path
            pass
        else:
            self._rs()["tokens"] += approx_tokens("".join(pas) + st["answer"])
        if not kept:
            return {"answer": "I can't answer this reliably: none of the draft claims were supported by the cited passages.",
                    "claims": [], "refused": True, "verified": False,
                    "trace": self._trace(st, "verify", f"all {len(dropped)} claim(s) unsupported -> refuse", t0,
                                         dropped_claims=[f"{c['text'][:220]} (cites {c['cites']})" for c in dropped], llm_flags=llm_flags)}
        text = " ".join(c["text"] + "".join(f"[{n}]" for n in c["cites"]) for c in kept)
        msg = "all claims supported" if not dropped else f"revised: dropped {len(dropped)} unsupported claim(s)"
        return {"answer": text, "claims": kept, "verified": not dropped,
                "trace": self._trace(st, "verify", msg, t0, kept=len(kept), dropped=len(dropped),
                                     dropped_claims=[f"{c['text'][:220]} (cites {c['cites']})" for c in dropped], llm_flags=llm_flags)}

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(State)
        for n in ("plan", "retrieve", "grade", "rewrite", "tools", "answer", "verify"):
            g.add_node(n, getattr(self, n))
        g.set_entry_point("plan")
        g.add_edge("plan", "retrieve"); g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self.route_after_grade, {"rewrite": "rewrite", "tools": "tools"})
        g.add_conditional_edges("rewrite", self.route_after_rewrite, {"retrieve": "retrieve", "tools": "tools"}); g.add_edge("tools", "answer"); g.add_edge("answer", "verify"); g.add_edge("verify", END)
        return g.compile()

    def run(self, question: str) -> dict:
        t0 = time.perf_counter(); terminated = True
        self._tl.rs = {"tokens": 0, "degraded": False, "llm_error": ""}
        try:
            st = self.app.invoke({"question": question, "trace": []}, {"recursion_limit": 40})
        except LLMQuotaError:
            raise
        except Exception as e:     # never crash a request
            st = {"answer": f"Internal error: {type(e).__name__}", "refused": True, "trace": [], "loops": 0, "claims": [], "passages": [], "tool_results": []}
            terminated = False
        by_id = {c["id"]: c for c in self.r.chunks}
        pas = st.get("passages", []); cites = []
        for c in st.get("claims", []):
            for n in c["cites"]:
                if 0 < n <= len(pas) and by_id[pas[n - 1]] not in [x["chunk"] for x in cites]:
                    ch = by_id[pas[n - 1]]
                    cites.append({"n": n, "chunk": ch, "doc": ch["doc_id"], "doc_title": ch["doc_title"], "section": ch["section"], "page": ch["page"], "page_end": ch["page_end"], "chunk_id": ch["id"], "text": ch["text"][:400]})
        for x in cites: x.pop("chunk", None)
        cites = sorted({c["chunk_id"]: c for c in cites}.values(), key=lambda c: c["n"])
        tr = st.get("tool_results", [])
        return {"system": "agent", "question": question, "answer": st.get("answer", ""), "refused": bool(st.get("refused")),
                "citations": cites, "trace": st.get("trace", []), "tokens": self._rs()["tokens"],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "steps": len(st.get("trace", [])),
                "loops": st.get("loops", 0), "tool_calls": [{"tool": r["tool"], "ok": r["ok"]} for r in tr],
                "terminated": terminated and st.get("loops", 0) <= MAX_LOOPS, "degraded": self._rs()["degraded"], "llm_error": self._rs()["llm_error"],
                "retrieved": [c for c in st.get("passages", [])], "engine": ENGINE}
