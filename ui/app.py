"""Streamlit UI: answer + citations + retrieved chunks + the agent's step-by-step trace.
streamlit run ui/app.py     (in-process by default; set API_URL to talk to the FastAPI service instead)"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st

API_URL = os.getenv("API_URL")
st.set_page_config(page_title="Agentic RAG", page_icon=":mag:", layout="wide")


@st.cache_resource
def get_service():
    from api.service import service
    service.ensure(); return service


def ask(q, system):
    if API_URL:
        import urllib.request
        req = urllib.request.Request(API_URL.rstrip("/") + "/query", data=json.dumps({"question": q, "system": system}).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())
    return get_service().query(q, system)


st.title("Agentic RAG Document Assistant")
st.caption("Plans retrieval, uses tools, cites sources, refuses when evidence is missing.")
with st.sidebar:
    system = st.selectbox("System", ["agent", "hybrid_rerank_rag", "naive_rag"], format_func={"agent": "Full agent", "hybrid_rerank_rag": "Hybrid + rerank", "naive_rag": "Naive RAG"}.get)
    if not API_URL:
        h = get_service().health(); st.markdown(f"**LLM:** `{h['llm']}`  \n**Docs / chunks:** {h['docs']} / {h['chunks']}  \n**Embedder:** `{h['embedder']}`  \n**Reranker:** `{h['reranker']}`")
    gp = Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"
    examples = [json.loads(l)["question"] for l in gp.read_text().splitlines()[:40]] if gp.exists() else []
    pick = st.selectbox("Example questions", [""] + examples)

q = st.text_input("Ask a question about the documents", value=pick)
if st.button("Ask", type="primary") and q.strip():
    with st.spinner("Thinking..."):
        try:
            res = ask(q, system)
        except Exception as e:
            st.error(f"Backend error: {e}"); st.stop()
    c1, c2 = st.columns([3, 2])
    with c1:
        (st.warning if res["refused"] else st.success)("Refused - insufficient evidence" if res["refused"] else "Answer")
        st.markdown(res["answer"])
        if res.get("degraded"): st.info("LLM rate-limited/unavailable: fell back to the rule-based backend for part of this answer.")
        if res.get("cached"): st.caption("served from cache")
        st.markdown("#### Citations")
        for c in res["citations"] or []:
            with st.expander(f"[{c['n']}] {c['doc_title']} > {c['section']} (p.{c['page']})"):
                st.write(c["text"])
        if not res["citations"]: st.caption("none")
    with c2:
        m = st.columns(4)
        m[0].metric("Steps", res["steps"]); m[1].metric("Loops", res["loops"]); m[2].metric("Tokens~", res["tokens"]); m[3].metric("ms", int(res["latency_ms"]))
        st.markdown("#### Agent trace")
        for t in res["trace"]:
            icon = {"plan": "🧭", "retrieve": "🔎", "grade": "⚖️", "rewrite": "✏️", "tools": "🧮", "answer": "💬", "verify": "✅"}.get(t["node"], "•")
            with st.expander(f"{t['step']}. {icon} {t['node']} - {t['summary'][:90]}"):
                st.caption(f"{t['ms']} ms | cumulative tokens ~{t['tokens']}")
                st.json(t.get("data", {}), expanded=False)
    with st.expander("Retrieved chunks (ids)"):
        st.write(res.get("retrieved", []))
