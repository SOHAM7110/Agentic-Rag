# Agentic RAG Document Assistant (with an evaluation harness)

A question-answering system over a corpus of long PDFs that **plans its own retrieval, uses tools, cites its sources
(doc, section, page), refuses when evidence is missing, and ships with an automated evaluation suite** that produces a
results table with one command (`make eval`).

> **Status of the numbers below.** The repo ships with a *synthetic* 30-PDF demo corpus and a rule-based *offline* backend
> so everything runs with zero downloads and zero API cost. Those numbers prove the machinery works; they are **not** evidence of
> LLM answer quality. To get numbers you can put on a resume: swap in real PDFs, hand-write the golden set, set
> `LLM_PROVIDER=groq` + a different `JUDGE_PROVIDER`, and run `make eval`. See *Going from demo to real* below.

## Architecture
```
PDFs -> parse (PyMuPDF | pdfplumber) -> section-aware chunks (+doc/section/page metadata) -> embeddings -> ChromaDB | numpy
                                                                 |-> BM25 index        \-> facts table (structured lookup)
question
  v
[LangGraph state graph]            (bundled mini-engine if langgraph isn't installed)
  1 plan      classify (single / compare / numeric), link scheme names, decompose, pick tools
  2 retrieve  hybrid (dense + BM25) -> RRF fusion -> cross-encoder rerank; per-entity sub-queries; descriptor -> doc hop
  3 grade     is the evidence sufficient?  no -> 4 rewrite -> back to 2     (max 3 retrieval loops, token budget)
  5 tools     calculator (safe AST) | structured lookup (facts table) | compare-two-documents
  6 answer    grounded: cite every claim, flag uncertainty, refuse if evidence is missing
  7 verify    every claim must be supported by its cited chunk; unsupported claims are dropped, or the answer is refused
  v
FastAPI (/query /ingest /health, JSONL request log)  ->  Streamlit UI showing answer, citations, chunks and the step-by-step trace
```
Every model call goes through one interface (`llm/`): `offline | groq | hf | ollama | openai`, with retries, a SQLite response cache and a
**graceful fallback** to the rule-based backend if the API is rate-limited or malformed, so a public demo never crashes.

## Quickstart
```bash
pip install -r requirements-lite.txt      # or requirements.txt for the full stack (PyMuPDF, Chroma, bge-small, cross-encoder, LangGraph, FastAPI, Streamlit)
make demo-corpus                          # (already generated) 30 synthetic PDFs + 80-question golden set
make eval                                 # index -> ablation -> 3-system comparison -> results/ + this README table
make test
make ui                                   # http://localhost:8501   |   make api -> http://localhost:8000/docs
```
Real LLM run (free tier): `export GROQ_API_KEY=... HF_TOKEN=...; LLM_PROVIDER=groq JUDGE_PROVIDER=hf make eval`

## Results
<!-- RESULTS:START -->
*14 golden questions (78 answerable). Generator: `groq:openai/gpt-oss-120b`; judge: `groq:openai/gpt-oss-20b`; retrieval: `st:BAAI/bge-small-en-v1.5 + BM25 + cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2; store=chroma`; graph engine: `langgraph`.*

### Retrieval ablation (answerable questions, page-level gold)

| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| Dense only | 56.4 | 85.9 | 91.0 | 81.4 | 87.2 | 0.712 |
| BM25 only | 52.6 | 73.1 | 88.5 | 68.6 | 85.9 | 0.655 |
| Hybrid (RRF) | 60.3 | 82.1 | 92.3 | 79.5 | 89.7 | 0.739 |
| Hybrid + rerank | 75.6 | 89.7 | 98.7 | 86.5 | 96.2 | 0.838 |

### Three-system comparison

| Metric | Naive RAG (dense top-3) | Hybrid + rerank (single shot) | Full agent |
|---|---|---|---|
| Correctness, all questions (%) | 0.0 | 100.0 | 100.0 |
| Correctness, answerable only (%) | 0.0 | 0.0 | 0.0 |
| Faithfulness (%) | 19.0 | 0.0 | 0.0 |
| Citation hit (gold page cited) (%) | n/a | 0.0 | 0.0 |
| Cited text contains answer (%) | n/a | 0.0 | 0.0 |
| Refusal precision (%) | 0.0 | 100.0 | 100.0 |
| Refusal recall (%) | 0.0 | 100.0 | 100.0 |
| False refusals (count) | 0 | 0 | 0 |
| Avg steps / question | 2.0 | 2.0 | 12.0 |
| Tool-call success (%) | - | - | - |
| Loop termination (%) | 100.0 | 100.0 | 100.0 |
| Degraded answers (LLM fell back to rules) | 0 | 0 | 0 |
| Latency p50 (ms) | 20 | 954 | 29317 |
| Latency p95 (ms) | 6028 | 5764 | 36142 |
| Tokens / query (est.) | 812 | 1307 | 5676 |
| Correct / n (95% Wilson interval) | 0/14 (0-22%) | 14/14 (78-100%) | 14/14 (78-100%) |

*Naive RAG is not asked to cite (its citation metrics are n/a). Hybrid and the agent use the same citation parser and the same post-hoc attribution rule. Latency includes any client-side throttling and excludes cached calls; use a cold run for latency claims.*

### Correctness by question type (%)

| System | single | multihop | numerical | unanswerable | ambiguous |
|---|---|---|---|---|---|
| Naive RAG (dense top-3) | - | - | - | 0 (0/14) | - |
| Hybrid + rerank (single shot) | - | - | - | 100 (14/14) | - |
| Full agent | - | - | - | 100 (14/14) | - |

### Findings (generated from the run)

- All comparisons below use the **14 questions that every system completed**.
- Paired, agent vs hybrid+rerank: agent right & hybrid+rerank wrong on 0 question(s); hybrid+rerank right & agent wrong on 0 (exact sign test p=1.00) - no statistically detectable difference.
- Paired, agent vs naive RAG: agent right & naive RAG wrong on 14 question(s); naive RAG right & agent wrong on 0 (exact sign test p=0.00) - a statistically detectable difference.
- Paired, hybrid+rerank vs naive RAG: hybrid+rerank right & naive RAG wrong on 14 question(s); naive RAG right & hybrid+rerank wrong on 0 (exact sign test p=0.00) - a statistically detectable difference.
- **multi-hop / comparison**: no questions of this type were completed by all systems yet.
- **numerical**: no questions of this type were completed by all systems yet.
- **single-fact**: no questions of this type were completed by all systems yet.
- **tricky-wording**: no questions of this type were completed by all systems yet.
- **unanswerable (refusal)** (14 questions): agent 14/14 vs hybrid+rerank 14/14 (within one question - noise).
- Cost of the agent: 12.0 graph steps and 5676 tokens per query vs 1307 for hybrid+rerank (4.3x); p95 latency 36142 ms vs 5764 ms (latency is distorted by caching/throttling).
- Refusal (14 unanswerable questions): naive RAG recall 0%; hybrid+rerank recall 100% at 100% precision (0 false refusals); the agent recall 100% at 100% (0 false refusals).

**Judge vs. human agreement:** not measured yet - run `python -m eval.hand_grade sample`, label 20 answers, then `python -m eval.hand_grade agree`.
<!-- RESULTS:END -->

Plots: `results/retrieval_ablation.png`, `results/correctness_by_type.png`. Per-question outputs: `results/per_question.jsonl`.

### How to read the metrics
- **Retrieval** (Hit@k, Recall@k, MRR): page-level gold. A chunk is relevant if it is in a gold doc and overlaps a gold page. Recall@k = fraction of gold documents covered.
- **Correctness / faithfulness**: LLM judge with the rubric in `agent/prompts.py` (rule-based judge when `JUDGE_PROVIDER=offline`). Use a *different* model than the generator, and hand-grade 20 answers (`make hand-grade`, fill the CSV, `python -m eval.hand_grade agree`) to report judge agreement.
- **Citation hit**: every gold document has a cited chunk on an acceptable page. **Cited text contains answer**: the cited text itself holds the answer keys (computed answers such as "each instalment = 36000/3" legitimately score 0 here because the result isn't in the text).
- **Refusal precision / recall**: on `unanswerable` questions; precision counts every refusal, so false refusals on answerable questions lower it.
- **Agent**: avg graph steps, tool-call success, loop-termination rate. **Ops**: p50/p95 latency, tokens/query (API usage when a real LLM is used; estimated prompt+completion size offline).

## Design decisions (and honest caveats)
- **Section-aware chunking**, not fixed tokens: chunks never cross a heading; table rows stay whole; metadata (doc, section, page) powers citations *and* evaluation.
- **Golden set written before the agent** (stage 3). The demo set is generated from structured facts, then each gold page is *verified by re-parsing the PDFs*.
- **Offline backend caveats.** (1) Single-shot baselines can only extract sentences, so multi-hop/numeric gaps are structural; a real LLM narrows them (8B models still make arithmetic and comparison mistakes - that is what the calculator/compare tools are for). (2) The agent's rules and the question templates share an author, so ~100% on the synthetic set is expected and means little. (3) The baseline refusal threshold (0.45 coverage) was picked by eye from the coverage distribution on this same set; the agent thresholds (0.6) were fixed a priori. (4) The rule-based faithfulness judge is stricter than the agent's verifier on derived statements ("X has the higher ..."), which is why agent faithfulness is below 100%. Replace the judge with an LLM for real numbers.
- **Fallbacks exist for every heavy dependency** (LSA instead of bge-small, numpy instead of Chroma, lexical instead of cross-encoder, mini state-graph instead of LangGraph). The ablation therefore shows *which* backend ran (see the header line of the results). The sentence-transformers / Chroma / cross-encoder / LangGraph / PyMuPDF / FastAPI / Streamlit code paths were written to the documented APIs but **not executed in the build sandbox (no network)** - run `make test && make eval` after `pip install -r requirements.txt` and expect to fix small integration issues.
- **Facts table** is extracted from "key | value" table rows via `ingest/facts.py` (`FACT_SPEC` holds labels, question cues and evidence regexes). For a new corpus, edit `FACT_SPEC` or plug in an LLM extractor with the same output schema.

## Going from demo to real
1. `rm data/pdfs/*.pdf`; put 15-30 real PDFs there (`scripts/download_corpus.py urls.txt` helps). Good free options: PM-KISAN / Ayushman Bharat / state scheme guidelines, regulator circulars, open-source library docs.
2. Hand-write 60-100 questions into `eval/golden_set.jsonl` (format in `eval/README.md`; mix 35% single, 25% multi-hop, 15% numeric, 15% unanswerable, 10% tricky). Write them before tuning anything.
3. Adapt the aspect cues in `FACT_SPEC` (`ingest/facts.py`) to your documents; entity linking uses document titles automatically.
4. `pip install -r requirements.txt`, set `LLM_PROVIDER` / `JUDGE_PROVIDER`, `make eval`, then `make hand-grade`.
5. Deploy: `deploy/HF_SPACES.md` (Docker + secrets).

## Repo layout
```
ingest/     parse.py chunk.py facts.py build_index.py        retrieval/  embed.py store.py bm25.py fusion.py rerank.py retriever.py
agent/      graph.py (state graph + nodes) tools.py heuristics.py prompts.py baselines.py minigraph.py
llm/        base.py providers.py (Groq/HF/Ollama/OpenAI-compatible, retry+cache)
eval/       golden_set.jsonl metrics.py judge.py run_eval.py hand_grade.py     api/  service.py main.py     ui/  app.py
scripts/    make_demo_corpus.py download_corpus.py     tests/  test_core.py     results/  tables, plots, per-question outputs
```

## Interview prep (answers live in the results)
- *Why hybrid?* -> the retrieval ablation table (dense vs BM25 vs hybrid vs +rerank).  *How do you know it isn't hallucinating?* -> faithfulness, citation hit, refusal P/R, verifier node.
- *When does the agent hurt?* -> the generated Findings (latency/token cost; types with no gain).  *How do you trust the judge?* -> judge-vs-human agreement.
- *How would you scale it?* -> bigger index (Chroma/pgvector/Qdrant), query + embedding caches, async LLM calls, batch ingestion, moving BM25 to OpenSearch.

## Resume bullets (fill from `results/results.md` after a real run)
- Built an agentic RAG assistant (LangGraph; hybrid dense+BM25 retrieval with RRF + cross-encoder rerank; calculator / structured-lookup / compare tools; claim-level verifier) over **N** PDFs / **M** chunks.
- Evaluated on **N_q** hand-labelled questions: Hit@5 **__%**, faithfulness **__%**, citation accuracy **__%**, refusal recall **__%**; p95 latency **__ s**; judge agreement **__%** with human grading.
- Shipped FastAPI + Streamlit (agent trace view) in Docker on Hugging Face Spaces: **<demo link>**.
