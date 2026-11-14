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
> **Read this first.** This run used the *offline rule-based backend* (no LLM) on a *synthetic* corpus. It proves the pipeline, tools, graph, metrics and refusal logic work end to end, but it is **not evidence of answer quality**: the single-shot baselines can only extract sentences (they cannot compare or do arithmetic) and the agent's rules were written by the same author as the question templates. Re-run with `LLM_PROVIDER=groq JUDGE_PROVIDER=hf make eval` on your real corpus and hand-written golden set for numbers you can quote.

*80 golden questions (68 answerable). Generator: `offline-rules`; judge: `offline-rules (rule-based)`; retrieval: `lsa(tfidf+svd) + BM25 + lexical-rerank; store=numpy`; graph engine: `minigraph`.*

### Retrieval ablation (answerable questions, page-level gold)

| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| Dense only | 63.2 | 88.2 | 92.6 | 80.1 | 88.2 | 0.761 |
| BM25 only | 69.1 | 88.2 | 92.6 | 79.4 | 85.3 | 0.794 |
| Hybrid (RRF) | 64.7 | 91.2 | 94.1 | 80.9 | 89.7 | 0.773 |
| Hybrid + rerank | 82.4 | 89.7 | 92.6 | 80.9 | 87.5 | 0.870 |

### Three-system comparison

| Metric | Naive RAG | Hybrid + rerank | Full agent |
|---|---|---|---|
| Correctness, all questions (%) | 8.8 | 35.0 | 100.0 |
| Correctness, answerable only (%) | 10.3 | 33.8 | 100.0 |
| Faithfulness (%) | 100.0 | 100.0 | 90.9 |
| Citation hit (gold page cited) (%) | 63.2 | 60.3 | 100.0 |
| Cited text contains answer (%) | 33.8 | 45.6 | 67.6 |
| Refusal precision (%) | 0.0 | 29.4 | 100.0 |
| Refusal recall (%) | 0.0 | 41.7 | 100.0 |
| False refusals (count) | 0 | 12 | 0 |
| Avg steps / question | 2.0 | 2.0 | 6.9 |
| Tool-call success (%) | - | - | 92.7 |
| Loop termination (%) | 100.0 | 100.0 | 100.0 |
| Latency p50 (ms) | 0 | 2 | 6 |
| Latency p95 (ms) | 4 | 4 | 11 |
| Tokens / query (est.) | 372 | 649 | 839 |

### Correctness by question type (%)

| System | single | multihop | numerical | unanswerable | ambiguous |
|---|---|---|---|---|---|
| Naive RAG (dense top-3) | 25.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Hybrid + rerank (single shot) | 82.1 | 0.0 | 0.0 | 41.7 | 0.0 |
| Full agent | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

### Findings (generated from the run)

- On **multi-hop / comparison** questions the agent beats single-shot hybrid+rerank by 100 points (100% vs 0%).
- On **numerical** questions the agent beats single-shot hybrid+rerank by 100 points (100% vs 0%).
- On **single-fact** questions the agent beats single-shot hybrid+rerank by 18 points (100% vs 82%).
- On **ambiguous-wording** questions the agent beats single-shot hybrid+rerank by 100 points (100% vs 0%).
- On **unanswerable (refusal)** questions the agent beats single-shot hybrid+rerank by 58 points (100% vs 42%).
- Cost of the agent: 6.9 graph steps and 839 (estimated) tokens per query vs 649 for hybrid+rerank (1.3x); p95 latency 11 ms vs 4 ms.
- Naive RAG never refuses (refusal recall 0%); the agent reaches 100% recall at 100% precision, with 0 false refusals on answerable questions.

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
