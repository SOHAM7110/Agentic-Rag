# Agentic RAG over RBI Directions - with an evaluation harness

A question-answering system for long regulatory PDFs. It **plans its own retrieval, searches inside the documents a question names,
uses tools (calculator, structured lookup, compare), cites document / section / page, refuses when the evidence is missing, and verifies
each claim against its cited passage** - and it ships with an automated evaluation that compares it against two simpler baselines
on a hand-checked golden set.

* **Corpus:** 27 RBI Master Directions / Directions (2023-2026): commercial-bank rules (interest rates on advances, income recognition & asset
  classification, resolution of stressed assets, KYC, concentration risk, securitisation, dividends, ALM ...), payments-bank rules and six
  near-identical *Internal Ombudsman* directions for different entity types (a deliberate hard case for retrieval).
* **Golden set:** 92 questions (35 single-fact, 18 multi-hop/comparison, 15 numerical, 14 unanswerable, 10 tricky wording), each with a gold answer
  and gold document/page. `make golden` re-verifies every gold page against the PDFs.
* **Models:** bge-small embeddings, BM25, RRF fusion, MiniLM cross-encoder rerank, ChromaDB, LangGraph; generator and judge are two different
  free Groq models (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`). Everything runs offline too (dependency-free fallbacks).

> **Honest status.** The code, tests and golden set are complete, but **there is no final generation result yet**: the free-tier quota (200k tokens/day)
> stopped earlier runs, and the last partial run happened to cover only unanswerable questions (the easiest case: 100 % just means "refused every time").
> The README's results block therefore stays a placeholder until you run the sampled evaluation below. Earlier development runs *did* include answerable
> questions, but they predate the final fixes and their cache is invalid. Judge-vs-human agreement is **not measured until you hand-grade** (`make hand-grade`).

## What is compared (four systems, so no baseline is a strawman)
| System | Retrieval | Prompt | Isolates |
|---|---|---|---|
| Naive RAG | dense top-3 | plain "answer from the context" (no refusal, no citations) | the starting point |
| Dense + grounded | dense top-5 | grounded + cites + may refuse | **value of the prompt** (vs naive) |
| Hybrid + rerank + grounded | hybrid, RRF, cross-encoder top-5 | same grounded prompt | **value of retrieval** (vs dense + grounded) |
| Full agent | hybrid + rerank, per-document search, retries | plan / grade / rewrite / tools / verify | **value of the agent loop** (vs hybrid + grounded) |

## Architecture
```
PDFs -> parse (PyMuPDF | pdfplumber) -> section-aware chunks (+doc/section/page; tables whole with their lead-in sentence) -> bge-small -> ChromaDB
                                                                       \-> BM25
question
  v
[LangGraph state graph]  (bundled mini engine if langgraph is not installed)
  1 plan      type (single/compare/numeric) + sub-questions (one per distinct fact) + link NAMED documents ("payments banks ... small finance banks")
  2 retrieve  hybrid (dense+BM25) -> RRF -> cross-encoder rerank; one search INSIDE each linked document; evidence is kept across loops
  3 grade     is the evidence sufficient? (sees exactly the passages the answerer will see)
       no -> 4 rewrite: query targeted at what the grader says is missing -> back to 2   (max 3 loops, token budget, stops if a retry finds nothing new)
  5 tools     calculator (safe AST) | facts lookup | compare   (facts table is only used when the corpus has "label | value" tables)
  6 answer    grounded; cite every claim [n]; refuse with INSUFFICIENT_EVIDENCE when the passages do not answer
  7 verify    each claim must follow from its cited passages + facts in the question + tool results; unsupported claims are dropped or the answer is refused
  v
FastAPI (/query /ingest /health, JSONL log, LRU cache)  ->  Streamlit UI: answer, citations, retrieved chunks, step-by-step trace
```
One model interface (`llm/`): `offline | groq | hf | ollama | openai`, retries with back-off, SQLite response cache, **abort on long rate-limit waits**
(`LLMQuotaError`), graceful fallback to rules on malformed output.

## Quickstart (Windows PowerShell shown; Linux/macOS use `export` and `make`)
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                    # full stack; requirements-lite.txt = no model downloads
python -m tests.test_core ; python -m tests.test_llm_paths     # hermetic: no keys, no network
python -m ingest.build_index                       # ~5 min on CPU for 27 PDFs (CHUNK_MAX_CHARS defaults to 900)
python -m eval.run_eval --retrieval-only           # FREE: retrieval ablation with your real embedder + reranker
python scripts\check_linking.py                    # FREE: document-linking / per-document retrieval check
python -m streamlit run ui\app.py                  # UI (after setting the LLM in .env)
```
Copy `.env.example` to `.env` and put your Groq key in it (create one at console.groq.com -> API Keys; never commit it).

### Run the evaluation (stratified sample, ~2-3 days of free-tier quota)
Running everything is not necessary or affordable on the free tier (agent ~4-6k tokens/question, 200k tokens/day for `gpt-oss-120b`). Instead evaluate a
**seeded, stratified sample**: 12 single-fact, 12 multi-hop, 10 numerical, 6 tricky-wording **answerable** questions plus 6 unanswerable ones, all four systems,
with the eight debugging questions left out (`DEV_IDS` in the Makefile). Estimated cost ~420k tokens on one model; ~240k on the main model if you set
`LLM_CONTROL_MODEL=openai/gpt-oss-20b` (the agent's plan/grade/rewrite/verify steps then use the second model's quota). Every response is cached, so a rate-limit
stop (`STOPPED at ...`) just means "run the same command tomorrow".
```powershell
python -m eval.run_eval --stratified single=12,multihop=12,numerical=10,ambiguous=6,unanswerable=6 --exclude r1_001,r1_020,r2_013,r2_033,r3_022,r4_004,r1_024,r1_027 --publish
python -m eval.hand_grade sample          # 20 answers, 16 answerable, agent + baseline mixed; the judge's score is hidden in hand_grade_key.csv
#  ... fill the `human` column of results\hand_grade.csv (1/0), then:
python -m eval.hand_grade agree           # agreement + kappa overall / answerable / unanswerable, and whether the judge is lenient or strict
python scripts\latency_bench.py 8         # COLD, unthrottled latency (spends ~100k tokens); quote these, not the table's latency rows
```
`results/per_question.jsonl` accumulates across runs, so partial runs never erase earlier ones. Without `--publish`, sampled runs write `results_partial.*` and leave this README alone.

## Results
<!-- RESULTS:START -->
(run `python -m eval.run_eval` - see "Run the final evaluation")
<!-- RESULTS:END -->

### How to read the metrics
* **Retrieval**: Hit@k / Recall@k / MRR against *page-level* gold (a chunk is relevant if it is in a gold document and overlaps a gold page; Recall = share of gold documents covered).
* **Correctness / faithfulness**: LLM judge (rubric in `agent/prompts.py`) - generator and judge are different models, but same family: check with the hand-graded agreement.
* **Citation hit**: every gold document has a cited chunk on an acceptable page. Naive RAG is not asked to cite (n/a). Calculated answers legitimately score 0 on "cited text contains answer".
* **Refusal precision / recall**: on `unanswerable` questions; `n/a` when a subset has none.
* **Statistics**: all comparisons use only questions every system completed; the report gives Wilson 95% intervals and an exact sign test on discordant pairs. With ~30-80 questions, differences of 1-3 questions are noise.
* **Latency** includes client-side throttling and excludes cached calls - re-measure on a cold, unthrottled run before quoting it.

### Development snapshot (earlier runs, BEFORE the final fixes - superseded by your final run)
* Reranking is the clearest retrieval win (real stack, pre-chunk-fix: hybrid+rerank Hit@5 98.7%, MRR 0.84 vs dense-only 91.0% / 0.71).
* Grounded answering with refusal is the clearest end-to-end win over naive RAG: 14/14 unanswerable questions refused vs 0/14.
* The agent loop: 25/30 vs 21/30 for single-shot hybrid+rerank on the multi-hop + numerical questions (paired sign test p = 0.22, i.e. *not* established),
  at ~3.7x the tokens. Several of its failures were pipeline bugs, fixed afterwards (below) - so re-measure. These runs predate the dense-grounded baseline, so they could not separate prompting from retrieval.

## Engineering log: bugs the evaluation exposed (each now has a regression test)
| Symptom | Root cause | Fix |
|---|---|---|
| Agent refused 6/10 answerable questions | model cited with `【1】`/no markers -> parser saw 0 citations -> verifier dropped every claim | tolerant citation parser + post-hoc attribution + stricter prompt |
| Correct calculations rejected | verifier never saw the question or tool results | verifier prompt includes both |
| "Evidence sufficient" then refusal | grader judged 8 passages, answerer saw 4 | one shared passage window |
| Token counts / fallback flags wrong under LangGraph | LangGraph passes *copies* of state; in-place updates were lost | per-run thread-local counters; mini-engine now copies state too |
| Near-duplicate documents confused (six Ombudsman directions) | entity linker needed every title word | link by distinguishing title tokens; per-document search; entity words stripped from per-document queries |
| Answers cut mid-sentence ("...not less than three") | parser read hanging-indent paragraph numbers `(4)` as table cells: **2,402 of 3,100 table chunks were fake** | paragraph-marker rule |
| Real tables unfindable | one chunk per row, no header, wrapped cells split | whole-table chunks with their lead-in sentence |
| Rewrites full of "benefit amount / nodal ministry" | scheme-specific terms hard-coded in the rewrite prompt | generic prompt; rewrite told what is missing; evidence kept across loops; stop when a retry adds nothing |
| Run "hung" for an hour | rate-limit retries slept for minutes with no output | abort on long waits, per-question progress, incremental saves |
| `HTTP 403 error code: 1010` | Cloudflare blocks urllib's default User-Agent | explicit User-Agent |
| Empty answers from gpt-oss | hidden reasoning tokens consumed `max_tokens` | `reasoning_effort` + larger budget |
| Misleading partial tables | systems compared on different questions; empty subsets shown as 0% | pair on common questions, CIs, sign tests, `n/a` |

## Limitations
* The facts table / lookup tool needs "label | value" tables; it is unused for RBI directions (kept for the synthetic demo). The calculator is driven by the LLM.
* Same-family judge; hand-graded agreement (n=20) is a rough check, not proof.
* Retrieval is page-level; multi-paragraph answers that span chunks can still be missed (see `make diagnose`).
* The free tier limits how often you can re-run the agent; token use per agent question is ~4x a single-shot answer. Retries now never re-issue a query that was already tried and stop when a retry adds no new evidence, but the cost per answered question is still several times a single-shot answer.
* The evaluation is a stratified **sample** (about 46 questions), not all 92: intervals are wide and per-type cells are small, so report counts, intervals and paired tests, not single percentages.
* The offline rule-based backend and the synthetic corpus (`make demo`, `data/demo_*`) only demonstrate the machinery; the rules were written by the same author as the demo questions.
* RBI documents are public, but check the RBI website terms before redistributing the PDFs or hosting a public demo.

## Repo layout
```
ingest/     parse.py chunk.py facts.py build_index.py      retrieval/  embed.py store.py bm25.py fusion.py rerank.py retriever.py
agent/      graph.py (state graph) heuristics.py (linking, rewrite, offline rules) prompts.py tools.py baselines.py minigraph.py
llm/        base.py providers.py                           eval/  golden_round1-4.jsonl golden_set.jsonl merge_golden.py verify_golden.py
            run_eval.py metrics.py judge.py hand_grade.py  api/ service.py main.py      ui/ app.py
scripts/    diagnose.py check_linking.py prepare_corpus.py download_corpus.py make_demo_corpus.py     tests/  test_core.py test_llm_paths.py helpers.py
data/       pdfs/ (27 RBI PDFs)  demo_pdfs/ + demo_golden.jsonl (synthetic offline demo)         results/  (written by the eval)
```
`make help`-style cheat sheet: `make test | index | eval-retrieval | eval | demo | golden | diagnose IDS=r1_027 | check-linking | hand-grade`.

## Interview prep
*Why hybrid + rerank?* -> the ablation table. *How do you know it isn't hallucinating?* -> verifier node, faithfulness, citation hit, refusal recall, plus the
"faithful but wrong passage" failure we found by hand. *When does the agent hurt?* -> cost (tokens, steps) and the paired comparison per type.
*Why trust the judge?* -> hand-graded agreement. *What broke?* -> the engineering log above. *Scale?* -> bigger index (pgvector/Qdrant), async calls, query/embedding caches, batch ingestion.

## Resume bullets (fill from `results/results.md`)
* Built an agentic RAG system (LangGraph, hybrid dense+BM25 retrieval with RRF + cross-encoder rerank, calculator/compare tools, claim-level verifier) over **27** RBI directions (**N** chunks).
* Evaluated on a **92**-question hand-checked golden set against naive and single-shot baselines: Hit@5 **__%**, correctness **__%** (95% CI __-__%), refusal recall **__%**, judge-human agreement **__%**; reported paired significance and token cost.
* Found and fixed pipeline bugs via evaluation (paragraph-marker parsing, near-duplicate document linking, LangGraph state-copy accounting, verifier blind spots).
