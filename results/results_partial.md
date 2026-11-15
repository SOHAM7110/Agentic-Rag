> **Run stopped early - results below cover only the questions finished.** STOPPED at agent question 24/60 (r2_020): rate limit: server asks to wait 259s (4 min). b'{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m7cs82q1e6bvev50bmkcbefe` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199245, Re'

*60 golden questions (78 answerable). Generator: `groq:openai/gpt-oss-120b`; judge: `groq:openai/gpt-oss-20b`; retrieval: `st:BAAI/bge-small-en-v1.5 + BM25 + cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2; store=chroma`; graph engine: `langgraph`.*

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
| Correctness, all questions (%) | 68.3 | 88.3 | 91.3 |
| Correctness, answerable only (%) | 80.4 | 86.3 | 90.5 |
| Faithfulness (%) | 74.4 | 91.5 | 97.4 |
| Citation hit (gold page cited) (%) | n/a | 82.4 | 85.7 |
| Cited text contains answer (%) | n/a | 76.5 | 85.7 |
| Refusal precision (%) | 0.0 | 69.2 | 50.0 |
| Refusal recall (%) | 0.0 | 100.0 | 100.0 |
| False refusals (count) | 0 | 4 | 2 |
| Avg steps / question | 2.0 | 2.0 | 6.8 |
| Tool-call success (%) | - | - | - |
| Loop termination (%) | 100.0 | 100.0 | 100.0 |
| Degraded answers (LLM fell back to rules) | 0 | 0 | 0 |
| Latency p50 (ms) | 732 | 4907 | 2652 |
| Latency p95 (ms) | 6322 | 5894 | 23767 |
| Tokens / query (est.) | 861 | 1413 | 4362 |

*Naive RAG is not asked to cite (its citation metrics are n/a). Hybrid and the agent use the same citation parser and the same post-hoc attribution rule. Latency includes any client-side throttling and excludes cached calls; use a cold run for latency claims.*

### Correctness by question type (%)

| System | single | multihop | numerical | unanswerable | ambiguous |
|---|---|---|---|---|---|
| Naive RAG (dense top-3) | 91.7 | 63.6 | 77.8 | 0.0 | 71.4 |
| Hybrid + rerank (single shot) | 100.0 | 63.6 | 77.8 | 100.0 | 85.7 |
| Full agent | 100.0 | 50.0 | - | 100.0 | 100.0 |

### Findings (generated from the run)

- On **multi-hop / comparison** questions the agent is *worse* than single-shot hybrid+rerank by 14 points (50% vs 64%).
- On **numerical** questions the agent is *worse* than single-shot hybrid+rerank by 78 points (0% vs 78%).
- On **single-fact** questions the agent is on par with single-shot hybrid+rerank (100% vs 100%): no measurable gain.
- On **ambiguous-wording** questions the agent beats single-shot hybrid+rerank by 14 points (100% vs 86%).
- On **unanswerable (refusal)** questions the agent is on par with single-shot hybrid+rerank (100% vs 100%): no measurable gain.
- Cost of the agent: 6.8 graph steps and 4362 (estimated) tokens per query vs 1413 for hybrid+rerank (3.1x); p95 latency 23767 ms vs 5894 ms.
- Naive RAG never refuses (refusal recall 0%); the agent reaches 100% recall at 50% precision, with 2 false refusals on answerable questions.

**Judge vs. human agreement:** not measured yet - run `python -m eval.hand_grade sample`, label 20 answers, then `python -m eval.hand_grade agree`.
