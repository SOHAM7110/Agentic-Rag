> **Run stopped early - results below cover only the questions finished.** STOPPED at naive_rag question 7/30 (r2_022): rate limit: server asks to wait 166s (3 min). b'{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m7cs82q1e6bvev50bmkcbefe` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199351, Re'

*30 golden questions (78 answerable). Generator: `groq:openai/gpt-oss-120b`; judge: `groq:openai/gpt-oss-20b`; retrieval: `st:BAAI/bge-small-en-v1.5 + BM25 + cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2; store=chroma`; graph engine: `langgraph`.*

### Retrieval ablation (answerable questions, page-level gold)

| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| Dense only | 56.4 | 85.9 | 91.0 | 81.4 | 87.2 | 0.712 |
| BM25 only | 52.6 | 73.1 | 88.5 | 68.6 | 85.9 | 0.655 |
| Hybrid (RRF) | 60.3 | 82.1 | 92.3 | 79.5 | 89.7 | 0.739 |
| Hybrid + rerank | 75.6 | 89.7 | 98.7 | 86.5 | 96.2 | 0.838 |

### Three-system comparison

| Metric | Naive RAG (dense top-3) |
|---|---|
| Correctness, all questions (%) | 50.0 |
| Correctness, answerable only (%) | 50.0 |
| Faithfulness (%) | 83.3 |
| Citation hit (gold page cited) (%) | n/a |
| Cited text contains answer (%) | n/a |
| Refusal precision (%) | 0.0 |
| Refusal recall (%) | 0.0 |
| False refusals (count) | 0 |
| Avg steps / question | 2.0 |
| Tool-call success (%) | - |
| Loop termination (%) | 100.0 |
| Degraded answers (LLM fell back to rules) | 0 |
| Latency p50 (ms) | 2 |
| Latency p95 (ms) | 6244 |
| Tokens / query (est.) | 854 |
| Correct / n (95% Wilson interval) | 3/6 (19-81%) |

*Naive RAG is not asked to cite (its citation metrics are n/a). Hybrid and the agent use the same citation parser and the same post-hoc attribution rule. Latency includes any client-side throttling and excludes cached calls; use a cold run for latency claims.*

### Correctness by question type (%)

| System | single | multihop | numerical | unanswerable | ambiguous |
|---|---|---|---|---|---|
| Naive RAG (dense top-3) | - | 40 (2/5) | 100 (1/1) | - | - |

### Findings (generated from the run)

- (findings need all three systems - run without --systems)

**Judge vs. human agreement:** not measured yet - run `python -m eval.hand_grade sample`, label 20 answers, then `python -m eval.hand_grade agree`.
