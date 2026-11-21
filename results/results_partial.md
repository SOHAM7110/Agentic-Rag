> **Run stopped early - results below cover only the questions finished.** STOPPED at agent question 26/46 (r3_020): rate limit: server asks to wait 605s (10 min). b'{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m7cs82q1e6bvev50bmkcbefe` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199936, Re'

*46 questions in this generation evaluation (the retrieval ablation uses all 78 answerable golden questions). Generator: `groq:openai/gpt-oss-120b`; judge: `groq:openai/gpt-oss-20b`; retrieval: `st:BAAI/bge-small-en-v1.5 + BM25 + cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2; store=chroma`; graph engine: `langgraph`.*

### Retrieval ablation (answerable questions, page-level gold)

| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| Dense only | 60.3 | 84.6 | 91.0 | 80.1 | 87.2 | 0.733 |
| BM25 only | 55.1 | 75.6 | 88.5 | 71.2 | 86.5 | 0.676 |
| Hybrid (RRF) | 59.0 | 85.9 | 92.3 | 82.7 | 89.7 | 0.745 |
| Hybrid + rerank | 79.5 | 93.6 | 98.7 | 90.4 | 96.2 | 0.862 |

### Three-system comparison

| Metric | Naive RAG (dense top-3, no refusal prompt) | Dense + grounded/refusing prompt | Hybrid + rerank + grounded prompt | Full agent |
|---|---|---|---|---|
| Correctness, all questions (%) | 72.0 | 92.0 | 96.0 | 96.0 |
| Correctness, answerable only (%) | 78.3 | 91.3 | 95.7 | 95.7 |
| Faithfulness (%) | 86.0 | 87.9 | 91.3 | 93.2 |
| Citation hit (gold page cited) (%) | n/a | 82.6 | 100.0 | 95.7 |
| Cited text contains answer (%) | n/a | 73.9 | 87.0 | 87.0 |
| Refusal precision (%) | 0.0 | 66.7 | 100.0 | 66.7 |
| Refusal recall (%) | 0.0 | 100.0 | 100.0 | 100.0 |
| False refusals (count) | 0 | 1 | 0 | 1 |
| Avg steps / question | 2.0 | 2.0 | 2.0 | 6.6 |
| Tool-call success (%) | - | - | - | 100.0 |
| Loop termination (%) | 100.0 | 100.0 | 100.0 | 100.0 |
| Degraded answers (LLM fell back to rules) | 0 | 0 | 0 | 0 |
| Latency p50 (ms) | 4 | 6 | 756 | 24059 |
| Latency p95 (ms) | 26 | 8 | 6225 | 35722 |
| Tokens / query (est.) | 799 | 1283 | 1323 | 4494 |
| Correct / n (95% Wilson interval) | 18/25 (52-86%) | 23/25 (75-98%) | 24/25 (80-99%) | 24/25 (80-99%) |

*Naive RAG is not asked to cite (its citation metrics are n/a). Hybrid and the agent use the same citation parser and the same post-hoc attribution rule. Latency includes any client-side throttling and excludes cached calls; use a cold run for latency claims.*

### Correctness by question type (%)

| System | single | multihop | numerical | unanswerable | ambiguous |
|---|---|---|---|---|---|
| Naive RAG (dense top-3, no refusal prompt) | 82 (9/11) | 50 (2/4) | 67 (2/3) | 0 (0/2) | 100 (5/5) |
| Dense + grounded/refusing prompt | 100 (11/11) | 50 (2/4) | 100 (3/3) | 100 (2/2) | 100 (5/5) |
| Hybrid + rerank + grounded prompt | 100 (11/11) | 100 (4/4) | 67 (2/3) | 100 (2/2) | 100 (5/5) |
| Full agent | 100 (11/11) | 100 (4/4) | 67 (2/3) | 100 (2/2) | 100 (5/5) |

### Findings (generated from the run)

- All comparisons below use the **25 questions that every system completed** (23 answerable, 2 unanswerable).
- Paired, agent vs hybrid+rerank, all questions (25): agent right & hybrid+rerank wrong on 1; the reverse on 1 (exact sign test p=1.00) - no statistically detectable difference.
- Paired, agent vs hybrid+rerank, answerable questions only (23): agent right & hybrid+rerank wrong on 1; the reverse on 1 (exact sign test p=1.00) - no statistically detectable difference.
- Paired, hybrid+rerank vs dense+grounded prompt (isolates RETRIEVAL quality), all questions (25): hybrid+rerank right & dense+grounded prompt wrong on 2; the reverse on 1 (exact sign test p=1.00) - no statistically detectable difference.
- Paired, hybrid+rerank vs dense+grounded prompt (isolates RETRIEVAL quality), answerable questions only (23): hybrid+rerank right & dense+grounded prompt wrong on 2; the reverse on 1 (exact sign test p=1.00) - no statistically detectable difference.
- Paired, dense+grounded prompt vs naive RAG (isolates the grounded/refusing PROMPT), all questions (25): dense+grounded prompt right & naive RAG wrong on 5; the reverse on 0 (exact sign test p=0.06) - no statistically detectable difference.
- Paired, dense+grounded prompt vs naive RAG (isolates the grounded/refusing PROMPT), answerable questions only (23): dense+grounded prompt right & naive RAG wrong on 3; the reverse on 0 (exact sign test p=0.25) - no statistically detectable difference.
- Paired, agent vs naive RAG, all questions (25): agent right & naive RAG wrong on 7; the reverse on 1 (exact sign test p=0.07) - no statistically detectable difference.
- Paired, agent vs naive RAG, answerable questions only (23): agent right & naive RAG wrong on 5; the reverse on 1 (exact sign test p=0.22) - no statistically detectable difference.
- **multi-hop / comparison** (4 questions): agent 4/4 vs hybrid+rerank 4/4 (within one question - noise).
- **numerical** (3 questions): agent 2/3 vs hybrid+rerank 2/3 (within one question - noise).
- **single-fact** (11 questions): agent 11/11 vs hybrid+rerank 11/11 (within one question - noise).
- **tricky-wording** (5 questions): agent 5/5 vs hybrid+rerank 5/5 (within one question - noise).
- **unanswerable (refusal)** (2 questions): agent 2/2 vs hybrid+rerank 2/2 (within one question - noise).
- Cost of the agent: 6.6 graph steps and 4494 tokens per query vs 1323 for hybrid+rerank (3.4x); p95 latency 35722 ms vs 6225 ms (distorted by caching/throttling - see scripts/latency_bench.py).
- Refusal on unanswerable questions: Naive RAG recall 0% / precision 0% (0 false refusals); Dense + grounded/refusing prompt recall 100% / precision 67% (1 false refusals); Hybrid + rerank + grounded prompt recall 100% / precision 100% (0 false refusals); Full agent recall 100% / precision 67% (1 false refusals).

**Judge vs. human:** 100% agreement, Cohen's kappa 1.0 on 14 hand-graded answers.
