*10 golden questions (78 answerable). Generator: `groq:openai/gpt-oss-120b`; judge: `groq:openai/gpt-oss-20b`; retrieval: `st:BAAI/bge-small-en-v1.5 + BM25 + cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2; store=chroma`; graph engine: `langgraph`.*

### Retrieval ablation (answerable questions, page-level gold)

| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|---|
| Dense only | 56.4 | 85.9 | 91.0 | 81.4 | 87.2 | 0.712 |
| BM25 only | 52.6 | 73.1 | 88.5 | 68.6 | 85.9 | 0.655 |
| Hybrid (RRF) | 60.3 | 82.1 | 92.3 | 79.5 | 89.7 | 0.739 |
| Hybrid + rerank | 75.6 | 89.7 | 98.7 | 86.5 | 96.2 | 0.838 |

### Three-system comparison

| Metric | Full agent |
|---|---|
| Correctness, all questions (%) | 40.0 |
| Correctness, answerable only (%) | 33.3 |
| Faithfulness (%) | 100.0 |
| Citation hit (gold page cited) (%) | 22.2 |
| Cited text contains answer (%) | 33.3 |
| Refusal precision (%) | 14.3 |
| Refusal recall (%) | 100.0 |
| False refusals (count) | 6 |
| Avg steps / question | 6.6 |
| Tool-call success (%) | 100.0 |
| Loop termination (%) | 100.0 |
| Latency p50 (ms) | 24094 |
| Latency p95 (ms) | 30224 |
| Tokens / query (est.) | 858 |

### Correctness by question type (%)

| System | single | multihop | numerical | unanswerable | ambiguous |
|---|---|---|---|---|---|
| Full agent | 50.0 | 50.0 | 0.0 | 100.0 | - |

### Findings (generated from the run)

- (findings need all three systems - run without --systems)

**Judge vs. human agreement:** not measured yet - run `python -m eval.hand_grade sample`, label 20 answers, then `python -m eval.hand_grade agree`.
