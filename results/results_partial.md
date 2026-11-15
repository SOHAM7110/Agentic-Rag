*30 golden questions (78 answerable). Generator: `groq:openai/gpt-oss-120b`; judge: `groq:openai/gpt-oss-20b`; retrieval: `st:BAAI/bge-small-en-v1.5 + BM25 + cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2; store=chroma`; graph engine: `langgraph`.*

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
| Correctness, all questions (%) | 70.0 | 83.3 | 86.7 |
| Correctness, answerable only (%) | 80.8 | 80.8 | 84.6 |
| Faithfulness (%) | 77.2 | 93.5 | 97.7 |
| Citation hit (gold page cited) (%) | 80.8 | 53.8 | 76.9 |
| Cited text contains answer (%) | 73.1 | 46.2 | 65.4 |
| Refusal precision (%) | 0.0 | 57.1 | 50.0 |
| Refusal recall (%) | 0.0 | 100.0 | 100.0 |
| False refusals (count) | 0 | 3 | 4 |
| Avg steps / question | 2.0 | 2.0 | 7.2 |
| Tool-call success (%) | - | - | 100.0 |
| Loop termination (%) | 100.0 | 100.0 | 100.0 |
| Degraded answers (LLM fell back to rules) | 0 | 0 | 0 |
| Latency p50 (ms) | 3 | 632 | 1028 |
| Latency p95 (ms) | 18 | 1115 | 29287 |
| Tokens / query (est.) | 911 | 1456 | 998 |

### Correctness by question type (%)

| System | single | multihop | numerical | unanswerable | ambiguous |
|---|---|---|---|---|---|
| Naive RAG (dense top-3) | 100.0 | 60.0 | 60.0 | 0.0 | 66.7 |
| Hybrid + rerank (single shot) | 100.0 | 40.0 | 80.0 | 100.0 | 66.7 |
| Full agent | 92.3 | 60.0 | 80.0 | 100.0 | 100.0 |

### Findings (generated from the run)

- On **multi-hop / comparison** questions the agent beats single-shot hybrid+rerank by 20 points (60% vs 40%).
- On **numerical** questions the agent is on par with single-shot hybrid+rerank (80% vs 80%): no measurable gain.
- On **single-fact** questions the agent is *worse* than single-shot hybrid+rerank by 8 points (92% vs 100%).
- On **ambiguous-wording** questions the agent beats single-shot hybrid+rerank by 33 points (100% vs 67%).
- On **unanswerable (refusal)** questions the agent is on par with single-shot hybrid+rerank (100% vs 100%): no measurable gain.
- Cost of the agent: 7.2 graph steps and 998 (estimated) tokens per query vs 1456 for hybrid+rerank (0.7x); p95 latency 29287 ms vs 1115 ms.
- Naive RAG never refuses (refusal recall 0%); the agent reaches 100% recall at 50% precision, with 4 false refusals on answerable questions.

**Judge vs. human agreement:** not measured yet - run `python -m eval.hand_grade sample`, label 20 answers, then `python -m eval.hand_grade agree`.
