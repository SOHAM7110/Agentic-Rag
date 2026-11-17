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
