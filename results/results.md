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
