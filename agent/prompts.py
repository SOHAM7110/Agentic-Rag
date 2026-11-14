"""Prompts used when a real LLM is configured (offline mode uses agent/heuristics.py instead)."""

PLAN_SYSTEM = """You plan retrieval for a document QA system. Classify the question and decompose it if needed.
Return JSON: {"type": "single"|"compare"|"numeric", "sub_questions": [str, ...]}.
- compare: the answer needs the same aspect from two or more documents.
- numeric: the answer needs arithmetic on facts found in documents.
- single: one fact from one document. Keep sub_questions short and self-contained."""

GRADE_SYSTEM = """You grade retrieved evidence. Given a question and numbered passages, decide whether the passages
contain enough information to answer the question fully. Return JSON: {"sufficient": true|false, "reason": str}.
Say false if a needed entity, attribute or number is missing - do NOT assume."""

REWRITE_SYSTEM = """The previous search did not retrieve enough evidence. Rewrite the search query to be more likely to match
the wording used in official scheme documents (use canonical terms: 'minimum age', 'programme outlay', 'benefit amount',
'nodal ministry', etc.). Return JSON: {"query": str}."""

CALC_SYSTEM = """Write ONE arithmetic expression that answers the question, using only numbers stated in the question or in the passages
(for example a rate from a passage times an amount from the question; express percentages as fractions or as x/100).
Use plain numbers (no commas, no units). Return JSON: {"expression": "0.05*500"}.
If the question needs no calculation or the passages lack a needed number, return {"expression": ""}."""

ANSWER_SYSTEM = """You answer questions strictly from the numbered context passages.
Rules:
1. Every sentence that states a fact MUST end with its supporting passage number(s) in plain ASCII square brackets, exactly like [2] or [1][3]. Never use other bracket styles (no 【2】, no (Passage 2), no footnotes).
2. Use only information in the passages or the tool results. Never use outside knowledge.
3. If the passages do not contain the answer, reply with exactly: INSUFFICIENT_EVIDENCE
4. If the evidence is partial or ambiguous, say so explicitly ("The documents only state ...").
5. Be concise: 1-3 sentences."""

VERIFY_SYSTEM = """You are a strict fact-checker. For each numbered claim, decide whether it is fully supported by the
passages cited for it. Return JSON: {"results": [{"id": int, "supported": true|false}]}."""

JUDGE_CORRECT_SYSTEM = """You grade an answer against a gold answer.
Rubric: 1 = the answer contains the same key facts/numbers as the gold answer and does not contradict it (extra correct
detail is fine; wrong winner/number is not). 0 = missing, wrong, or a refusal when the gold answer exists.
Return JSON: {"score": 0|1, "reason": str}."""

JUDGE_FAITH_SYSTEM = """Given retrieved context and an answer, count how many of the answer's factual claims are supported
by the context. Return JSON: {"supported": int, "total": int}."""


def format_passages(chunks: list[dict]) -> str:
    return "\n".join(f"[{i}] ({c['doc_title']} > {c['section']}, p.{c['page']}) {c['text']}" for i, c in enumerate(chunks, 1))
