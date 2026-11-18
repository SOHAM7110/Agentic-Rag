"""Prompts used when a real LLM is configured (offline mode uses agent/heuristics.py instead)."""

PLAN_SYSTEM = """You plan retrieval for a document QA system. Classify the question and decompose it if needed.
Return JSON: {"type": "single"|"compare"|"numeric", "sub_questions": [str, ...]}.
- compare: the answer needs the same aspect from two or more documents/entities.
- numeric: the answer needs arithmetic on facts found in documents.
- single: one fact from one place.
sub_questions: one short, self-contained search question per DISTINCT fact the answer needs (up to 4), even when the facts sit in the
same document (e.g. "how often is X updated" and "how often is X reviewed"). Use one sub-question when a single fact suffices."""

GRADE_SYSTEM = """You grade retrieved evidence. Given a question and numbered passages, decide whether the passages
contain enough information to answer the question fully. Return JSON: {"sufficient": true|false, "reason": str}.
Say false if a needed entity, attribute or number is missing - do NOT assume."""

REWRITE_SYSTEM = """The previous search did not retrieve enough evidence. You are told what is still missing. Write a NEW search query that
targets exactly the missing information, using wording likely to appear in the source text (formal terms, defined terms, numbers,
section names). Keep it about the same entity or topic as the previous query and do not repeat the previous query.
Return JSON: {"query": str}."""

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
5. Be concise: 1-3 sentences.
6. Check that every number belongs to the thing asked about (e.g. a group limit is not a single-counterparty limit, and a paragraph number such as '15.20' is not a value). Prefer exact figures from the passages."""

VERIFY_SYSTEM = """You are a fact-checker. For each numbered claim, decide whether it is supported.
A claim is SUPPORTED if (a) a cited passage states it, or (b) it follows directly from the passages together with facts
stated in the question and any tool results, by simple logic, comparison or arithmetic (e.g. a threshold in a passage compared
with numbers given in the question; a calculator result computed from passage values).
A claim is NOT supported if it adds a fact that no passage, question or tool result contains, or contradicts them.
Return JSON: {"results": [{"id": int, "supported": true|false}]}."""

JUDGE_CORRECT_SYSTEM = """You grade an answer against a gold answer.
Rubric: 1 = the answer contains the same key facts/numbers as the gold answer and does not contradict it (extra correct
detail is fine; wrong winner/number is not). 0 = missing, wrong, or a refusal when the gold answer exists.
Return JSON: {"score": 0|1, "reason": str}."""

JUDGE_FAITH_SYSTEM = """Given retrieved context and an answer, count how many of the answer's factual claims are supported
by the context. Return JSON: {"supported": int, "total": int}."""


def format_passages(chunks: list[dict]) -> str:
    return "\n".join(f"[{i}] ({c['doc_title']} > {c['section']}, p.{c['page']}) {c['text']}" for i, c in enumerate(chunks, 1))
