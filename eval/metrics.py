"""Pure-python metric functions (retrieval, citations, refusal, ops)."""
from __future__ import annotations
import re, statistics as st


def chunk_pages(c: dict) -> set[int]:
    return set(range(c["page"], c.get("page_end", c["page"]) + 1))


def relevant(chunk: dict, gold_sources: list[dict]) -> int | None:
    """Index of the gold source this chunk satisfies (same doc, overlapping page), else None."""
    for gi, g in enumerate(gold_sources):
        if chunk["doc_id"] == g["doc"] and chunk_pages(chunk) & set(g["pages"]):
            return gi
    return None


def retrieval_scores(hits: list[dict], gold_sources: list[dict], ks=(1, 3, 5)) -> dict:
    """hits: ranked chunk dicts. Hit@k = any gold source in top-k; Recall@k = fraction of gold sources covered;
    MRR = 1/rank of first relevant chunk."""
    out = {}
    for k in ks:
        covered = {relevant(c, gold_sources) for c in hits[:k]} - {None}
        out[f"hit@{k}"] = 1.0 if covered else 0.0
        out[f"recall@{k}"] = len(covered) / len(gold_sources)
    rr = 0.0
    for rank, c in enumerate(hits, 1):
        if relevant(c, gold_sources) is not None:
            rr = 1.0 / rank; break
    out["mrr"] = rr
    return out


def norm(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", text.lower())


def has_key(text: str, key: str) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(norm(key)) + r"(?![\w])", norm(text)) is not None


def citation_scores(citations: list[dict], gold_sources: list[dict], answer_keys: list[str]) -> dict:
    """citation_hit: every gold doc has a cited chunk on an acceptable page.
    cited_contains_answer: the cited text itself contains all answer keys (checks the page 'actually contains the answer')."""
    covered = {relevant(c_as_chunk(c), gold_sources) for c in citations} - {None}
    cited_text = " ".join(c.get("full_text", c.get("text", "")) for c in citations)
    return {"citation_hit": 1.0 if gold_sources and len(covered) == len(gold_sources) else 0.0,
            "citation_precision": (sum(relevant(c_as_chunk(c), gold_sources) is not None for c in citations) / len(citations)) if citations else 0.0,
            "cited_contains_answer": 1.0 if answer_keys and all(has_key(cited_text, k) for k in answer_keys) else 0.0}


def c_as_chunk(c: dict) -> dict:
    return {"doc_id": c["doc"], "page": c["page"], "page_end": c.get("page_end", c["page"])}


def refusal_scores(rows: list[dict]) -> dict:
    tp = sum(1 for r in rows if r["type"] == "unanswerable" and r["refused"])
    pred = sum(1 for r in rows if r["refused"])
    pos = sum(1 for r in rows if r["type"] == "unanswerable")
    p = tp / pred if pred else 0.0; rc = tp / pos if pos else 0.0
    return {"refusal_precision": p, "refusal_recall": rc, "refusal_f1": (2 * p * rc / (p + rc)) if p + rc else 0.0,
            "false_refusals": sum(1 for r in rows if r["type"] != "unanswerable" and r["refused"])}


def pct(xs, q):
    xs = sorted(xs)
    if not xs: return 0.0
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def mean(xs): return sum(xs) / len(xs) if xs else 0.0
