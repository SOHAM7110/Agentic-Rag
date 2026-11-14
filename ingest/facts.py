"""Structured facts table extracted from 'key | value' rows (the 'at a glance' tables) of each document.

The table is the backing store for the structured-lookup tool. LABELS maps row labels (and synonyms
used in questions) to canonical keys; extend it for your own corpus, or plug an LLM extractor in
`extract_facts` (same output schema)."""
from __future__ import annotations
import re
from retrieval.text import norm_numbers

# canonical key -> (row label regex, human label, question cue phrases, evidence cue tokens, kind)
FACT_SPEC = {
    "launch_year":        (r"launch year", "launch year", ["launch", "launch year", "come into force", "came into force", "begin", "began", "start", "started", "since"], r"launched in|came into force|launch year", "int"),
    "ministry":           (r"nodal ministry", "nodal ministry", ["ministry", "administers", "administered", "runs", "run by"], r"ministry", "text"),
    "benefit_amount":     (r"benefit amount", "benefit amount", ["benefit amount", "benefit", "financial support", "assistance", "grant", "subsidy", "pension", "cover", "stipend", "scholarship", "hand out", "provide"], r"\brs\.? ?\d", "money"),
    "installments":       (r"number of instalments", "number of instalments", ["instalment", "installment", "tranche"], r"instalment", "int"),
    "min_age":            (r"minimum age", "minimum age", ["minimum age", "minimum eligible age", "lowest age", "youngest", "how young", "from what age", "at least"], r"at least \d+ years|minimum age", "int"),
    "max_age":            (r"maximum age", "maximum age", ["maximum age", "maximum eligible age", "upper age", "age limit", "beyond what age", "above what age", "after which birthday", "oldest"], r"upper age limit|above \d+ years|maximum age", "int"),
    "income_ceiling":     (r"household income ceiling", "household income ceiling", ["income ceiling", "income limit", "income cap", "household income"], r"income of|income above|income ceiling|household income", "money"),
    "outlay_crore":       (r"programme outlay", "programme outlay", ["outlay", "budget", "earmarked", "funding", "allocation", "allocated"], r"outlay|budget", "int"),
    "beneficiaries_lakh": (r"target beneficiaries", "target beneficiaries", ["beneficiaries", "beneficiary numbers", "target beneficiar", "how many lakh", "people", "reach", "covered", "cover"], r"lakh", "int"),
    "helpline":           (r"helpline", "helpline", ["helpline", "toll-free", "toll free", "phone number", "call"], r"helpline", "text"),
    "grievance_days":     (r"grievance resolution time", "grievance resolution time", ["grievance", "complaint", "complaints", "time limit for closing"], r"within \d+ days|closed within|resolution time", "int"),
    "app_fee":            (r"application fee", "application fee", ["application fee", "fee", "charged"], r"\bfee\b|free of cost", "money"),
    "portal":             (r"portal", "portal", ["portal", "website", "apply online", "apply for", "registration"], r"portal", "text"),
}


def parse_value(raw: str, kind: str):
    t = norm_numbers(raw.lower())
    if kind in ("int", "money"):
        if t.strip() in ("nil", "none", "free"):
            return 0
        m = re.search(r"\d+", t)
        return int(m.group()) if m else None
    return raw.strip()


def extract_facts(chunks: list[dict]) -> dict[str, dict]:
    """-> {doc_id: {key: {value, raw, label, page, chunk_id, doc_title}}}"""
    facts: dict[str, dict] = {}
    for c in chunks:
        if c["kind"] != "table":
            continue
        for row in c["text"].split("\n"):
            if " | " not in row:
                continue
            label, raw = [x.strip() for x in row.split(" | ", 1)]
            for key, (lab_re, human, *_rest, kind) in FACT_SPEC.items():
                if re.fullmatch(lab_re, label.lower()):
                    facts.setdefault(c["doc_id"], {})[key] = {
                        "value": parse_value(raw, kind), "raw": raw, "label": human, "page": c["page"],
                        "chunk_id": c["id"], "doc_title": c["doc_title"], "section": c["section"],
                    }
    return facts
