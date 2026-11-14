"""Rule-based controller logic. Used (a) as the whole 'brain' in offline mode (no LLM, zero cost), and
(b) as a safety net when a real LLM returns malformed JSON. Everything here is deterministic."""
from __future__ import annotations
import re
from config import REFERENCE_YEAR
from ingest.facts import FACT_SPEC
from retrieval.text import tokenize, split_sentences, norm_numbers

GENERIC = {"yojana", "scheme", "abhiyan", "mission", "programme", "program"}
QWORDS = {"scheme", "programme", "program", "person", "people", "amount", "number", "get", "give", "provide", "under", "apply", "applicant"}


# ------------------------------------------------------------------ entities / aspect
def link_entities(question: str, docs: dict[str, dict]) -> list[str]:
    """Match scheme titles (exact, or all distinctive title tokens) in the question; order = order of mention."""
    q = norm_numbers(question.lower())
    qtok = set(re.findall(r"[a-z0-9]+", q))
    found = []
    for did, d in docs.items():
        title = d["title"].lower()
        pos = q.find(title)
        if pos < 0:
            dist = [t for t in re.findall(r"[a-z0-9]+", title) if t not in GENERIC]
            if len(dist) >= 2 and all(t in qtok for t in dist):
                pos = min(q.find(t) for t in dist)
        if pos >= 0:
            found.append((pos, did))
    return [d for _, d in sorted(found)]


def detect_aspect(question: str, docs: dict | None = None) -> str | None:
    """Which attribute is being asked about? Entity names are removed first (the 'Pension' in a scheme's name
    must not trigger the 'pension' cue); the longest matched word wins."""
    q = question.lower()
    if docs:
        for d in docs.values():
            q = q.replace(d["title"].lower(), " ")
    best, blen = None, 0
    for key, spec in FACT_SPEC.items():
        for cue in spec[2]:
            m = re.search(r"\b" + re.escape(cue) + r"\w*", q)
            if m and len(m.group()) > blen:
                best, blen = key, len(m.group())
    return best


# ------------------------------------------------------------------ planning
CMP = re.compile(r"\b(compare|versus|vs\.?|which (?:has|is|one|scheme)|higher|lower|larger|smaller|earlier|later|same ministry|differ)\b", re.I)
NUM = re.compile(r"\b(how much|combined|total amount|per beneficiary|each instalment|each installment|per instalment|how many years|exceed|by how much|difference between|difference in|over \d+ years|for \d+ years|calculate)\b", re.I)


def classify(question: str, entities: list[str]) -> str:
    if NUM.search(question) and not re.search(r"^how much (?:financial|money|support|funding|budget)", question.lower()):
        return "numeric"
    if len(entities) >= 2 or CMP.search(question):
        return "compare" if len(entities) >= 2 else "single"
    return "single"


def numeric_plan(question: str, entities: list[str], aspect: str | None = None) -> dict | None:
    q = question.lower()
    e = entities
    if re.search(r"(each|per|every) (instalment|installment)|amount of each instalment", q) and e:
        return {"op": "div", "docs": [e[0]], "keys": ["benefit_amount", "installments"], "tmpl": "{0}/{1}", "unit": "Rs. ", "what": "each instalment"}
    m = re.search(r"(?:for|over|in) (\d+) years", q)
    if m and e and "benefit" in q:
        return {"op": "mul", "docs": [e[0]], "keys": ["benefit_amount"], "tmpl": "{0}*" + m.group(1), "unit": "Rs. ", "what": f"total benefit over {m.group(1)} years"}
    if re.search(r"combined|together|sum", q) and len(e) >= 2 and "outlay" in q:
        return {"op": "sum", "docs": e[:2], "keys": ["outlay_crore"], "tmpl": "{0}+{1}", "unit": "Rs. ", "suffix": " crore", "what": "combined outlay"}
    if "per beneficiary" in q and e:
        return {"op": "per", "docs": [e[0]], "keys": ["outlay_crore", "beneficiaries_lakh"], "tmpl": "{0}*100/{1}", "unit": "Rs. ", "what": "outlay per beneficiary"}
    if "difference between the maximum and minimum" in q and e:
        return {"op": "sub", "docs": [e[0]], "keys": ["max_age", "min_age"], "tmpl": "{0}-{1}", "suffix": " years", "what": "age range"}
    if re.search(r"how many years.*(since|launch)|years have passed", q) and e:
        y = re.search(r"as of (\d{4})", q)
        return {"op": "age", "docs": [e[0]], "keys": ["launch_year"], "tmpl": f"{y.group(1) if y else REFERENCE_YEAR}-{{0}}", "suffix": " years", "what": "years since launch"}
    if re.search(r"exceed|how much more|by how much", q) and len(e) >= 2:
        key = aspect or "benefit_amount"
        return {"op": "diff", "docs": e[:2], "keys": [key], "tmpl": "{0}-{1}", "unit": "Rs. " if key == "benefit_amount" else "", "what": "difference"}
    return None


def make_plan(question: str, docs: dict[str, dict]) -> dict:
    ents = link_entities(question, docs)
    qtype = classify(question, ents)
    aspect = detect_aspect(question, docs)
    plan = {"type": qtype, "entities": ents, "aspect": aspect, "numeric": None, "descriptor": None, "tools": []}
    if qtype == "numeric":
        plan["numeric"] = numeric_plan(question, ents, aspect)
        if plan["numeric"] is None:
            plan["type"] = "single"
        else:
            plan["tools"] = ["lookup", "calculator"]
    if plan["type"] == "compare":
        plan["tools"] = ["compare"]
    if plan["type"] == "single" and ents and aspect:
        plan["tools"] = ["lookup"]
    if not ents:
        m = re.search(r"(?:scheme|programme|program) that (.+?)\??$", question.strip(), re.I)
        if m:
            plan["descriptor"] = m.group(1)
            plan["tools"] = ["lookup"]
    return plan


def entity_tokens(entities: list[str], docs: dict) -> set[str]:
    out = set()
    for e in entities:
        out |= set(tokenize(docs[e]["title"]))
    return out


def query_tokens(q: str, ignore: set[str], drop_numbers: bool = False) -> list[str]:
    """Content tokens of a query. drop_numbers=True ignores figures the *user* supplied (e.g. 'Rs 200 crore' in a
    calculation question) - they never appear in the documents and must not lower the evidence-coverage score."""
    return [t for t in tokenize(q) if t not in ignore and t not in QWORDS and t not in GENERIC and not (drop_numbers and t.isdigit())]


# ------------------------------------------------------------------ query rewriting
def _strip_cue(question: str, aspect: str | None) -> str:
    """Remove the paraphrase that expressed the aspect (it gets replaced by the canonical label)."""
    q = question
    if aspect:
        for cue in sorted(FACT_SPEC[aspect][2], key=len, reverse=True):
            q2 = re.sub(r"\b" + re.escape(cue) + r"\w*", " ", q, flags=re.I)
            if q2 != q:
                return q2
    return q


def rewrite_query(question: str, plan: dict, docs: dict, attempt: int) -> dict:
    """attempt 1: canonical aspect wording (paraphrase replaced, other words - e.g. an unknown scheme name - kept);
    2: add cue words; 3: drop the document filter (global search)."""
    ent_titles = " ".join(docs[e]["title"] for e in plan["entities"])
    asp = plan.get("aspect")
    label = FACT_SPEC[asp][1] if asp else ""
    rest = " ".join(t for t in tokenize(_strip_cue(question, asp)) if t not in GENERIC and t not in QWORDS)
    if attempt == 1:
        q = f"{ent_titles} {rest} {label}"
    elif attempt == 2:
        cues = " ".join(FACT_SPEC[asp][2][:3]) if asp else ""
        q = f"{rest} {label} {cues}"
    else:
        q = f"{label} {rest}"
    return {"q": " ".join(q.split()), "drop_filter": attempt >= 3}


# ------------------------------------------------------------------ extractive answering / support checks
def best_sentences(question: str, chunks: list[dict], retriever, ignore: set[str], n: int = 1) -> list[tuple[float, str, dict]]:
    qt = set(query_tokens(question, ignore))
    scored = []
    for rank, c in enumerate(chunks):
        units = [r.replace(" | ", ": ") for r in c["text"].split("\n")] if c["kind"] == "table" else split_sentences(c["text"])
        for u in units:
            ut = set(tokenize(u))
            if not qt: continue
            ov = sum(retriever.bm25.idf(t) for t in qt if t in ut) / (sum(retriever.bm25.idf(t) for t in qt) or 1)
            scored.append((ov - 0.01 * rank, u, c))
    scored.sort(key=lambda x: -x[0])
    return scored[:n]


def claim_support(claim: str, passages: list[str], inputs: list[str] | None = None) -> float:
    """Fraction of the claim's content tokens present in the cited passages; numbers must all be present."""
    ptoks = set()
    for p in passages: ptoks |= set(tokenize(p))
    if inputs is not None:                      # computed claim: its *inputs* must be in the passages
        nums = [norm_numbers(str(i)) for i in inputs]
        text = norm_numbers(" ".join(passages))
        return 1.0 if all(re.search(r"(?<!\d)" + re.escape(n) + r"(?!\d)", text) for n in nums) else 0.0
    ctoks = [t for t in tokenize(claim) if t not in GENERIC]
    if not ctoks: return 1.0
    nums = [t for t in ctoks if t.isdigit()]
    if any(n not in ptoks for n in nums): return 0.0
    return sum(t in ptoks for t in ctoks) / len(ctoks)


# [1] [1,2] [1][2] but also fullwidth ［1］, lenticular 【1】, "[Passage 1]", "[Passages 1, 2]", "[1†L5-L9]" style tails
CITE_RE = re.compile(r"\s*[\[\uff3b\u3010]\s*(?:passages?\s*)?(\d+(?:\s*[,;\u2013-]\s*(?:passages?\s*)?\d+)*)[^\]\uff3d\u3011]{0,20}[\]\uff3d\u3011]", re.I)


def parse_cites(text: str) -> list[int]:
    nums = []
    for m in CITE_RE.finditer(text):
        nums += [int(n) for n in re.findall(r"\d+", m.group(1))]
    return sorted(set(nums))


def strip_cites(text: str) -> str:
    return CITE_RE.sub("", text).strip()


def content_overlap(claim: str, passage: str) -> float:
    """Share of the claim's non-numeric content words found in the passage (numbers may be *computed*, so ignored)."""
    ct = {t for t in tokenize(claim) if not t.isdigit() and t not in GENERIC}
    if not ct: return 0.0
    pt = set(tokenize(passage))
    return sum(t in pt for t in ct) / len(ct)
