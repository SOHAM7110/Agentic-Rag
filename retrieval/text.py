"""Shared text normalisation / tokenisation used by BM25, the lexical reranker and the agent."""
import re

STOPWORDS = set("""a an the of to in on for and or is are was were be been by with as at from that this these those it its
what which who whom how much many does do did under into than then there their has have had can could should would will may
might per each any all if not no one whose where when while about out up over both either also""".split())


def norm_numbers(text: str) -> str:
    """'Rs. 2,50,000' -> 'Rs. 250000' so the same number always tokenises identically."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return text


def stem(tok: str) -> str:
    if tok.isdigit():
        return tok
    tok = tok.replace("installment", "instalment")
    if len(tok) > 5 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    if len(tok) > 6 and tok.endswith("ing"):
        tok = tok[:-3]
    elif len(tok) > 5 and tok.endswith("ed"):
        tok = tok[:-2]
    return tok


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    text = norm_numbers(text.lower())
    toks = re.findall(r"[a-z0-9]+", text)
    out = []
    for t in toks:
        if t == "rs":
            continue
        if not keep_stop and t in STOPWORDS:
            continue
        out.append(stem(t))
    return out


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that does not break on 'Rs. 6,000', 'No.', 'approx.' etc."""
    protected = re.sub(r"\b(Rs|No|Sec|approx|e\.g|i\.e|etc|Dr|Sl)\.\s", lambda m: m.group(1) + ".\x00", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", protected)
    return [p.replace("\x00", " ").strip() for p in parts if p.strip()]


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
