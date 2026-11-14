"""python -m eval.verify_golden [golden.jsonl] [pdf_dir]
Checks that every gold source really contains its evidence: for each gold_sources entry, every string in
`must_contain` (falls back to answer_keys) must appear in the text of the listed pages of that PDF.
Comparison ignores case, whitespace and thousands separators, so PDF-extractor spacing quirks don't matter.
Run this after editing the golden set; fix or delete any question it flags."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from config import GOLDEN_PATH, PDF_DIR
from ingest.parse import parse_pdf


def norm(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u00ad", "")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return re.sub(r"[\s|]+", "", s)   # ignore whitespace and the ' | ' the parser puts between table cells


def main():
    gp = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN_PATH
    pdfs = Path(sys.argv[2]) if len(sys.argv) > 2 else PDF_DIR
    rows = [json.loads(l) for l in gp.read_text(encoding="utf-8").splitlines() if l.strip()]
    cache, bad = {}, 0
    for q in rows:
        for src in q["gold_sources"]:
            f = pdfs / f"{src['doc']}.pdf"
            if not f.exists():
                print(f"[{q['id']}] MISSING PDF {src['doc']}"); bad += 1; continue
            if f not in cache: cache[f] = parse_pdf(f)
            doc = cache[f]
            text = norm(" ".join(doc.page_text(p) for p in src["pages"]))
            for phrase in src.get("must_contain", q["answer_keys"]):
                if norm(phrase) not in text:
                    print(f"[{q['id']}] '{phrase}' NOT on pages {src['pages']} of {src['doc']}"); bad += 1
    types = {}
    for q in rows: types[q["type"]] = types.get(q["type"], 0) + 1
    print(f"\n{len(rows)} questions {types} | {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
