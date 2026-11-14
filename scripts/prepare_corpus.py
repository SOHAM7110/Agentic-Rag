"""python scripts/prepare_corpus.py <folder_with_downloaded_pdfs> [--out data/pdfs]
 1. drops byte-identical duplicates ("file (1).pdf")
 2. reads the real title from page 1-2 ("Reserve Bank of India (X) Directions, 2025") and renames the file to it,
    so document ids/citations are readable instead of '161MD'.
Uses pdfplumber (already a dependency)."""
import hashlib, re, shutil, sys
from pathlib import Path
import pdfplumber

TITLE = re.compile(r"((?:Reserve Bank of India|Securities and Exchange Board of India|Master Direction)[^.]{0,200}?(?:Directions?|Circular|Regulations?)(?:,)?\s*(?:20\d\d)?)", re.I)


def find_title(path: Path) -> str | None:
    with pdfplumber.open(path) as pdf:
        text = " ".join((pdf.pages[i].extract_text() or "") for i in range(min(2, len(pdf.pages))))
    text = re.sub(r"\s+", " ", text)
    for m in TITLE.finditer(text):
        t = m.group(1).strip(" ,")
        if "(" in t and ")" in t and len(t) > 25:
            return t
    return None


def slug(s: str) -> str:
    s = re.sub(r"reserve bank of india", "rbi", s, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:90]


def main():
    src = Path(sys.argv[1]); out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("data/pdfs")
    out.mkdir(parents=True, exist_ok=True)
    seen, used = {}, set()
    for p in sorted(src.glob("*.pdf")):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            print(f"skip duplicate: {p.name} (same as {seen[h]})"); continue
        title = find_title(p)
        name = slug(title) if title else slug(p.stem)
        base, i = name, 2
        while name in used: name = f"{base}_{i}"; i += 1
        used.add(name); seen[h] = p.name
        shutil.copy(p, out / f"{name}.pdf")
        print(f"{p.name[:40]:40} -> {name}.pdf" + ("" if title else "   (title not found - rename by hand)"))


if __name__ == "__main__":
    main()
