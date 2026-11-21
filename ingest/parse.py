"""PDF parsing -> pages of typed lines (text, font size, bold).

Primary backend: PyMuPDF (fitz). Fallback: pdfplumber (used automatically if PyMuPDF is missing).
Horizontal gaps inside a line (table cells) are rendered as ' | ' so that table rows stay parseable.
"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    page: int


@dataclass
class ParsedDoc:
    doc_id: str
    path: str
    title: str
    pages: dict[int, list[Line]] = field(default_factory=dict)
    body_size: float = 10.0

    def page_text(self, p: int) -> str:
        return "\n".join(l.text for l in self.pages.get(p, []))


GAP = 14.0  # points of horizontal whitespace that mark a new table cell
# "(4)", "4.", "(a)", "(iv)", "12)", "4.1" - numbering at the start of a hanging-indent paragraph, NOT a table cell
MARKER = re.compile(r"^(\(?[0-9A-Za-z]{1,5}[.)]|\d+(\.\d+)+\.?)$")


def merge_marker_lines(items: list) -> list:
    """items: [(y, x0, Line)]. A marker-only line such as '(4)' and the text line at the same height to its right are ONE paragraph line
    ('(4) The tenure ...'). PyMuPDF returns them as separate lines when the hanging indent is wide; pdfplumber as one line with a gap."""
    used, out = set(), []
    for i, (y, x, l) in enumerate(items):
        if i in used:
            continue
        if MARKER.match(l.text.strip()):
            best = None
            for j, (y2, x2, l2) in enumerate(items):
                if j != i and j not in used and abs(y2 - y) < 4 and x2 > x and not MARKER.match(l2.text.strip()):
                    if best is None or x2 < items[best][1]:
                        best = j
            if best is not None:
                used.add(best)
                l2 = items[best][2]
                out.append((y, x, Line(l.text.strip() + " " + l2.text, l2.size, l2.bold or l.bold, l.page)))
                continue
        out.append((y, x, l))
    return out


def _gap_sep(so_far: str) -> str:
    """Separator to use for a big horizontal gap: a space after a paragraph marker, ' | ' (table cell) otherwise."""
    return " " if MARKER.match(so_far.strip()) else " | "


def _lines_fitz(path: str) -> dict[int, list[Line]]:
    import fitz  # PyMuPDF
    out: dict[int, list[Line]] = {}
    with fitz.open(path) as pdf:
        for pno, page in enumerate(pdf, start=1):
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for ln in block.get("lines", []):
                    text, last_x1, sizes, bold = "", None, [], False
                    for sp in ln["spans"]:
                        t = sp["text"]
                        if not t.strip():
                            continue
                        x0, x1 = sp["bbox"][0], sp["bbox"][2]
                        if last_x1 is not None and x0 - last_x1 > GAP:
                            text = text.rstrip() + _gap_sep(text) + t.strip()
                        else:
                            text += t
                        last_x1 = x1
                        sizes.append(sp["size"])
                        bold = bold or bool(sp["flags"] & 16) or "bold" in sp["font"].lower()
                    text = text.strip()
                    if text:
                        lines.append((ln["bbox"][1], ln["bbox"][0], Line(text, max(sizes), bold, pno)))
            lines = merge_marker_lines(sorted(lines, key=lambda x: (x[0], x[1])))
            out[pno] = [l for _, _, l in sorted(lines, key=lambda x: (x[0], x[1]))]
    return out


def _lines_plumber(path: str) -> dict[int, list[Line]]:
    import pdfplumber
    out: dict[int, list[Line]] = {}
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False, use_text_flow=False)
            rows: dict[int, list[dict]] = {}
            for w in words:
                rows.setdefault(round(w["top"] / 3), []).append(w)
            lines = []
            for key in sorted(rows):
                ws = sorted(rows[key], key=lambda w: w["x0"])
                parts, last = [], None
                for w in ws:
                    if last is not None:
                        parts.append(_gap_sep("".join(parts)) if w["x0"] - last > GAP else " ")
                    parts.append(w["text"])
                    last = w["x1"]
                lines.append(Line("".join(parts).strip(), max(w["size"] for w in ws),
                                  any("bold" in w["fontname"].lower() for w in ws), pno))
            out[pno] = lines
    return out


def parse_pdf(path: str | Path) -> ParsedDoc:
    path = str(path)
    try:
        pages = _lines_fitz(path)
    except ImportError:
        pages = _lines_plumber(path)
    sizes = Counter()
    for ls in pages.values():
        for l in ls:
            sizes[round(l.size, 1)] += len(l.text)
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    # title: biggest text on page 1 - but only if it is readable (bilingual PDFs lead with a Hindi heading);
    # otherwise use the file name (scripts/prepare_corpus.py renames files from their real titles).
    title = Path(path).stem.replace("_", " ").title()
    p1 = pages.get(1, [])
    if p1:
        top = max(p1, key=lambda l: l.size)
        ascii_share = sum(ch.isascii() for ch in top.text) / max(len(top.text), 1)
        if top.size > body + 3 and ascii_share > 0.9 and len(top.text) >= 15:
            title = top.text
    return ParsedDoc(doc_id=Path(path).stem, path=path, title=title, pages=pages, body_size=body)


def is_heading(line: Line, body_size: float) -> bool:
    t = line.text.strip()
    # wrapped continuation lines ("(viii) of the Act ...", lowercase starts) are not headings
    if len(t) > 90 or " | " in t or t[:1] in "(,;:" or t[:1].islower() or len(t) < 3 or not any(c.isalpha() for c in t):
        return False
    return line.size >= body_size + 1.5 or (line.bold and line.size >= body_size and not line.text.endswith((".", ",")))
