"""Hermetic test fixtures: forces the dependency-free backends, and builds small PDFs/indexes in temp folders so tests never
depend on (or touch) your real corpus, index, API keys or network."""
import os
for _k, _v in {"EMBEDDER": "lsa", "RERANKER": "lexical", "VECTOR_STORE": "numpy", "LLM_PROVIDER": "offline", "JUDGE_PROVIDER": "offline"}.items():
    os.environ[_k] = _v                                   # must happen before `config` is imported

import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_cache: dict = {}


def _retriever(pdf_dir: Path, key: str):
    if key not in _cache:
        from ingest.build_index import build
        from retrieval.retriever import Retriever
        idx = Path(tempfile.mkdtemp(prefix=f"idx_{key}_"))
        build(pdf_dir, idx, verbose=False)
        _cache[key] = Retriever(idx)
    return _cache[key]


def demo_retriever():
    """The synthetic scheme corpus (offline demo)."""
    d = ROOT / "data" / "demo_pdfs"
    if not list(d.glob("*.pdf")):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_corpus.py")], check=True)
    return _retriever(d, "demo")


def make_regulatory_pdfs(out: Path):
    """Two near-duplicate 'regulation' PDFs with the layout features that broke the parser: hanging-indent numbered paragraphs
    ('(4)' drawn as a separate cell), and a table with a lead-in sentence and wrapped cells."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle
    from reportlab.lib import colors
    h1 = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=6)
    h2 = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=10.5, leading=14, spaceAfter=5)
    hang = ParagraphStyle("hg", parent=body, leftIndent=40, bulletIndent=0)
    cell = ParagraphStyle("c", fontName="Helvetica", fontSize=10, leading=12)
    out.mkdir(parents=True, exist_ok=True)
    for name, single, group in [("Alpha", 20, 25), ("Beta", 15, 30)]:
        doc = SimpleDocTemplate(str(out / f"rbi_{name.lower()}_lending_directions_2025.pdf"), pagesize=A4)
        story = [Paragraph(f"Rbi {name} Lending Directions 2025", h1), Paragraph("Exposure Limits", h2)]
        for i, t in enumerate([
            f"The exposure of a bank to a single counterparty shall not exceed {single} percent of the eligible capital base of the bank, "
            f"measured at the end of each quarter and reported to the Reserve Bank without delay.",
            f"The exposure of a bank to a group of connected counterparties shall not exceed {group} percent of the eligible capital base.",
            "The tenure of the Internal Ombudsman shall be a fixed term of not less than three years and not more than five years in total.",
            "Banks shall review the risk categorisation of customer accounts at least once in every six months."], 1):
            story.append(Paragraph(t, hang, bulletText=f"({i})"))
        story += [Paragraph("Provisioning", h2),
                  Paragraph("A bank shall make incremental provisions on unhedged foreign currency exposures of borrowers as under:", body)]
        rows = [[Paragraph("Likely loss / earnings before interest and depreciation (per cent)", cell), Paragraph("Incremental provisioning requirement", cell)],
                [Paragraph("Up to 15 per cent", cell), Paragraph("Nil", cell)],
                [Paragraph("More than 30 per cent and up to 50 per cent", cell), Paragraph("40 bps", cell)],
                [Paragraph("More than 75 per cent", cell), Paragraph("80 bps", cell)]]
        t = Table(rows, colWidths=[9 * cm, 6 * cm]); t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
        story.append(t)
        doc.build(story)


def regulatory_retriever():
    if "reg" not in _cache:
        d = Path(tempfile.mkdtemp(prefix="regpdfs_")); make_regulatory_pdfs(d); _cache["regpdfs"] = d
    return _retriever(_cache["regpdfs"], "reg")
