"""Section-aware chunking. Chunks never cross a heading; real tables stay together (wrapped cells included) and carry the sentence
that introduces them; every chunk keeps doc / section / page metadata which later powers citations and evaluation."""
from __future__ import annotations
import re
from config import CHUNK_MAX_CHARS
from retrieval.text import split_sentences
from .parse import MARKER, ParsedDoc, is_heading


def chunk_doc(doc: ParsedDoc, max_chars: int = CHUNK_MAX_CHARS) -> list[dict]:
    units: list[tuple[str, str, str, int]] = []       # (section, kind, text, page)
    leads: dict[int, str] = {}                         # unit index of a table's first row -> sentence that introduces it
    section = "Front matter"
    title_line = doc.title
    buf: list[tuple[str, int]] = []

    def flush_prose():
        nonlocal buf
        if buf:
            text = " ".join(t for t, _ in buf)
            page_of, pos = {}, 0
            for t, p in buf:
                page_of[pos] = p
                pos += len(t) + 1
            for sent in split_sentences(text):
                idx = text.find(sent[:30])
                pg = buf[0][1]
                for start, p in page_of.items():
                    if start <= max(idx, 0):
                        pg = p
                units.append((section, "prose", sent, pg))
            buf = []

    for pno in sorted(doc.pages):
        for line in doc.pages[pno]:
            if line.text == title_line and pno == 1 and line.size > doc.body_size + 3:
                continue
            # Hanging paragraph labels may be extracted on a separate line at
            # the vertical position of a wrapped text line. Keeping them here
            # inserts e.g. ``(1)`` into the middle of a sentence after prose
            # joining, and can make its meaning appear to change.
            if MARKER.fullmatch(line.text.strip()):
                continue
            if is_heading(line, doc.body_size):
                flush_prose()
                section = re.sub(r"^\d+(\.\d+)*\.?\s*", "", line.text).strip()
                continue
            if " | " in line.text:
                flush_prose()
                if units and units[-1][1] == "prose" and units[-1][0] == section:
                    leads[len(units)] = units[-1][2]
                units.append((section, "table", line.text, pno))
            elif (units and units[-1][1] == "table" and not buf and units[-1][3] == pno and len(line.text) <= 45
                  and not line.text.rstrip().endswith((".", ":"))):
                s, k, t, pg = units[-1]                # wrapped table cell ("50 per cent", "(EBID)"): keep it in the table
                units[-1] = (s, k, t + "\n" + line.text, pg)
            else:
                buf.append((line.text, pno))
    flush_prose()

    chunks, cur, cur_len, cur_sec, cur_lead = [], [], 0, None, None

    def emit():
        nonlocal cur, cur_len, cur_lead
        if not cur:
            return
        pages = sorted({u[3] for u in cur})
        body = "\n".join(u[2] for u in cur) if cur[0][1] == "table" else " ".join(u[2] for u in cur)
        if cur[0][1] == "table" and cur_lead:
            body = cur_lead + "\n" + body
        chunks.append({
            "id": f"{doc.doc_id}::{len(chunks):03d}", "doc_id": doc.doc_id, "doc_title": doc.title,
            "section": cur[0][0], "page": pages[0], "page_end": pages[-1], "text": body, "kind": cur[0][1],
        })
        cur, cur_len, cur_lead = [], 0, None

    for ui, u in enumerate(units):
        sec, kind, text, pg = u
        if cur and (sec != cur_sec or kind != cur[0][1] or cur_len + len(text) > max_chars
                    or (kind == "table" and cur[0][3] != pg)):
            last = cur[-1] if (kind == "prose" and sec == cur_sec and cur[0][1] == "prose") else None
            emit()
            if last is not None:
                cur, cur_len = [last], len(last[2])
        if not cur and kind == "table":
            cur_lead = leads.get(ui)
        cur.append(u)
        cur_len += len(text) + 1
        cur_sec = sec
    emit()
    return chunks
