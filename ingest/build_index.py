"""python -m ingest.build_index  -> parse PDFs, chunk, embed, write data/index/*"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from config import PDF_DIR, INDEX_DIR
from retrieval.embed import make_embedder
from retrieval.retriever import ctx_text
from retrieval.store import make_store
from .chunk import chunk_doc
from .facts import extract_facts
from .parse import parse_pdf


def build(pdf_dir: Path = PDF_DIR, index_dir: Path = INDEX_DIR, verbose: bool = True) -> dict:
    t0 = time.time()
    index_dir = Path(index_dir); index_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}. Run `make demo-corpus` or drop PDFs there.")
    chunks, docs = [], {}
    for p in pdfs:
        d = parse_pdf(p)
        cs = chunk_doc(d)
        chunks += cs
        docs[d.doc_id] = {"title": d.title, "pages": len(d.pages), "chunks": len(cs), "file": p.name}
    facts = extract_facts(chunks)
    emb = make_embedder()
    texts = [ctx_text(c) for c in chunks]
    emb.fit(texts)
    vecs = emb.encode(texts)
    store = make_store(index_dir); store.build(chunks, vecs)
    (index_dir / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks) + "\n")
    (index_dir / "facts.json").write_text(json.dumps(facts, indent=1))
    (index_dir / "docs.json").write_text(json.dumps(docs, indent=1))
    emb.save(index_dir)
    info = {"docs": len(docs), "chunks": len(chunks), "facts_docs": len(facts), "embedder": emb.name, "store": store.name,
            "seconds": round(time.time() - t0, 1)}
    if verbose: print("[ingest]", info)
    return info


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else PDF_DIR)
