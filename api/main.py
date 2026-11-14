"""FastAPI app.  uvicorn api.main:app --port 8000
POST /query   {"question": "...", "system": "agent|hybrid_rerank_rag|naive_rag"}
POST /ingest  multipart PDFs -> rebuilds the index
GET  /health"""
from __future__ import annotations
import time
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from .service import service, SYSTEMS

app = FastAPI(title="Agentic RAG Document Assistant", version="1.0")


class QueryIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    system: str = "agent"


@app.get("/health")
def health():
    return service.health()


@app.post("/query")
def query(body: QueryIn):
    if body.system not in SYSTEMS:
        raise HTTPException(400, f"system must be one of {SYSTEMS}")
    t0 = time.perf_counter()
    res = service.query(body.question, body.system)
    res["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    bad = [f.filename for f in files if not (f.filename or "").lower().endswith(".pdf")]
    if bad:
        raise HTTPException(400, f"only PDF files are accepted: {bad}")
    return service.ingest([(f.filename, await f.read()) for f in files])
