"""Central configuration. Everything is overridable through environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDF_DIR = Path(os.getenv("PDF_DIR", ROOT / "data" / "pdfs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
RESULTS_DIR = ROOT / "results"
GOLDEN_PATH = Path(os.getenv("GOLDEN_PATH", ROOT / "eval" / "golden_set.jsonl"))
LOG_PATH = Path(os.getenv("LOG_PATH", ROOT / "results" / "query_log.jsonl"))
CACHE_PATH = Path(os.getenv("LLM_CACHE_PATH", ROOT / "data" / "llm_cache.sqlite"))

# --- LLM (generation) -------------------------------------------------------
# offline | groq | hf | ollama | openai
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline")
LLM_MODEL = os.getenv("LLM_MODEL", "")          # empty -> provider default
# --- LLM (judge): must differ from the generator when a real LLM is used -----
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "offline")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")

# --- Retrieval ---------------------------------------------------------------
EMBEDDER = os.getenv("EMBEDDER", "auto")        # auto | st | lsa
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER = os.getenv("RERANKER", "auto")        # auto | cross | lexical
VECTOR_STORE = os.getenv("VECTOR_STORE", "auto")  # auto | chroma | numpy
TOP_K = int(os.getenv("TOP_K", 5))
CANDIDATES = int(os.getenv("CANDIDATES", 20))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", 400))

# --- Agent hard limits -------------------------------------------------------
MAX_LOOPS = int(os.getenv("MAX_LOOPS", 3))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", 8000))
REFERENCE_YEAR = int(os.getenv("REFERENCE_YEAR", 2026))
