"""Central configuration. Everything is overridable through environment variables (or a .env file next to this file)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path = ROOT / ".env"):
    """Minimal .env reader (KEY=value per line, # comments). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
PDF_DIR = Path(os.getenv("PDF_DIR", ROOT / "data" / "pdfs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))
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
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", 900))

# --- Agent hard limits -------------------------------------------------------
MAX_LOOPS = int(os.getenv("MAX_LOOPS", 3))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", 8000))
REFERENCE_YEAR = int(os.getenv("REFERENCE_YEAR", 2026))

# --- real-LLM tuning -----------------------------------------------------------
GRADE_FLOOR = float(os.getenv("GRADE_FLOOR", 0.25))        # min evidence coverage even when the LLM grader says 'sufficient'
LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", 0)) # seconds between LLM calls (free-tier rate limits); e.g. 2.5 for Groq
PROMPT_PASSAGES = int(os.getenv("PROMPT_PASSAGES", 6))     # evidence chunks shown to the LLM
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")   # gpt-oss models: low | medium | high (higher = better but slower/more tokens)
LLM_MAX_WAIT = float(os.getenv("LLM_MAX_WAIT", 90))   # a 429 asking us to wait longer than this (seconds) aborts instead of sleeping
LLM_CONTROL_MODEL = os.getenv("LLM_CONTROL_MODEL", "")   # optional cheaper model for the agent's plan/grade/rewrite/verify steps (separate free-tier quota)
