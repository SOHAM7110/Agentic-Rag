"""OpenAI-compatible chat client (Groq, Hugging Face Inference Providers, Ollama, OpenAI) using only the stdlib,
with retry/backoff on 429/5xx and a SQLite response cache (so reruns of `make eval` are free and fast)."""
from __future__ import annotations
import hashlib, json, os, re, sqlite3, time, urllib.error, urllib.request
from config import CACHE_PATH, LLM_MIN_INTERVAL, LLM_REASONING_EFFORT, LLM_MAX_WAIT
from .base import LLM, LLMError, LLMQuotaError, LLMResult

PROVIDERS = {
    "groq":   dict(url="https://api.groq.com/openai/v1", key="GROQ_API_KEY", model="llama-3.1-8b-instant"),
    "hf":     dict(url="https://router.huggingface.co/v1", key="HF_TOKEN", model="meta-llama/Llama-3.1-8B-Instruct"),
    "ollama": dict(url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), key=None, model="llama3.1:8b"),
    "openai": dict(url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), key="OPENAI_API_KEY", model="gpt-4o-mini"),
}


class Cache:
    def __init__(self, path=CACHE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k):
        r = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v))); self.db.commit()


_last_call = [0.0]


def _throttle():
    """Space out calls so free-tier rate limits (requests/tokens per minute) are not hit."""
    if LLM_MIN_INTERVAL > 0:
        wait = _last_call[0] + LLM_MIN_INTERVAL - time.time()
        if wait > 0: time.sleep(wait)
        _last_call[0] = time.time()


class OpenAICompatLLM(LLM):
    def __init__(self, provider: str, model: str = "", retries: int = 6, use_cache: bool = True):
        cfg = PROVIDERS[provider]
        self.base, self.key_env = cfg["url"], cfg["key"]
        self.model = model or cfg["model"]
        self.name = f"{provider}:{self.model}"
        self.retries = retries
        self.cache = Cache() if use_cache else None
        self.api_key = os.getenv(self.key_env, "") if self.key_env else ""
        if self.key_env and not self.api_key:
            raise LLMError(f"{self.key_env} is not set")

    def generate(self, system, user, json_mode=False, max_tokens=600) -> LLMResult:
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model.lower():
            # reasoning model: hidden thinking tokens count against max_tokens, so ask for little thinking and leave room
            body["reasoning_effort"] = LLM_REASONING_EFFORT
            body["max_tokens"] = max(max_tokens, 1500)
        optional = ("reasoning_effort", "response_format")      # dropped once if the API rejects them (HTTP 400)
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if self.cache and (hit := self.cache.get(key)):
            return LLMResult(hit["text"], hit["in"], hit["out"], cached=True)
        last = None
        for attempt in range(self.retries):
            try:
                _throttle()
                req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json", "Accept": "application/json",
                                                      "Authorization": f"Bearer {self.api_key}",
                                                      # Cloudflare (in front of Groq/HF) blocks urllib's default UA with HTTP 403 / error 1010
                                                      "User-Agent": "agentic-rag/1.0 (+python)"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.loads(r.read())
                text = d["choices"][0]["message"].get("content") or ""
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # some reasoning models inline their thoughts
                u = d.get("usage", {})
                res = LLMResult(text, u.get("prompt_tokens", (len(system) + len(user)) // 4), u.get("completion_tokens", len(text) // 4))
                if self.cache and text: self.cache.put(key, {"text": text, "in": res.in_tokens, "out": res.out_tokens})
                return res
            except urllib.error.HTTPError as e:
                last = e
                detail = e.read()[:300]
                if e.code in (429, 500, 502, 503, 504):
                    try: hdr = float(e.headers.get("retry-after", 0) or 0)
                    except ValueError: hdr = 0.0
                    if e.code == 429 and hdr > LLM_MAX_WAIT:      # e.g. daily token allowance exhausted: do not sleep for hours
                        raise LLMQuotaError(f"rate limit: server asks to wait {hdr:.0f}s ({hdr / 60:.0f} min). {detail[:200]!r}")
                    time.sleep(min(hdr or 2 ** attempt, LLM_MAX_WAIT)); continue
                if e.code == 400 and any(k in body for k in optional):
                    for k in optional: body.pop(k, None)             # model doesn't accept these options: retry plain
                    continue
                raise LLMError(f"HTTP {e.code}: {detail!r}")
            except Exception as e:  # network error / timeout
                last = e; time.sleep(2 ** attempt)
        raise LLMError(f"gave up after {self.retries} attempts: {last}")
