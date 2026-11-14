"""python scripts/download_corpus.py urls.txt   -> downloads each URL (one per line) into data/pdfs/
Use this to build the REAL corpus (e.g. PM-KISAN / Ayushman Bharat guidelines, regulator circulars, library docs).
Remove the synthetic PDFs first:  rm data/pdfs/*.pdf"""
import re, sys, urllib.request
from pathlib import Path

out = Path(__file__).resolve().parents[1] / "data" / "pdfs"; out.mkdir(parents=True, exist_ok=True)
for url in [u.strip() for u in Path(sys.argv[1]).read_text().splitlines() if u.strip() and not u.startswith("#")]:
    name = re.sub(r"[^a-z0-9]+", "_", url.split("/")[-1].lower().replace(".pdf", "")).strip("_") or "doc"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        (out / f"{name}.pdf").write_bytes(urllib.request.urlopen(req, timeout=60).read()); print("ok  ", name)
    except Exception as e:
        print("FAIL", url, e)
