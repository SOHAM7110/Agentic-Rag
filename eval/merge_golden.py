"""python -m eval.merge_golden [--out eval/golden_set.jsonl] [--seed 42] [--keep-all]
Merges eval/golden_round*.jsonl into one golden set, trimming over-represented types (seeded, reproducible) so the mix
approaches the target (single 35 / multihop 25 / numerical 15 / unanswerable 15 / ambiguous 10 per 100). Scarce types are always kept in full.
Backs up an existing golden_set.jsonl to golden_set.previous.jsonl first. Re-run `python -m eval.verify_golden` afterwards."""
from __future__ import annotations
import argparse, glob, json, random, shutil
from collections import Counter, defaultdict
from pathlib import Path
from config import GOLDEN_PATH

CAP = {"single": 35, "multihop": 25, "numerical": 15, "unanswerable": 15, "ambiguous": 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(GOLDEN_PATH)); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--keep-all", action="store_true")
    a = ap.parse_args()
    here = Path(__file__).parent
    rows = [json.loads(l) for f in sorted(glob.glob(str(here / "golden_round*.jsonl"))) for l in open(f, encoding="utf-8") if l.strip()]
    ids = [r["id"] for r in rows]; assert len(ids) == len(set(ids)), "duplicate ids"
    by = defaultdict(list)
    for r in rows: by[r["type"]].append(r)
    rnd = random.Random(a.seed); kept = []
    for t, items in by.items():
        if not a.keep_all and len(items) > CAP.get(t, 10**9):
            items = sorted(rnd.sample(items, CAP[t]), key=lambda r: r["id"])
        kept += items
    kept.sort(key=lambda r: r["id"])
    out = Path(a.out)
    if out.exists(): shutil.copy(out, out.with_name("golden_set.previous.jsonl"))
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8")
    c = Counter(r["type"] for r in kept); n = len(kept)
    print(f"read {len(rows)} questions -> wrote {n} to {out}")
    print({t: f"{c[t]} ({100 * c[t] // n}%)" for t in CAP})


if __name__ == "__main__":
    main()
