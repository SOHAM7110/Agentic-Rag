"""Judge-vs-human agreement, built so it is NOT trivial: the sample is stratified over question types (answerable ones included) and
mixes the agent with the single-shot baseline.
  python -m eval.hand_grade sample   -> results/hand_grade.csv (sheet you fill) + results/hand_grade_key.csv (judge scores: do not open it first)
  python -m eval.hand_grade agree    -> % agreement and Cohen's kappa overall / answerable / unanswerable, plus how the judge errs
Run `sample` after the final evaluation so the rows come from the final code (results/per_question.jsonl accumulates across runs)."""
from __future__ import annotations
import csv, json, random, sys
from config import RESULTS_DIR

SHEET, KEY = RESULTS_DIR / "hand_grade.csv", RESULTS_DIR / "hand_grade_key.csv"
QUOTA = {"single": 5, "multihop": 5, "numerical": 4, "ambiguous": 2, "unanswerable": 4}
SYSTEMS = ("agent", "hybrid_rerank_rag")


def pick_sample(rows: list[dict], quota: dict = QUOTA, systems=SYSTEMS, seed: int = 1) -> list[dict]:
    rows = [r for r in rows if r["system"] in systems]
    rnd = random.Random(seed); out, used = [], set()
    for t, n_t in quota.items():
        pool = [r for r in rows if r["type"] == t]; rnd.shuffle(pool)
        by_sys = {s: [r for r in pool if r["system"] == s] for s in systems}
        k = 0
        while len([x for x in out if x["type"] == t]) < n_t and any(by_sys.values()):
            s = systems[k % len(systems)]; k += 1
            while by_sys[s] and (by_sys[s][0]["id"], s) in used: by_sys[s].pop(0)
            if by_sys[s]:
                r = by_sys[s].pop(0); used.add((r["id"], s)); out.append(r)
    return out


def sample(n: int = 20):
    rows = [json.loads(l) for l in (RESULTS_DIR / "per_question.jsonl").read_text().splitlines() if l.strip()]
    chosen = pick_sample(rows)
    if len(chosen) < n:                       # not every type evaluated yet: top up with any other rows
        have = {(r["id"], r["system"]) for r in chosen}
        rest = [r for r in rows if r["system"] in SYSTEMS and (r["id"], r["system"]) not in have]; random.Random(2).shuffle(rest)
        chosen += rest[: n - len(chosen)]
    random.Random(3).shuffle(chosen)
    with open(SHEET, "w", newline="", encoding="utf-8") as f, open(KEY, "w", newline="", encoding="utf-8") as k:
        w = csv.writer(f); wk = csv.writer(k)
        w.writerow(["id", "system", "type", "question", "gold_answer", "system_answer", "human", "notes"]); wk.writerow(["id", "system", "judge_score"])
        for r in chosen:
            w.writerow([r["id"], r["system"], r["type"], r["question"], r["gold_answer"], r["answer"], "", ""]); wk.writerow([r["id"], r["system"], r["correct"]])
    kinds = {t: sum(r["type"] == t for r in chosen) for t in QUOTA}
    print(f"wrote {SHEET} ({len(chosen)} answers: {kinds}). Fill 'human' with 1 (correct) / 0 (incorrect) by comparing system_answer with gold_answer - "
          f"for refusals, 1 = a refusal was the right behaviour. Do not open {KEY.name} until you have finished. Then run `agree`.")


def _stats(pairs):
    n = len(pairs)
    if n == 0: return None
    h = [a for a, _ in pairs]; j = [b for _, b in pairs]
    po = sum(a == b for a, b in pairs) / n
    pe = sum((h.count(c) / n) * (j.count(c) / n) for c in (0, 1))
    return {"n": n, "agreement": round(po, 3), "cohen_kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
            "judge_too_lenient": sum(1 for a, b in pairs if b == 1 and a == 0), "judge_too_strict": sum(1 for a, b in pairs if b == 0 and a == 1)}


def agree():
    key = {(r["id"], r["system"]): float(r["judge_score"]) for r in csv.DictReader(open(KEY, encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(SHEET, encoding="utf-8")) if r["human"].strip() in ("0", "1")]
    if not rows: raise SystemExit("no human labels found in the 'human' column")
    pairs = [(int(r["human"]), int(key[(r["id"], r["system"])] >= 0.5), r["type"]) for r in rows]
    out = {"overall": _stats([(a, b) for a, b, _ in pairs]),
           "answerable": _stats([(a, b) for a, b, t in pairs if t != "unanswerable"]),
           "unanswerable": _stats([(a, b) for a, b, t in pairs if t == "unanswerable"])}
    n_ans = out["answerable"]["n"] if out["answerable"] else 0
    if n_ans < 10: print(f"WARNING: only {n_ans} answerable answers labelled - agreement on them is not meaningful; label at least 10.")
    flat = dict(out["overall"], answerable=out["answerable"], unanswerable=out["unanswerable"])
    (RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(flat, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"sample": sample, "agree": agree}[sys.argv[1]]()
