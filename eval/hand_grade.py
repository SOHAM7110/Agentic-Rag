"""Judge-vs-human agreement.
  python -m eval.hand_grade sample   -> writes results/hand_grade.csv (20 random answers, blank 'human' column)
  python -m eval.hand_grade agree    -> reads your 0/1 labels, reports % agreement and Cohen's kappa
"""
from __future__ import annotations
import csv, json, random, sys
from config import RESULTS_DIR

PATH = RESULTS_DIR / "hand_grade.csv"


def sample(n=20, system="agent", seed=1):
    rows = [json.loads(l) for l in (RESULTS_DIR / "per_question.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["system"] == system]
    random.Random(seed).shuffle(rows)
    # UTF-8 preserves Unicode answers and lets Excel detect the encoding on Windows.
    with open(PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "type", "question", "gold_answer", "system_answer", "judge_score", "human"])
        for r in rows[:n]:
            w.writerow([r["id"], r["type"], r["question"], r["gold_answer"], r["answer"], r["correct"], ""])
    print(f"wrote {PATH} - fill the 'human' column with 1 (correct) / 0 (incorrect), then run `agree`")


def agree():
    with open(PATH, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["human"].strip() in ("0", "1")]
    if not rows: raise SystemExit("no human labels found")
    h = [int(r["human"]) for r in rows]; j = [int(float(r["judge_score"]) >= 0.5) for r in rows]
    n = len(h); po = sum(a == b for a, b in zip(h, j)) / n
    pe = sum((h.count(c) / n) * (j.count(c) / n) for c in (0, 1))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    out = {"n": n, "agreement": round(po, 3), "cohen_kappa": round(kappa, 3)}
    (RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(out)); print(out)


if __name__ == "__main__":
    {"sample": sample, "agree": agree}[sys.argv[1]]()
