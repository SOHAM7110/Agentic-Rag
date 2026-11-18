# Golden set (`golden_set.jsonl`, one JSON object per line)
Built in four rounds (`golden_round1-4.jsonl`, written from the PDFs; every gold page re-verified by `python -m eval.verify_golden`) and merged by
`python -m eval.merge_golden`, which trims over-represented types (seeded) toward 35/25/15/15/10 % single/multihop/numerical/unanswerable/tricky.

| field | meaning |
|---|---|
| `id`, `type` | `single` / `multihop` / `numerical` / `unanswerable` / `ambiguous` |
| `question`, `gold_answer` | what is asked; the full gold answer shown to the LLM judge |
| `answer_keys` | normalised strings that must appear in a correct answer (used by the rule-based judge only) |
| `gold_sources` | `[{"doc": "<pdf stem>", "pages": [5], "must_contain": ["..."]}]`; one entry per document; empty for unanswerable. `must_contain` is checked against those pages by `verify_golden` |
| `winner`/`loser`, `answer_prefix`, `subtype` | optional: verdict-order check, yes/no check, unanswerable flavour (`not_in_docs`, `near_miss`) |

Review the questions yourself: a golden set written by one author is not independent evidence until a second pair of eyes has checked it.
Questions used while debugging the agent are listed in the Makefile (`DEV_IDS`); report them separately.
