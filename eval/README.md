# Golden set format (`golden_set.jsonl`, one JSON object per line)
| field | meaning |
|---|---|
| `id`, `type` | `single` / `multihop` / `numerical` / `unanswerable` / `ambiguous` |
| `question` | what the user asks |
| `gold_answer` | the full gold answer (shown to the LLM judge) |
| `answer_keys` | lowercase strings/numbers that must appear in a correct answer (used by the rule-based judge) |
| `gold_sources` | `[{"doc": "<doc_id = pdf file stem>", "pages": [3, 4]}]` - any listed page is acceptable; multi-doc questions list one entry per document. Empty for unanswerable |
| `winner`/`loser`, `answer_prefix` | optional: 'which is higher' verdict check / yes-no check |

**For a real project, hand-write these from the real PDFs *before* building the agent.** The shipped file is generated
from synthetic facts (`make demo-corpus`) and verified by re-parsing the PDFs for the gold pages.
