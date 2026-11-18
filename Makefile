.PHONY: install index eval eval-retrieval demo test diagnose check-linking hand-grade golden clean
PY ?= python3

install:            ## full stack: PyMuPDF, ChromaDB, bge-small + cross-encoder, LangGraph, FastAPI, Streamlit
	$(PY) -m pip install -r requirements.txt

index:              ## parse -> section-aware chunks -> embeddings -> vector store   (set CHUNK_MAX_CHARS=900 for the RBI documents)
	$(PY) -m ingest.build_index

eval-retrieval:     ## free: retrieval ablation only (no LLM, no API key)
	$(PY) -m eval.run_eval --retrieval-only

eval: index         ## the full three-system evaluation (needs LLM_PROVIDER / JUDGE_PROVIDER, see .env.example)
	$(PY) -m eval.run_eval --exclude $(DEV_IDS)

demo:               
	RESULTS_DIR=data/demo_results UPDATE_README=0 CHUNK_MAX_CHARS=400 PDF_DIR=data/demo_pdfs INDEX_DIR=data/demo_index GOLDEN_PATH=data/demo_golden.jsonl LLM_PROVIDER=offline JUDGE_PROVIDER=offline $(PY) -m ingest.build_index
	RESULTS_DIR=data/demo_results UPDATE_README=0 CHUNK_MAX_CHARS=400 PDF_DIR=data/demo_pdfs INDEX_DIR=data/demo_index GOLDEN_PATH=data/demo_golden.jsonl LLM_PROVIDER=offline JUDGE_PROVIDER=offline $(PY) -m eval.run_eval

golden:             ## merge golden_round*.jsonl -> golden_set.jsonl and verify every gold page against the PDFs
	$(PY) -m eval.merge_golden && $(PY) -m eval.verify_golden

test:
	$(PY) -m tests.test_core && $(PY) -m tests.test_llm_paths

diagnose:           ## print every agent step for a few questions (make diagnose IDS=r1_027,r4_006)
	$(PY) scripts/diagnose.py --ids=$(IDS)

check-linking:      ## free: document linking + per-document retrieval check with your real models
	$(PY) scripts/check_linking.py

hand-grade:         ## 20-answer sheet for judge-vs-human agreement (then: python -m eval.hand_grade agree)
	$(PY) -m eval.hand_grade sample

clean:
	rm -rf data/index data/demo_index data/llm_cache.sqlite results/query_log.jsonl

# questions used while debugging the agent (report them separately, or leave them out as `make eval` does)
DEV_IDS = r1_001,r1_020,r2_013,r2_033,r3_022,r4_004,r1_024,r1_027
