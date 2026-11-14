.PHONY: install demo-corpus index eval test api ui docker hand-grade clean
PY ?= python3

install:            ## full deps (PyMuPDF, ChromaDB, sentence-transformers, LangGraph, FastAPI, Streamlit)
	$(PY) -m pip install -r requirements.txt

demo-corpus:        ## regenerate the SYNTHETIC 30-PDF corpus + 80-question golden set
	$(PY) scripts/make_demo_corpus.py

index:              ## parse -> section-aware chunks -> embeddings -> vector store (+ facts table)
	$(PY) -m ingest.build_index

eval: index         ## ONE COMMAND: ablation + 3-system comparison; regenerates results/ and the README table
	$(PY) -m eval.run_eval

hand-grade:         ## write a 20-answer sheet for judge-vs-human agreement
	$(PY) -m eval.hand_grade sample

test:
	$(PY) -m tests.test_core

api:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

ui:
	$(PY) -m streamlit run ui/app.py --server.port 8501

docker:
	docker build -t agentic-rag . && docker run -p 7860:7860 --env-file .env agentic-rag

clean:
	rm -rf data/index data/llm_cache.sqlite results/query_log.jsonl
