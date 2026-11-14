# Deploy the demo on Hugging Face Spaces (free CPU)
1. Create a new Space -> SDK: **Docker**, hardware: CPU basic.
2. Space *Settings -> Variables and secrets*: add secret `GROQ_API_KEY` (or `HF_TOKEN`) and variables `LLM_PROVIDER=groq`.
3. Put this at the very top of the Space's README.md (not needed in the GitHub README):
   ```
   ---
   title: Agentic RAG Document Assistant
   sdk: docker
   app_port: 7860
   ---
   ```
4. `git push` this repo to the Space remote. The Dockerfile builds the index and starts Streamlit on 7860.
Rate limits: answers are cached (LRU) and any LLM failure falls back to the rule-based backend, so the demo degrades instead of crashing.
