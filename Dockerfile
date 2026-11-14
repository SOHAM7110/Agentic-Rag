FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Bake the index into the image (downloads bge-small + cross-encoder once). The service also rebuilds on first boot if missing.
RUN python -m ingest.build_index || true
EXPOSE 7860
# Default: Streamlit UI (in-process service) on 7860 = Hugging Face Spaces. For the REST API use:
#   docker run -p 8000:8000 agentic-rag uvicorn api.main:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "ui/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
