import logging
import time
import re, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .rag_pipeline import RAGPipeline

# === Logging Config ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DeepSeek RAG System")

app = FastAPI(title="DeepSeek RAG System")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
PERSIST_DIR = os.path.join(BASE_DIR, "db", "chroma_db")

rag = RAGPipeline()

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting RAG API...")
        logger.info("Pipeline loaded successfully.")

        rag.ingest_pdfs(PDF_DIR)
    except Exception as e:
        logger.critical(f"Pipeline or ingestion failed: {e}", exc_info=True)
        raise RuntimeError("Pipeline or ingestion initialization failed.")


def extract_final_answer(response: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return cleaned

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    logger.info(f"Received query: {request.question}")
    start_time = time.time()

    try:
        answer = rag.query(request.question)
        elapsed = time.time() - start_time
        logger.info(f"Query processed in {elapsed:.2f} seconds.")
        return QueryResponse(answer=extract_final_answer(answer))
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing your request.")

@app.get("/")
def health_check():
    logger.info("Health check called.")
    return {"status": "RAG pipeline is running with Deepseek"}
