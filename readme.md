# DeepSeek RAG System

A Retrieval-Augmented Generation system that extracts content from PDF documents and answers questions using the DeepSeek-R1 language model. The system processes text, tables, and images from PDFs to provide accurate, source-grounded responses.

## Features

- **Multi-modal PDF processing**: Extracts text, tables, and OCR content from images
- **Semantic search**: Uses vector embeddings for intelligent content retrieval
- **Local AI processing**: Runs DeepSeek-R1 locally for privacy and control
- **Conversation memory**: Handles follow-up questions with context awareness
- **Source attribution**: Tracks where information comes from in responses

## Prerequisites

- Python 3.8+
- Ollama installed with DeepSeek-R1 model
- Tesseract OCR engine

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd deepseek-rag-system
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama**
   ```bash
   # Install Ollama (visit https://ollama.ai for platform-specific instructions)
   # Pull DeepSeek R1 model
   ollama pull deepseek-r1:1.5b
   ```

4. **Install Tesseract OCR**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   
   # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   ```

## Project Structure

```
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── rag_pipeline.py         # Core RAG orchestration logic
│   ├── pdf_processor.py        # PDF content extraction
│   ├── tessaract_ocr.py       # OCR processing for images
│   └── llm_loader.py          # DeepSeek model initialization
├── data/
│   └── pdfs/                  # Place PDF files here
├── db/
│   └── chroma_db/             # Vector database storage (auto-created)
├── requirements.txt           # Python dependencies
└── README.md
```

## Usage

### 1. Prepare Your Documents
Place PDF files in the `data/pdfs/` directory:
```bash
mkdir -p data/pdfs
cp your-documents.pdf data/pdfs/
```

### 2. Start the System
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The system will:
- Automatically process all PDFs in the data folder
- Create a vector database for searching
- Start the API server on http://localhost:8000

### 3. Query the System
Send POST requests to `/query` endpoint:

**Using FastAPI Interactive Docs:**
Visit http://localhost:8000/docs for the interactive API documentation where you can test queries directly in your browser.

**Using curl:**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the revenue figures for Q3?"}'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What are the revenue figures for Q3?"}
)
print(response.json()["answer"])
```

### 4. Check System Status
Visit http://localhost:8000 for health check or use:
```bash
curl http://localhost:8000
```

## How It Works

1. **Document Processing**: PDFs are processed to extract text, tables, and images
2. **Content Integration**: All extracted content is combined and split into searchable chunks
3. **Vector Storage**: Text chunks are converted to embeddings and stored in ChromaDB
4. **Query Processing**: User questions are converted to vectors and matched against stored content
5. **Response Generation**: DeepSeek-R1 generates answers based only on retrieved relevant content

## Configuration

Key settings can be modified in `rag_pipeline.py`:

```python
CHUNK_SIZE = 1024          # Size of text chunks for processing
CHUNK_OVERLAP = 256        # Overlap between adjacent chunks
```

## API Documentation

### Endpoints

**POST /query**
- **Description**: Submit a question about the ingested documents
- **Request Body**: `{"question": "your question here"}`
- **Response**: `{"answer": "generated answer with sources"}`

**GET /**
- **Description**: Health check endpoint
- **Response**: `{"status": "RAG pipeline is running with Deepseek"}`

## Dependencies

```
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langchain-chroma==0.1.0
langchain-community==0.0.10
langchain-ollama==0.1.0
chromadb==0.4.18
sentence-transformers==2.2.2
PyMuPDF==1.23.8
camelot-py[cv]==0.10.1
pytesseract==0.3.10
Pillow==10.1.0
pandas==2.1.4
```

## Troubleshooting

**Common Issues:**

1. **"Model not found" error**
   - Ensure DeepSeek-R1 model is pulled: `ollama pull deepseek-r1:1.5b`
   - Check Ollama is running: `ollama list`

2. **OCR errors**
   - Verify Tesseract installation: `tesseract --version`
   - Check image file permissions

3. **Empty responses**
   - Ensure PDFs are in `data/pdfs/` directory
   - Check logs in `api.log` for processing errors

4. **Slow processing**
   - Large PDFs take time to process initially
   - Subsequent queries are faster due to vector caching

## Logs

System logs are written to `api.log` for debugging and monitoring purposes.

## Technical Details

For a comprehensive explanation of the methods and reasoning behind this implementation, see the technical documentation provided separately.

## License

This project is for technical evaluation purposes.