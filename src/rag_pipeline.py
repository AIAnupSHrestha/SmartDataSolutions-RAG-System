import os
import glob
from typing import List
import chromadb
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import LLMChain

from .llm_loader import load_deepseek_llm
from .pdf_processor import extract_pdf_content
from .tessaract_ocr import ocr_images
import logging

os.environ["CHROMA_DISABLE_TELEMETRY"] = "true"

logger = logging.getLogger("DeepSeekRAGAPI")

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
PERSIST_DIR = os.path.join(BASE_DIR, "db", "chroma_db")
embedder = SentenceTransformerEmbeddings(model_name="BAAI/bge-base-en-v1.5")
# Ensure directories exist
for directory in [PDF_DIR, PERSIST_DIR, IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

DEFAULT_PROMPT_TEMPLATE = """
You are a precise and factual assistant. Answer the question using ONLY the provided context from ingested documents. 
Do NOT include any reasoning, explanations, or "think" processes in your response. If the context lacks the necessary information, 
respond exactly with: "The answer is not available in the provided documents."

Format your answer as follows:
1. List all relevant figures mentioned in the context in chronological order, including date, value, and source filename.
2. Provide a single sentence summarizing the trend based solely on the listed figures.

Do NOT use external knowledge, make assumptions, or include speculative information.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""


def get_pdf_files(pdf_dir: str) -> List[str]:
    """Get all PDF files from the specified directory"""
    if not os.path.exists(pdf_dir):
        logger.warning(f"Data folder {pdf_dir} does not exist")
        return []
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    return pdf_files

def process_pdf_files(pdf_files: List[str]) -> List[Document]:
    """Process PDF files and return document objects"""
    documents = []

    # Create images directory if it doesn't exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    logger.info(f"Ensured images directory exists at {IMAGES_DIR}")

    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path}...")
        try:
            # Extract text, tables, images
            data = extract_pdf_content(pdf_path)

            # Combine extracted text
            full_text = data["text"]

            # Append tables as text
            if data["tables"]:
                table_texts = [df.to_string(index=False) for df in data["tables"]]
                full_text += "\n\n" + "\n\n".join(table_texts)

            # Save images and process with OCR
            if data["images"]:
                for i, image_path_src in enumerate(data["images"]):
                    image_filename = f"image_{i}_{os.path.basename(pdf_path).replace('.pdf', '')}.png"
                    image_path_dst = os.path.join(IMAGES_DIR, image_filename)
                    shutil.copy(image_path_src, image_path_dst)
                    logger.info(f"Saved image to {image_path_dst}")

                ocr_results = ocr_images(data["images"])
                ocr_texts = [ocr["text"] for ocr in ocr_results if ocr["text"].strip()]
                logger.info(f"OCR processed {len(ocr_texts)} texts from images")
                if ocr_texts:
                    full_text += "\n\n" + "\n\n".join(ocr_texts)

            # Create LangChain Document
            documents.append(Document(
                page_content=full_text,
                metadata={"source": os.path.basename(pdf_path)}
            ))

        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
    logger.info(f"Total documents processed: {len(documents)}")
    return documents

def create_or_load_vector_store() -> Chroma:
    """Create or load Chroma vector store with embeddings from processed PDFs"""
    # Load existing Chroma store
    if os.path.exists(PERSIST_DIR):
        logger.info("Loading existing ChromaDB...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedder
        )

    # Create a new Chroma store
    logger.info("No existing ChromaDB found. Creating new store...")
    pdf_files = get_pdf_files(PDF_DIR)
    if not pdf_files:
        logger.warning("No PDF files found in data folder")
        return None

    documents = process_pdf_files(pdf_files)
    if not documents:
        logger.warning("No documents were processed successfully")
        return None

    # Chunk documents for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(split_texts):
            chunks.append(Document(
                page_content=chunk,
                metadata={"source": doc.metadata["source"], "chunk_id": i}
            ))

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=PERSIST_DIR
    )
    logger.info("ChromaDB created and persisted.")
    return vector_store




class RAGPipeline:
    """DeepSeek-R1 powered Retrieval-Augmented Generation pipeline."""
    
    def __init__(self):
        self.vector_store = create_or_load_vector_store()
        if not self.vector_store:
            raise RuntimeError("Vector store could not be created or loaded.")
        self.llm = load_deepseek_llm()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )
        self.prompt = PromptTemplate(
            template=DEFAULT_PROMPT_TEMPLATE,
            input_variables=["context", "question", "chat_history"]
        )
        logger.info("RAG pipeline initialized successfully.")

    def ingest_pdfs(self, pdf_dir: str):
        pdf_files = get_pdf_files(pdf_dir)
        documents = process_pdf_files(pdf_files)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        chunks = []
        for doc in documents:
            split_texts = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(split_texts):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={"source": doc.metadata["source"], "chunk_id": i}
                ))

        self.vector_store.add_documents(chunks)
        logger.info(f"Ingestion complete for PDFs in {pdf_dir}")
    
    def query(self, question: str) -> str:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )
        response = qa_chain.invoke({"question": question})["answer"]
        return response