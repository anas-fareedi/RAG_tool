import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from RAG.db import init_db, list_documents, delete_document
from RAG.vector_store import init_vector_store, delete_vectors_by_doc
from RAG.ingestion import process_pdf
from RAG.query_engine import execute_rag_query

# Compute database and vector store paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "rag_tool.db")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "qdrant")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Initialize Directories and Databases
os.makedirs(TEMP_DIR, exist_ok=True)
init_db(DB_PATH)
init_vector_store(VECTOR_DB_PATH)

app = FastAPI(title="Multimodal Document Intelligence RAG API")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="The search query (1-4000 chars).")
    document_ids: Optional[List[int]] = None

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Multimodal Document Intelligence RAG Tool",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    api_key_set = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "status": "ok",
        "gemini_api_key_configured": api_key_set
    }

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF file and processes it page-by-page for text, tables, and images."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )
        
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY environment variable is not configured on the server."
        )
        
    # Sanitize filename to prevent path traversal attacks
    safe_name = os.path.basename(file.filename)
    if not safe_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename."
        )
    temp_file_path = os.path.join(TEMP_DIR, safe_name)
    try:
        # Save file to temp path
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Reject files larger than 50 MB
        MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
        if os.path.getsize(temp_file_path) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds the 50 MB upload limit."
            )
            
        # Process document
        doc_id, msg = process_pdf(temp_file_path, DB_PATH, VECTOR_DB_PATH)
        return {
            "document_id": doc_id,
            "message": msg
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Executes a grounded query against the processed documents."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY environment variable is not configured on the server."
        )
        
    try:
        answer = execute_rag_query(
            query=request.query,
            db_path=DB_PATH,
            vector_db_path=VECTOR_DB_PATH,
            document_ids=request.document_ids,
            top_k=5
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query knowledge base: {str(e)}"
        )

@app.get("/documents")
async def get_documents():
    """Returns a list of all processed documents."""
    try:
        docs = list_documents(DB_PATH)
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch documents: {str(e)}"
        )

@app.delete("/documents/{document_id}")
async def delete_uploaded_document(document_id: int):
    """Deletes a document from the relational index and vector database."""
    try:
        # Remove from Qdrant Vector Store
        delete_vectors_by_doc(VECTOR_DB_PATH, document_id)
        # Remove from SQLite (will cascade delete chunks)
        delete_document(DB_PATH, document_id)
        return {"message": f"Successfully deleted document ID {document_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )