import os
from typing import List, Optional
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load .env so GEMINI_API_KEY is available when server.py is the entry point
load_dotenv()

from RAG.db import init_db, list_documents
from RAG.vector_store import init_vector_store
from RAG.ingestion import process_pdf
from RAG.query_engine import execute_rag_query

# Compute paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "rag_tool.db")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "qdrant")

# Initialize Databases on import/startup
init_db(DB_PATH)
init_vector_store(VECTOR_DB_PATH)

# Initialize MCP Server
mcp = FastMCP("Local_Multimodal_RAG")

@mcp.tool()
def search_documents(query: str, document_ids: Optional[List[int]] = None) -> str:
    """
    Search across uploaded PDF documents (including text, tables, and images/charts)
    and return a grounded, cited answer using Gemini.
    
    Args:
        query: The user's search query or question.
        document_ids: Optional list of document IDs to restrict search to.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return "Error: GEMINI_API_KEY environment variable is not set on the server."
    
    try:
        answer = execute_rag_query(
            query=query,
            db_path=DB_PATH,
            vector_db_path=VECTOR_DB_PATH,
            document_ids=document_ids,
            top_k=5
        )
        return answer
    except Exception as e:
        return f"Error executing query: {str(e)}"

@mcp.tool()
def upload_document(file_path: str) -> str:
    """
    Ingest a local PDF document into the RAG knowledge base.
    This processes text, tables, and images, creating embeddings and structural indexes.
    
    Args:
        file_path: The absolute local file path to the PDF document.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist on the local system."
    
    if not file_path.lower().endswith(".pdf"):
        return "Error: Only PDF files are supported for ingestion currently."
        
    if not os.environ.get("GEMINI_API_KEY"):
        return "Error: GEMINI_API_KEY environment variable is not set on the server."
        
    try:
        _, msg = process_pdf(file_path, DB_PATH, VECTOR_DB_PATH)
        return msg
    except Exception as e:
        return f"Error during ingestion: {str(e)}"

@mcp.tool()
def get_uploaded_documents() -> str:
    """
    Retrieve a list of all documents that have been ingested into the knowledge base,
    showing their file names, total pages, and internal IDs.
    """
    try:
        docs = list_documents(DB_PATH)
        if not docs:
            return "No documents uploaded yet."
        
        lines = ["Uploaded Documents:"]
        for d in docs:
            lines.append(f"- [ID: {d['id']}] {d['filename']} ({d['total_pages']} pages, uploaded: {d['created_at']})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing documents: {str(e)}"

if __name__ == "__main__":
    # Standard stdio mode for MCP host connection
    mcp.run()