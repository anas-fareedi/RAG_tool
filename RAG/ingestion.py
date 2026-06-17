import os
import hashlib
import uuid
import logging
import base64
import pdfplumber
from pypdf import PdfReader
from typing import List, Dict, Any, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from RAG.db import add_document, add_chunks, get_document_by_hash, delete_document
from RAG.vector_store import upsert_chunks, PatchedGoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

def get_file_hash(file_path: str) -> str:
    """Computes the SHA-256 hash of a file for duplicate check."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def table_to_markdown(table: List[List[Any]]) -> str:
    """Formats a raw table from pdfplumber as a clean Markdown table."""
    if not table or not table[0]:
        return ""
    # Clean up cells to convert None to empty strings and strip extra spaces
    cleaned = [[str(cell or "").strip() for cell in row] for row in table]
    headers = cleaned[0]
    num_cols = len(headers)
    rows = cleaned[1:]
    
    # Form markdown structure
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * num_cols) + " |\n"
    for row in rows:
        # Pad short rows so the Markdown table stays well-formed
        padded = row + [""] * (num_cols - len(row))
        markdown += "| " + " | ".join(padded[:num_cols]) + " |\n"
    return markdown

def describe_image_with_gemini(image_bytes: bytes, mime_type: str) -> str:
    """Calls Gemini-3.1-Flash-Lite VLM to generate a brief summary of a visual element."""
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{image_base64}"
        
        prompt = (
            "Describe this chart, diagram, or image concisely in under 3 sentences for search indexing. "
            "Focus on: 1. Chart/Image type. 2. Main subject/variables. 3. Key trend or takeaway. "
            "Do not write conversational or introductory text (like 'Here is a description...')."
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        )
        
        response = chat.invoke([message])
        return response.content.strip() if response.content else ""
    except Exception as e:
        logger.error("Error describing image with Gemini: %s", e)
        return "[Visual element failed to process]"

def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Splits plain text into overlapping chunks."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
    chunks = []
    if not text or len(text.strip()) == 0:
        return chunks
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break on a space rather than cutting a word
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start + (chunk_size // 2):
                end = last_space
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else len(text)
        if start >= len(text) or chunk_size - overlap <= 0:
            break
            
    return [c for c in chunks if len(c) > 0]

def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """Generates batch embeddings using PatchedGoogleGenerativeAIEmbeddings."""
    if not texts:
        return []
    try:
        embeddings = PatchedGoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        return embeddings.embed_documents(texts)
    except Exception as e:
        raise RuntimeError(f"Error generating embeddings with Gemini: {e}")

def process_pdf(file_path: str, db_path: str, vector_db_path: str) -> Tuple[int, str]:
    """
    Ingests a PDF:
    - Extracts text, tables, and describes images page-by-page.
    - Saves metadata to SQLite.
    - Saves vectors to Qdrant.
    Returns (document_id, message).
    """
    filename = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)
    
    # Check if document already exists
    existing_doc = get_document_by_hash(db_path, file_hash)
    if existing_doc:
        return existing_doc['id'], f"Document '{filename}' already processed (skipped)."
    
    # Open readers
    pypdf_reader = PdfReader(file_path)
    total_pages = len(pypdf_reader.pages)
    
    # Add document to SQLite metadata
    doc_id = add_document(db_path, filename, file_hash, total_pages)
    
    chunks_to_insert = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            plumber_page = pdf.pages[page_idx]
            pypdf_page = pypdf_reader.pages[page_idx]
            
            sibling_order = 0
            
            # 1. Extract and process tables
            tables = plumber_page.extract_tables()
            for t in tables:
                md_table = table_to_markdown(t)
                if md_table:
                    chunks_to_insert.append({
                        "id": str(uuid.uuid4()),
                        "document_id": doc_id,
                        "page_number": page_num,
                        "chunk_type": "table",
                        "content": md_table,
                        "sibling_order": sibling_order,
                        "filename": filename
                    })
                    sibling_order += 1
            
            # 2. Extract and process images (VLM descriptions)
            # pypdf has simple image access
            if hasattr(pypdf_page, "images"):
                for img_idx, img in enumerate(pypdf_page.images):
                    try:
                        # Determine MIME type if possible
                        mime = "image/png"
                        if img.name.lower().endswith(".jpg") or img.name.lower().endswith(".jpeg"):
                            mime = "image/jpeg"
                        
                        desc = describe_image_with_gemini(img.data, mime)
                        if desc:
                            chunks_to_insert.append({
                                "id": str(uuid.uuid4()),
                                "document_id": doc_id,
                                "page_number": page_num,
                                "chunk_type": "image",
                                "content": f"[Image Description: {desc}]",
                                "sibling_order": sibling_order,
                                "filename": filename
                            })
                            sibling_order += 1
                    except Exception as e:
                        logger.warning("Skipped image %s on page %s: %s", img_idx, page_num, e)
            
            # 3. Extract and process standard page text
            page_text = plumber_page.extract_text() or ""
            text_chunks = split_text(page_text)
            for c in text_chunks:
                chunks_to_insert.append({
                    "id": str(uuid.uuid4()),
                    "document_id": doc_id,
                    "page_number": page_num,
                    "chunk_type": "text",
                    "content": c,
                    "sibling_order": sibling_order,
                    "filename": filename
                })
                sibling_order += 1
                
    if not chunks_to_insert:
        return doc_id, f"Uploaded '{filename}', but no content (text, tables, or images) was extracted."
        
    try:
        # Write to SQLite
        add_chunks(db_path, [
            {
                "id": c["id"],
                "document_id": c["document_id"],
                "page_number": c["page_number"],
                "chunk_type": c["chunk_type"],
                "content": c["content"],
                "sibling_order": c["sibling_order"]
            }
            for c in chunks_to_insert
        ])
        
        # Write to Qdrant Vector DB (LangChain will generate embeddings under the hood)
        upsert_chunks(vector_db_path, chunks_to_insert)
        
    except Exception as e:
        # Rollback: remove the orphaned document record so the user can re-upload after fixing the issue
        logger.error("Ingestion failed for '%s', rolling back document record: %s", filename, e)
        try:
            delete_document(db_path, doc_id)
        except Exception as rollback_err:
            logger.error("Rollback also failed: %s", rollback_err)
        raise
    
    return doc_id, f"Successfully processed '{filename}' ({total_pages} pages, {len(chunks_to_insert)} chunks created)."
