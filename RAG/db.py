import sqlite3
import os
from typing import List, Dict, Any, Optional

def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Establishes a connection to the SQLite database, creating directories if needed."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Enable foreign key enforcement (disabled by default in Python's sqlite3 module)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db(db_path: str):
    """Initializes the SQLite schema for tracking documents and chunks."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Create documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        file_hash TEXT UNIQUE NOT NULL,
        total_pages INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create chunks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id INTEGER,
        page_number INTEGER,
        chunk_type TEXT CHECK(chunk_type IN ('text', 'table', 'image')),
        content TEXT NOT NULL,
        sibling_order INTEGER,
        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
    )
    """)
    
    # Create indexes for fast retrieval
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks (document_id, page_number)")
    
    conn.commit()
    conn.close()

def add_document(db_path: str, filename: str, file_hash: str, total_pages: int) -> int:
    """Adds a document metadata record and returns its ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO documents (filename, file_hash, total_pages) VALUES (?, ?, ?)",
            (filename, file_hash, total_pages)
        )
        conn.commit()
        doc_id = cursor.lastrowid
        return doc_id
    except sqlite3.IntegrityError:
        # Document already exists, retrieve its ID
        cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        return row['id']
    finally:
        conn.close()

def add_chunks(db_path: str, chunks: List[Dict[str, Any]]):
    """Bulk inserts chunks into the chunks table."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO chunks (id, document_id, page_number, chunk_type, content, sibling_order)
            VALUES (:id, :document_id, :page_number, :chunk_type, :content, :sibling_order)
            """,
            chunks
        )
        conn.commit()
    finally:
        conn.close()

def get_document_by_hash(db_path: str, file_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieves document record by file hash."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_chunk(db_path: str, chunk_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a chunk by its ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_sibling_chunks(db_path: str, document_id: int, page_number: int) -> List[Dict[str, Any]]:
    """Retrieves all chunks belonging to a specific page of a document, ordered chronologically."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT * FROM chunks WHERE document_id = ? AND page_number = ? ORDER BY sibling_order",
            (document_id, page_number)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def list_documents(db_path: str) -> List[Dict[str, Any]]:
    """Lists all uploaded documents."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def delete_document(db_path: str, document_id: int):
    """Deletes a document and cascade deletes its chunks."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
    finally:
        conn.close()
