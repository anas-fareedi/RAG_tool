import os
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

from concurrent.futures import ThreadPoolExecutor
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

class PatchedGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(self.embed_query, texts))

COLLECTION_NAME = "document_chunks"
VECTOR_DIMENSION = 3072  # gemini-embedding-2 output dimension

def get_qdrant_client(storage_path: str) -> QdrantClient:
    """Gets a local-directory client for Qdrant."""
    os.makedirs(storage_path, exist_ok=True)
    return QdrantClient(path=storage_path)

def init_vector_store(storage_path: str):
    """Initializes the Qdrant local collection if it does not exist."""
    client = get_qdrant_client(storage_path)
    
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE),
        )
    client.close()

def get_vector_store(storage_path: str) -> QdrantVectorStore:
    """Gets a LangChain QdrantVectorStore client initialized with local client and PatchedGoogleGenerativeAIEmbeddings."""
    client = get_qdrant_client(storage_path)
    embeddings = PatchedGoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

def upsert_chunks(storage_path: str, chunks: List[Dict[str, Any]]):
    """
    Upserts chunk texts to the local Qdrant collection using LangChain.
    Embeddings are automatically computed using GoogleGenerativeAIEmbeddings.
    """
    if not chunks:
        return
        
    vector_store = get_vector_store(storage_path)
    
    documents = []
    ids = []
    for chunk in chunks:
        # Validate ID is a valid UUID
        uid = chunk['id']
        try:
            uuid.UUID(uid)
        except ValueError:
            uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, uid))
            
        doc = Document(
            page_content=chunk['content'],
            metadata={
                "document_id": int(chunk['document_id']),
                "page_number": int(chunk['page_number']),
                "chunk_type": chunk['chunk_type'],
                "filename": chunk.get('filename', '')
            }
        )
        documents.append(doc)
        ids.append(uid)
        
    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.client.close()

def search_vectors(
    storage_path: str, 
    query_vector: List[float], 
    document_ids: Optional[List[int]] = None, 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Searches the vector store using LangChain's QdrantVectorStore and a pre-computed query vector.
    Applies metadata filtering using Qdrant filters on nested metadata fields.
    """
    vector_store = get_vector_store(storage_path)
    
    query_filter = None
    if document_ids:
        # Match any of the document IDs in the list (metadata is nested under "metadata" key in LangChain payloads)
        conditions = [
            FieldCondition(
                key="metadata.document_id",
                match=MatchValue(value=doc_id)
            ) for doc_id in document_ids
        ]
        query_filter = Filter(should=conditions)
        
    search_results = vector_store.similarity_search_with_score_by_vector(
        embedding=query_vector,
        k=top_k,
        filter=query_filter
    )
    vector_store.client.close()
    
    results = []
    for doc, score in search_results:
        results.append({
            "id": doc.id,
            "score": score,
            "payload": doc.metadata
        })
    return results

def delete_vectors_by_doc(storage_path: str, document_id: int):
    """Deletes all vectors belonging to a specific document ID."""
    client = get_qdrant_client(storage_path)
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
    )
    client.close()
