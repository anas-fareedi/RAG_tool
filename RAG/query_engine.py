import os
from typing import List, Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from RAG.db import get_sibling_chunks
from RAG.vector_store import search_vectors, PatchedGoogleGenerativeAIEmbeddings

def get_query_embedding(query: str) -> List[float]:
    """Generates an embedding vector for the search query."""
    try:
        embeddings = PatchedGoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        return embeddings.embed_query(query)
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding for query: {e}")

def execute_rag_query(
    query: str, 
    db_path: str, 
    vector_db_path: str, 
    document_ids: Optional[List[int]] = None,
    top_k: int = 5
) -> str:
    """
    Executes the grounded query pipeline:
    1. Embed query.
    2. Search local Qdrant index.
    3. Retrieve full-page context from SQLite (sibling expansion).
    4. Compile grounded prompt.
    5. Generate response using Gemini-1.5-Flash.
    """
    # 1. Embed query
    query_vector = get_query_embedding(query)
    
    # 2. Search local vector store
    search_results = search_vectors(vector_db_path, query_vector, document_ids, top_k)
    if not search_results:
        return "No relevant information found in the documents. Please upload documents first."
        
    # 3. Sibling expansion & Deduplication
    # Group results by (document_id, page_number) to avoid loading the same page multiple times
    pages_to_load = {}
    for hit in search_results:
        payload = hit['payload']
        doc_id = payload['document_id']
        page_num = payload['page_number']
        filename = payload.get('filename', f"Doc-{doc_id}")
        
        page_key = (doc_id, page_num)
        if page_key not in pages_to_load:
            pages_to_load[page_key] = filename
            
    # Load and format the full content for each unique page
    context_blocks = []
    for (doc_id, page_num), filename in pages_to_load.items():
        # Get all chunks for this page ordered chronologically (by sibling_order)
        page_chunks = get_sibling_chunks(db_path, doc_id, page_num)
        
        # Combine chunk content
        page_content_parts = []
        for chunk in page_chunks:
            chunk_type = chunk['chunk_type']
            content = chunk['content']
            if chunk_type == "table":
                page_content_parts.append(f"\n[Structured Table]:\n{content}\n")
            elif chunk_type == "image":
                page_content_parts.append(f"\n{content}\n")
            else:
                page_content_parts.append(content)
                
        full_page_text = "\n".join(page_content_parts)
        context_blocks.append(
            f"--- START SOURCE: {filename} (Page {page_num}) ---\n"
            f"{full_page_text}\n"
            f"--- END SOURCE: {filename} (Page {page_num}) ---\n"
        )
        
    context_str = "\n\n".join(context_blocks)
    
    # 4. Formulate the grounded prompt
    system_instruction = (
        "You are a helpful, highly accurate Multimodal Document Intelligence Assistant. "
        "Your task is to answer the user's question using ONLY the provided document context. "
        "Follow these strict instructions:\n"
        "1. Base your answer exclusively on the retrieved sources between '--- START SOURCE' and '--- END SOURCE'. "
        "Do not use external knowledge, do not assume, and do not extrapolate.\n"
        "2. If the answer cannot be found in the provided sources, explicitly state: "
        "'I cannot find the answer to this in the uploaded documents.'\n"
        "3. Format your answers clearly. If tables are present in the sources, you may construct summary tables. "
        "Keep the responses clean and structured.\n"
        "4. **CRITICAL CITATIONS:** You MUST cite the source document and page number for every fact you present. "
        "Use the exact format: [Filename.pdf (Page X)]. Do not write citations from memory; use the source boundaries."
    )
    
    prompt = (
        f"Retrieved Document Context:\n"
        f"{context_str}\n\n"
        f"User Query: {query}\n\n"
        f"Answer:"
    )
    
    # 5. Generate response using Gemini-3.1-Flash-Lite (via LangChain)
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "{prompt_text}")
        ])
        
        chain = prompt_tmpl | chat | StrOutputParser()
        response = chain.invoke({"prompt_text": prompt})
        return response.strip() if response else "No response text generated."
    except Exception as e:
        return f"Error generating answer with Gemini: {e}"
