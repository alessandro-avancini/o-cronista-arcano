"""
Memory Module - Sistema de Memória Vetorial
============================================

Expõe as principais funcionalidades do módulo de memória vetorial.
"""

from src.memory.text_chunker import split_text, TextChunk
from src.memory.embeddings import generate_embeddings, get_embedding_model
from src.memory.vector_store import (
    get_chroma_client,
    get_or_create_collection,
    add_documents,
    query_collection,
    delete_collection,
    get_collection_info
)

__all__ = [
    # Text Chunker
    "split_text",
    "TextChunk",
    # Embeddings
    "generate_embeddings",
    "get_embedding_model",
    # Vector Store
    "get_chroma_client",
    "get_or_create_collection",
    "add_documents",
    "query_collection",
    "delete_collection",
    "get_collection_info",
]
