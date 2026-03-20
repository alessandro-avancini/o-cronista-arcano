"""
Vector Store - Gerenciamento do ChromaDB
=========================================

Módulo para configuração e gerenciamento do banco vetorial ChromaDB.
Persistente no disco para manter dados entre sessões.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

import config.helper as config
from config.settings import CHROMA_ADD_BATCH_SIZE, CHROMA_COLLECTION_NAME
from src.memory.embeddings import (
    SentenceTransformerEmbeddingFunction,
    embed_for_query,
)

logger = logging.getLogger(__name__)

# Cache global do client
_client_cache: Optional[chromadb.PersistentClient] = None

# Cache da collection por nome (evita get_or_create_collection e nova embedding function a cada busca)
_collection_cache: Dict[str, chromadb.Collection] = {}


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Retorna um client persistente do ChromaDB.
    Usa cache para evitar múltiplas conexões.
    
    Returns:
        chromadb.PersistentClient: Client configurado
    """
    global _client_cache
    
    if _client_cache is None:
        db_path = str(config.get_vector_db_dir())
        logger.info(f"Inicializando ChromaDB em: {db_path}")
        
        _client_cache = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        logger.info("ChromaDB inicializado com sucesso!")
    
    return _client_cache


def get_or_create_collection(
    collection_name: str = CHROMA_COLLECTION_NAME
) -> chromadb.Collection:
    """
    Obtém ou cria uma collection no ChromaDB.
    Usa cache para evitar recriação e nova embedding function a cada chamada.
    """
    if collection_name in _collection_cache:
        return _collection_cache[collection_name]

    client = get_chroma_client()
    embedding_function = SentenceTransformerEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"description": "Transcrições do Cronista Arcano"}
    )
    _collection_cache[collection_name] = collection
    logger.info(f"Collection '{collection_name}' obtida. Documentos: {collection.count()}")
    return collection


def add_documents(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    collection_name: str = CHROMA_COLLECTION_NAME,
    embeddings: Optional[List[List[float]]] = None,
) -> None:
    """
    Adiciona documentos à collection.

    Args:
        documents: Lista de textos (armazenados e retornados na query)
        metadatas: Lista de metadados para cada documento
        ids: Lista de IDs únicos para cada documento
        collection_name: Nome da collection
        embeddings: Opcional. Se fornecido, estes vetores são usados em vez de
            embedar documents (ex.: small-to-big RAG com vetor do resumo).
    """
    if not documents:
        logger.warning("Nenhum documento para adicionar")
        return
    if embeddings is not None and len(embeddings) != len(documents):
        raise ValueError(
            f"embeddings length ({len(embeddings)}) must match documents length ({len(documents)})"
        )

    collection = get_or_create_collection(collection_name)

    logger.info(f"Adicionando {len(documents)} documentos à collection...")

    batch_size = CHROMA_ADD_BATCH_SIZE
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        add_kwargs: Dict[str, Any] = {
            "documents": batch_docs,
            "metadatas": batch_meta,
            "ids": batch_ids,
        }
        if embeddings is not None:
            add_kwargs["embeddings"] = embeddings[i:i + batch_size]
        collection.add(**add_kwargs)
        logger.info(f"  Batch {i // batch_size + 1}: {len(batch_docs)} docs adicionados")

    logger.info(f"Total de documentos na collection: {collection.count()}")


def query_collection(
    query_text: str,
    n_results: int = 5,
    collection_name: str = CHROMA_COLLECTION_NAME,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Realiza busca semântica na collection.

    Args:
        query_text: Texto da consulta
        n_results: Número de resultados a retornar
        collection_name: Nome da collection
        where: Filtro opcional de metadados (ex.: {"video_id": "..."})

    Returns:
        Dict com resultados da busca
    """
    collection = get_or_create_collection(collection_name)
    query_embedding = embed_for_query([query_text])

    kwargs = {
        "query_embeddings": query_embedding,
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    return results


def delete_collection(collection_name: str = CHROMA_COLLECTION_NAME) -> bool:
    """
    Remove uma collection do ChromaDB.
    Limpa o cache para que a próxima get_or_create_collection crie nova instância.
    """
    try:
        client = get_chroma_client()
        client.delete_collection(collection_name)
        _collection_cache.pop(collection_name, None)
        logger.info(f"Collection '{collection_name}' removida com sucesso")
        return True
    except Exception as e:
        logger.error(f"Erro ao remover collection: {e}")
        return False


def get_collection_info(collection_name: str = CHROMA_COLLECTION_NAME) -> Dict[str, Any]:
    """
    Retorna informações sobre a collection.
    
    Args:
        collection_name: Nome da collection
        
    Returns:
        Dict com informações da collection
    """
    collection = get_or_create_collection(collection_name)
    
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata
    }


def list_video_ids(collection_name: str = CHROMA_COLLECTION_NAME) -> List[str]:
    """
    Retorna lista de video_id únicos presentes na collection.
    """
    collection = get_or_create_collection(collection_name)
    total = collection.count()
    if total == 0:
        return []
    result = collection.get(
        limit=min(total, 50_000),
        include=["metadatas"],
    )
    metadatas = result.get("metadatas") or []
    ids = sorted({m.get("video_id") for m in metadatas if m and m.get("video_id")})
    return ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testando Vector Store...")
    
    # Info da collection
    info = get_collection_info()
    print(f"\nInfo da collection:")
    print(f"  Nome: {info['name']}")
    print(f"  Documentos: {info['count']}")
    print(f"  Metadata: {info['metadata']}")
    
    # Teste de adição (comentado para não poluir)
    # test_docs = ["Teste documento 1", "Teste documento 2"]
    # test_meta = [{"source": "test"}, {"source": "test"}]
    # test_ids = ["test_1", "test_2"]
    # add_documents(test_docs, test_meta, test_ids)
    
    # Teste de query
    if info['count'] > 0:
        print("\nTestando busca...")
        results = query_collection("personagens do RPG")
        print(f"Resultados encontrados: {len(results['documents'][0])}")
