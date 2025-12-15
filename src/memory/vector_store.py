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

# Adiciona path do projeto
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config.helper as config
from config.settings import CHROMA_COLLECTION_NAME
from src.memory.embeddings import SentenceTransformerEmbeddingFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache global do client
_client_cache: Optional[chromadb.PersistentClient] = None


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
    
    Args:
        collection_name: Nome da collection
        
    Returns:
        chromadb.Collection: Collection configurada com embedding function
    """
    client = get_chroma_client()
    embedding_function = SentenceTransformerEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"description": "Transcrições do Cronista Arcano"}
    )
    
    logger.info(f"Collection '{collection_name}' obtida. Documentos: {collection.count()}")
    return collection


def add_documents(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    collection_name: str = CHROMA_COLLECTION_NAME
) -> None:
    """
    Adiciona documentos à collection.
    
    Args:
        documents: Lista de textos
        metadatas: Lista de metadados para cada documento
        ids: Lista de IDs únicos para cada documento
        collection_name: Nome da collection
    """
    if not documents:
        logger.warning("Nenhum documento para adicionar")
        return
    
    collection = get_or_create_collection(collection_name)
    
    logger.info(f"Adicionando {len(documents)} documentos à collection...")
    
    # ChromaDB tem limite de batch, então dividimos em lotes
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        logger.info(f"  Batch {i // batch_size + 1}: {len(batch_docs)} docs adicionados")
    
    logger.info(f"Total de documentos na collection: {collection.count()}")


def query_collection(
    query_text: str,
    n_results: int = 5,
    collection_name: str = CHROMA_COLLECTION_NAME
) -> Dict[str, Any]:
    """
    Realiza busca semântica na collection.
    
    Args:
        query_text: Texto da consulta
        n_results: Número de resultados a retornar
        collection_name: Nome da collection
        
    Returns:
        Dict com resultados da busca
    """
    collection = get_or_create_collection(collection_name)
    
    # Usa embedding function com prefixo de query para modelos E5
    query_embedding_fn = SentenceTransformerEmbeddingFunction(is_query=True)
    query_embedding = query_embedding_fn([query_text])
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results


def delete_collection(collection_name: str = CHROMA_COLLECTION_NAME) -> bool:
    """
    Remove uma collection do ChromaDB.
    
    Args:
        collection_name: Nome da collection a remover
        
    Returns:
        bool: True se removida com sucesso
    """
    try:
        client = get_chroma_client()
        client.delete_collection(collection_name)
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


if __name__ == "__main__":
    # Teste simples
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
