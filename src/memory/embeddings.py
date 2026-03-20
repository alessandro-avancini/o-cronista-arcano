"""
Embeddings Module - Geração de Embeddings com Sentence Transformers
====================================================================

Módulo para geração de embeddings usando sentence-transformers,
otimizado para textos em português.
"""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import chromadb
import torch
from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Força CPU para embeddings (deixa GPU livre para Ollama)
# Descomente a linha abaixo para usar GPU nos embeddings (pode causar OOM com LLMs grandes)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"  # Recomendado quando usar Ollama na GPU

# Cache global do modelo para evitar recarregamentos
_model_cache: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Carrega e retorna o modelo de embeddings.
    Usa cache para evitar múltiplos carregamentos.
    
    Returns:
        SentenceTransformer: Modelo carregado
    """
    global _model_cache
    
    if _model_cache is None:
        logger.info(
            "Carregando modelo de embeddings: %s no %s",
            EMBEDDING_MODEL,
            DEVICE.upper(),
        )
        _model_cache = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        logger.info("Modelo carregado com sucesso.")
    
    return _model_cache


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Gera embeddings para uma lista de textos.
    
    Args:
        texts: Lista de strings para gerar embeddings
        
    Returns:
        Lista de vetores de embeddings
    """
    if not texts:
        return []
    
    model = get_embedding_model()
    
    logger.info(f"Gerando embeddings para {len(texts)} textos...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Converte para lista de listas (formato esperado pelo ChromaDB)
    return embeddings.tolist()


def embed_for_query(texts: List[str]) -> List[List[float]]:
    """
    Gera embeddings para textos de consulta (prefixo "query: " em modelos E5).
    Usa o modelo em cache. Use em query_collection em vez de instanciar outra
    embedding function.
    """
    if not texts:
        return []
    model = get_embedding_model()
    if "e5" in EMBEDDING_MODEL.lower():
        texts = ["query: " + t for t in texts]
    return model.encode(texts).tolist()


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Gera embeddings para documentos a indexar (prefixo "passage: " em modelos E5).
    Use na ingestão quando o vetor deve ser do resumo e o documento armazenado for outro texto.
    """
    if not texts:
        return []
    model = get_embedding_model()
    if "e5" in EMBEDDING_MODEL.lower():
        texts = ["passage: " + t for t in texts]
    return model.encode(texts).tolist()


class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Função de embedding compatível com ChromaDB.
    Usa sentence-transformers internamente.
    
    Para modelos E5 (intfloat/multilingual-e5-*), é necessário usar prefixos:
    - "query: " para textos de busca
    - "passage: " para documentos a serem indexados
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, is_query: bool = False):
        """
        Args:
            model_name: Nome do modelo sentence-transformers
            is_query: Se True, adiciona prefixo "query:" (para buscas).
                     Se False, adiciona prefixo "passage:" (para documentos).
        """
        self.model_name = model_name
        self.is_query = is_query
        self._is_e5_model = "e5" in model_name.lower()

    @property
    def model(self) -> SentenceTransformer:
        return get_embedding_model()
    
    def _add_prefix(self, texts: List[str]) -> List[str]:
        """Adiciona prefixo apropriado para modelos E5."""
        if not self._is_e5_model:
            return texts
        
        prefix = "query: " if self.is_query else "passage: "
        return [prefix + text for text in texts]
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Gera embeddings para os textos de entrada.
        
        Args:
            input: Lista de textos
            
        Returns:
            Lista de embeddings como listas de floats
        """
        prefixed_input = self._add_prefix(input)
        embeddings = self.model.encode(prefixed_input)
        return embeddings.tolist()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_texts = [
        "O mago lançou uma bola de fogo contra o dragão.",
        "Os aventureiros entraram na masmorra escura.",
        "A taverna estava cheia de viajantes misteriosos."
    ]
    
    print("Testando geração de embeddings...")
    embeddings = generate_embeddings(test_texts)
    
    print(f"\nTextos processados: {len(test_texts)}")
    print(f"Embeddings gerados: {len(embeddings)}")
    print(f"Dimensão dos embeddings: {len(embeddings[0])}")
    
    # Teste da função compatível com ChromaDB
    print("\nTestando embedding function para ChromaDB...")
    ef = SentenceTransformerEmbeddingFunction()
    chroma_embeddings = ef(test_texts)
    print(f"ChromaDB embeddings: {len(chroma_embeddings)} vetores")
