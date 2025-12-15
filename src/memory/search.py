"""
Search - Busca Semântica
========================

Módulo para realizar buscas semânticas no banco vetorial.
Permite consultas em linguagem natural sobre as transcrições.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Adiciona path do projeto
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.memory.vector_store import query_collection, get_collection_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Resultado individual de uma busca."""
    content: str
    video_id: str
    source_file: str
    chunk_index: int
    distance: float
    relevance_score: float  # 1 - distance (quanto maior, mais relevante)


def semantic_search(
    query: str,
    top_k: int = 5
) -> List[SearchResult]:
    """
    Realiza busca semântica no banco vetorial.
    
    Args:
        query: Pergunta ou texto para buscar
        top_k: Número de resultados a retornar
        
    Returns:
        Lista de SearchResult ordenados por relevância
    """
    if not query.strip():
        logger.warning("Query vazia")
        return []
    
    logger.info(f"Buscando: '{query}'")
    
    # Realiza a query no ChromaDB
    results = query_collection(query, n_results=top_k)
    
    if not results['documents'][0]:
        logger.info("Nenhum resultado encontrado")
        return []
    
    # Converte para SearchResult
    search_results = []
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for doc, meta, dist in zip(documents, metadatas, distances):
        result = SearchResult(
            content=doc,
            video_id=meta.get('video_id', 'unknown'),
            source_file=meta.get('source_file', 'unknown'),
            chunk_index=meta.get('chunk_index', -1),
            distance=dist,
            relevance_score=max(0, 1 - dist)  # Normaliza para 0-1
        )
        search_results.append(result)
    
    logger.info(f"Encontrados {len(search_results)} resultados")
    return search_results


def format_results(results: List[SearchResult], show_metadata: bool = True) -> str:
    """
    Formata os resultados para exibição.
    
    Args:
        results: Lista de SearchResult
        show_metadata: Se True, mostra metadados adicionais
        
    Returns:
        String formatada com os resultados
    """
    if not results:
        return "Nenhum resultado encontrado."
    
    output = []
    output.append(f"\n{'=' * 60}")
    output.append(f"🔍 {len(results)} RESULTADOS ENCONTRADOS")
    output.append(f"{'=' * 60}\n")
    
    for i, result in enumerate(results, 1):
        output.append(f"[{i}] Relevância: {result.relevance_score:.2%}")
        
        if show_metadata:
            output.append(f"    📹 Vídeo: {result.video_id}")
            output.append(f"    📄 Chunk: #{result.chunk_index}")
        
        output.append(f"\n    \"{result.content[:500]}{'...' if len(result.content) > 500 else ''}\"")
        output.append(f"\n{'-' * 60}\n")
    
    return "\n".join(output)


def interactive_search():
    """
    Interface interativa para realizar buscas.
    """
    print("\n" + "=" * 60)
    print("🧠 O CRONISTA ARCANO - Busca Semântica")
    print("=" * 60)
    
    # Mostra info do banco
    info = get_collection_info()
    print(f"\n📊 Banco vetorial: {info['count']} documentos indexados")
    print("\nDigite sua pergunta ou 'sair' para encerrar.\n")
    
    while True:
        try:
            query = input("🔍 Pergunta: ").strip()
            
            if query.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Até logo!")
                break
            
            if not query:
                continue
            
            results = semantic_search(query, top_k=5)
            formatted = format_results(results)
            print(formatted)
            
        except KeyboardInterrupt:
            print("\n\n👋 Busca encerrada.")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")


def search_by_video(
    query: str,
    video_id: str,
    top_k: int = 5
) -> List[SearchResult]:
    """
    Busca semântica filtrada por vídeo específico.
    
    Args:
        query: Pergunta ou texto para buscar
        video_id: ID do vídeo para filtrar
        top_k: Número de resultados
        
    Returns:
        Lista de SearchResult do vídeo especificado
    """
    # Por enquanto, faz busca normal e filtra
    # Futuramente pode usar where clause do ChromaDB
    all_results = semantic_search(query, top_k=top_k * 3)
    
    filtered = [r for r in all_results if r.video_id == video_id]
    return filtered[:top_k]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Busca semântica nas transcrições do Cronista Arcano"
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Pergunta para buscar. Se não fornecida, abre modo interativo."
    )
    parser.add_argument(
        "-n", "--num-results",
        type=int,
        default=5,
        help="Número de resultados (padrão: 5)"
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        help="Filtrar por vídeo específico"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Busca direta
        if args.video:
            results = search_by_video(args.query, args.video, args.num_results)
        else:
            results = semantic_search(args.query, args.num_results)
        
        print(format_results(results))
    else:
        # Modo interativo
        interactive_search()
