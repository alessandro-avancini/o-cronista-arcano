"""
Ingest - Pipeline de Ingestão de Dados
======================================

Script para ingerir transcrições no banco vetorial ChromaDB.
Lê apenas os arquivos full_transcript.txt de cada vídeo.
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import config.helper as config
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, FULL_TRANSCRIPT_FILENAME
from src.memory.text_chunker import TextChunk, split_text
from src.memory.vector_store import (
    add_documents,
    get_collection_info,
    get_or_create_collection
)

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Resultado da ingestão de um vídeo."""
    video_id: str
    source_file: Path
    chunks_created: int
    success: bool
    error_message: Optional[str] = None


def generate_chunk_id(video_id: str, chunk_index: int, content: str) -> str:
    """
    Gera um ID único para cada chunk baseado no conteúdo.
    
    Args:
        video_id: ID do vídeo
        chunk_index: Índice do chunk
        content: Conteúdo do chunk
        
    Returns:
        ID único no formato: video_id_chunk_index_hash
    """
    # Hash do conteúdo para garantir unicidade
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{video_id[:50]}_chunk_{chunk_index:04d}_{content_hash}"


def find_transcripts() -> List[Path]:
    """
    Encontra todos os arquivos full_transcript.txt no diretório de transcrições.
    
    Returns:
        Lista de caminhos para os arquivos de transcrição
    """
    transcripts_dir = config.TRANSCRIPTS_ROOT_DIR
    
    if not transcripts_dir.exists():
        logger.warning(f"Diretório de transcrições não encontrado: {transcripts_dir}")
        return []
    
    transcripts = list(transcripts_dir.glob(f"*/{FULL_TRANSCRIPT_FILENAME}"))
    logger.info(f"Encontrados {len(transcripts)} arquivos de transcrição")
    
    return transcripts


def extract_video_id_from_path(transcript_path: Path) -> str:
    """
    Extrai o video_id do caminho do arquivo de transcrição.
    
    Args:
        transcript_path: Caminho para o arquivo full_transcript.txt
        
    Returns:
        video_id extraído do nome da pasta pai
    """
    return transcript_path.parent.name


def _chunks_to_document_batches(
    chunks: List[TextChunk],
    video_id: str,
    transcript_path: Path,
) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Converte chunks em listas de documentos, metadados e IDs para o ChromaDB.
    """
    documents = []
    metadatas = []
    ids = []
    for chunk in chunks:
        documents.append(chunk.content)
        metadatas.append({
            "video_id": video_id,
            "source_file": str(transcript_path),
            "chunk_index": chunk.index,
        })
        ids.append(generate_chunk_id(video_id, chunk.index, chunk.content))
    return documents, metadatas, ids


def ingest_transcript(transcript_path: Path) -> IngestResult:
    """
    Ingere uma transcrição no banco vetorial.
    
    Args:
        transcript_path: Caminho para o arquivo de transcrição
        
    Returns:
        IngestResult com informações sobre a ingestão
    """
    video_id = extract_video_id_from_path(transcript_path)
    
    logger.info(f"Ingerindo transcrição: {video_id}")
    
    try:
        # Lê o conteúdo do arquivo
        content = transcript_path.read_text(encoding='utf-8')
        
        if not content.strip():
            return IngestResult(
                video_id=video_id,
                source_file=transcript_path,
                chunks_created=0,
                success=False,
                error_message="Arquivo de transcrição vazio"
            )
        
        # Divide em chunks
        chunks = split_text(
            content,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        if not chunks:
            return IngestResult(
                video_id=video_id,
                source_file=transcript_path,
                chunks_created=0,
                success=False,
                error_message="Nenhum chunk gerado"
            )

        documents, metadatas, ids = _chunks_to_document_batches(chunks, video_id, transcript_path)
        add_documents(documents, metadatas, ids)
        
        logger.info(f"  ✅ {len(chunks)} chunks ingeridos para: {video_id}")
        
        return IngestResult(
            video_id=video_id,
            source_file=transcript_path,
            chunks_created=len(chunks),
            success=True
        )
        
    except Exception as e:
        logger.error(f"  ❌ Erro ao ingerir {video_id}: {e}")
        return IngestResult(
            video_id=video_id,
            source_file=transcript_path,
            chunks_created=0,
            success=False,
            error_message=str(e)
        )


def ingest_all_transcripts(force: bool = False) -> List[IngestResult]:
    """
    Ingere todas as transcrições encontradas.

    Args:
        force: Se True, reingere mesmo quando o vídeo já estiver no banco.

    Returns:
        Lista de IngestResult para cada transcrição processada
    """
    logger.info("=" * 60)
    logger.info("Iniciando ingestão de transcrições")
    logger.info("=" * 60)

    transcripts = find_transcripts()

    if not transcripts:
        logger.warning("Nenhuma transcrição encontrada para ingerir")
        return []

    results = []
    for transcript_path in transcripts:
        video_id = extract_video_id_from_path(transcript_path)
        if not force and check_if_already_ingested(video_id):
            logger.info("Pulando %s (já ingerido; use --force para reingerir)", video_id)
            continue
        result = ingest_transcript(transcript_path)
        results.append(result)
    
    # Resumo
    successful = sum(1 for r in results if r.success)
    total_chunks = sum(r.chunks_created for r in results)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 RESUMO DA INGESTÃO")
    logger.info("=" * 60)
    logger.info(f"  Transcrições processadas: {len(results)}")
    logger.info(f"  Sucessos: {successful}")
    logger.info(f"  Falhas: {len(results) - successful}")
    logger.info(f"  Total de chunks: {total_chunks}")
    
    # Info da collection
    info = get_collection_info()
    logger.info(f"  Documentos no banco: {info['count']}")
    
    return results


def check_if_already_ingested(video_id: str) -> bool:
    """
    Verifica se um vídeo já foi ingerido no banco.
    
    Args:
        video_id: ID do vídeo
        
    Returns:
        True se já existe no banco
    """
    collection = get_or_create_collection()
    
    # Busca por metadados
    results = collection.get(
        where={"video_id": video_id},
        limit=1
    )
    
    return len(results['ids']) > 0


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Ingere transcrições no banco vetorial ChromaDB"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força reingestão mesmo se já existir"
    )

    args = parser.parse_args()

    results = ingest_all_transcripts(force=args.force)

    if any(not r.success for r in results):
        sys.exit(1)
    sys.exit(0)
