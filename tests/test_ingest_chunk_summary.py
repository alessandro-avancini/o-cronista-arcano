"""
Testa a ingestão com resumos nos chunks: verifica que chunk_summary está nos metadados do ChromaDB.
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.memory.ingest import (
    _chunks_to_document_batches,
    find_transcripts,
    ingest_transcript,
)
from src.memory.text_chunker import TextChunk, split_text
from src.memory.vector_store import get_or_create_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_chunks_to_document_batches_includes_chunk_summary():
    """Metadados produzidos por _chunks_to_document_batches incluem chunk_summary."""
    chunks = [
        TextChunk(content="Fulano e Ciclano discutem na taverna.", index=0),
        TextChunk(content="O grupo decide ir ao castelo.", index=1),
    ]
    video_id = "test-video"
    transcript_path = Path("downloads/transcripts/test/full_transcript.txt")

    with patch("src.memory.ingest.generate_chunk_summary", side_effect=lambda t: f"Resumo: {t[:20]}…"), \
         patch("src.memory.ingest.embed_documents", return_value=[[0.1] * 1024, [0.2] * 1024]):
        documents, metadatas, ids, embeddings = _chunks_to_document_batches(chunks, video_id, transcript_path)

    assert len(metadatas) == 2
    assert len(documents) == 2
    assert len(ids) == 2
    assert len(embeddings) == 2
    for meta in metadatas:
        assert "chunk_summary" in meta
        assert meta["video_id"] == video_id
        assert meta["source_file"] == str(transcript_path)
        assert "chunk_index" in meta
    assert metadatas[0]["chunk_summary"].startswith("Resumo:")
    assert metadatas[1]["chunk_summary"].startswith("Resumo:")


@pytest.mark.slow
def test_ingest_one_transcript_and_verify_chunk_summary_in_metadata():
    """Ingere uma transcrição e verifica que os documentos no ChromaDB têm chunk_summary nos metadados.
    Requer Ollama com granite4:3b-h. Execute: pytest tests/test_ingest_chunk_summary.py -v -m slow"""
    transcripts = find_transcripts()
    assert transcripts, "Nenhuma transcrição encontrada em downloads/transcripts/*/full_transcript.txt"

    path = min(transcripts, key=lambda p: p.stat().st_size)
    logger.info("Ingerindo transcrição: %s", path)
    result = ingest_transcript(path)
    assert result.success, f"Ingestão falhou: {result.error_message}"
    assert result.chunks_created > 0, "Nenhum chunk foi criado"

    collection = get_or_create_collection()
    out = collection.get(limit=min(5, result.chunks_created), include=["metadatas"])
    metadatas = out.get("metadatas") or []
    assert metadatas, "Nenhum documento retornado do ChromaDB"

    for i, meta in enumerate(metadatas):
        assert "chunk_summary" in meta, f"Metadado do doc {i} não tem 'chunk_summary': {list(meta.keys())}"
        summary = meta.get("chunk_summary", "")
        assert isinstance(summary, str), "chunk_summary deve ser string"
        logger.info("Doc %d: chunk_summary (len=%d) = %s...", i, len(summary), (summary[:80] + "…") if len(summary) > 80 else summary)

    logger.info("OK: %d documentos com chunk_summary nos metadados", len(metadatas))


if __name__ == "__main__":
    test_chunks_to_document_batches_includes_chunk_summary()
    print("Teste unitário OK. Para teste de integração (Ollama): pytest tests/test_ingest_chunk_summary.py -v -k test_ingest_one")
