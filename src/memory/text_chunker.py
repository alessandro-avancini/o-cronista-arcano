"""
Text Chunker - Divisão de Texto usando LangChain
=================================================

Divide textos em chunks menores usando RecursiveCharacterTextSplitter
do LangChain. Tamanhos são medidos em tokens (tokenizer do modelo de embeddings).
"""

from dataclasses import dataclass
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

_tokenizer_cache: Optional[AutoTokenizer] = None


def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer_cache
    if _tokenizer_cache is None:
        _tokenizer_cache = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _tokenizer_cache


def _token_length(text: str) -> int:
    if not text:
        return 0
    try:
        return len(_get_tokenizer().encode(text, add_special_tokens=False))
    except Exception:
        return len(text)


@dataclass
class TextChunk:
    """Representa um chunk de texto com metadados."""
    content: str
    index: int


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[TextChunk]:
    """
    Divide texto em chunks usando LangChain.

    chunk_size e chunk_overlap são em tokens (medidos pelo tokenizer do EMBEDDING_MODEL).

    Args:
        text: Texto a ser dividido
        chunk_size: Tamanho máximo de cada chunk (em tokens)
        chunk_overlap: Sobreposição entre chunks (em tokens)

    Returns:
        Lista de TextChunk
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
        length_function=_token_length
    )

    chunks = splitter.split_text(text)

    return [
        TextChunk(content=chunk, index=i)
        for i, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    sample_text = """
    Este é um texto de exemplo para testar o chunker.

    Ele tem múltiplos parágrafos e sentenças. Cada parágrafo deve ser
    tratado como um ponto natural de divisão.

    O objetivo é garantir que não cortemos frases no meio, mantendo
    a coerência do texto em cada chunk.
    """

    chunks = split_text(sample_text, chunk_size=50, chunk_overlap=10)

    print(f"Texto original: {_token_length(sample_text)} tokens, {len(sample_text)} caracteres")
    print(f"Chunks gerados: {len(chunks)}\n")

    for chunk in chunks:
        n_tokens = _token_length(chunk.content)
        print(f"[Chunk {chunk.index}] ({n_tokens} tokens)")
        print(f"  {chunk.content[:80]}...")
        print()
