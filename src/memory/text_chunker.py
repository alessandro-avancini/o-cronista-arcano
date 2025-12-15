"""
Text Chunker - Divisão de Texto usando LangChain
=================================================

Divide textos em chunks menores usando RecursiveCharacterTextSplitter
do LangChain, que respeita limites naturais de parágrafos e frases.
"""

from dataclasses import dataclass
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Adiciona path do projeto
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


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
    
    Args:
        text: Texto a ser dividido
        chunk_size: Tamanho máximo de cada chunk
        chunk_overlap: Sobreposição entre chunks
        
    Returns:
        Lista de TextChunk
    """
    if not text or not text.strip():
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
        length_function=len
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
    
    chunks = split_text(sample_text, chunk_size=100, chunk_overlap=20)
    
    print(f"Texto original: {len(sample_text)} caracteres")
    print(f"Chunks gerados: {len(chunks)}\n")
    
    for chunk in chunks:
        print(f"[Chunk {chunk.index}] ({len(chunk.content)} chars)")
        print(f"  {chunk.content[:80]}...")
        print()
