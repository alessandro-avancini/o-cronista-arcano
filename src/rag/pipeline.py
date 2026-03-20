import logging
from typing import Iterator, Optional

import ollama

from config.settings import RAG_TOP_K_DEFAULT
from src.memory.search import semantic_search, search_by_video, SearchResult
from src.rag.models import RAGResult, RAGSource

logger = logging.getLogger(__name__)

DEFAULT_RAG_MODEL = "gemma3:4b"

RAG_PROMPT_TEMPLATE = """
Você é o Cronista Arcano, um assistente especializado em sessões de RPG de mesa.
Use o contexto abaixo para responder à pergunta do usuário.

CONTEXTO DAS TRANSCRIÇÕES:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def _format_context(results: list[SearchResult]) -> str:
    if not results:
        return "Nenhum contexto relevante encontrado."
    parts = [
        f"[Trecho {i} - {r.video_id}]\n{r.content}"
        for i, r in enumerate(results, 1)
    ]
    return "\n\n---\n\n".join(parts)


def _search_results_to_sources(results: list[SearchResult]) -> list[RAGSource]:
    return [
        RAGSource(
            video_id=r.video_id,
            content=r.content,
            chunk_index=r.chunk_index,
            relevance_score=r.relevance_score,
        )
        for r in results
    ]


def rag_query(
    question: str,
    video_id: Optional[str] = None,
    top_k: int = RAG_TOP_K_DEFAULT,
    model: str = DEFAULT_RAG_MODEL,
) -> RAGResult:
    """
    Executa o pipeline RAG: busca semântica + prompt + LLM.
    Retorna resposta e fontes (trechos usados).
    """
    if not question.strip():
        return RAGResult(answer="Pergunta vazia.", sources=[])

    logger.info("Buscando contexto para: '%s'", question)
    if video_id:
        results = search_by_video(question, video_id=video_id, top_k=top_k)
    else:
        results = semantic_search(question, top_k=top_k)

    context = _format_context(results)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    logger.info("Gerando resposta com %s...", model)
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    sources = _search_results_to_sources(results)
    return RAGResult(answer=answer, sources=sources)


def _sources_to_dicts(sources: list[RAGSource]) -> list[dict]:
    return [{"video_id": s.video_id, "chunk_index": s.chunk_index, "relevance_score": s.relevance_score} for s in sources]


def rag_query_stream(
    question: str,
    video_id: Optional[str] = None,
    top_k: int = RAG_TOP_K_DEFAULT,
    model: str = DEFAULT_RAG_MODEL,
) -> Iterator[dict]:
    """
    Pipeline RAG em modo streaming: busca semântica + Ollama com stream=True.
    Produz eventos para SSE: sources, token (por chunk), done.
    """
    if not question.strip():
        yield {"type": "done"}
        return

    logger.info("Buscando contexto para: '%s'", question)
    if video_id:
        results = search_by_video(question, video_id=video_id, top_k=top_k)
    else:
        results = semantic_search(question, top_k=top_k)

    sources = _search_results_to_sources(results)
    yield {"type": "sources", "sources": _sources_to_dicts(sources)}

    context = _format_context(results)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    logger.info("Gerando resposta com %s (stream)...", model)

    stream = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], stream=True)
    for chunk in stream:
        part = (chunk.get("message") or {}).get("content") or ""
        if part:
            yield {"type": "token", "content": part}
    yield {"type": "done"}
