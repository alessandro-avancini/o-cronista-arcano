from typing import Any, Optional

from strands import tool
from strands.types.tools import ToolContext

from src.rag.models import RAGResult, RAGSource
from src.rag.pipeline import rag_query, rag_query_stream


def _resolve_video_id(
    video_id: Optional[str],
    tool_context: Optional[ToolContext],
) -> Optional[str]:
    """Obtém video_id do parâmetro ou do invocation_state do contexto."""
    if video_id is not None:
        return video_id
    if not tool_context or not hasattr(tool_context, "invocation_state"):
        return None
    state = tool_context.invocation_state
    if isinstance(state, dict):
        vid = state.get("video_id")
        if vid is not None:
            return vid
        inner = state.get("invocation_state") or {}
        if isinstance(inner, dict):
            return inner.get("video_id")
    return None


def _get_stream_queue(tool_context: Optional[ToolContext]) -> Any:
    """Obtém stream_queue do invocation_state se presente."""
    if not tool_context or not hasattr(tool_context, "invocation_state"):
        return None
    state = tool_context.invocation_state
    if isinstance(state, dict):
        return state.get("stream_queue")
    return None


def _format_rag_response(result: RAGResult) -> str:
    """Formata a resposta RAG e fontes em texto para o LLM."""
    parts = [f"Resposta baseada nas transcrições:\n\n{result.answer}"]
    if result.sources:
        parts.append("\nFontes: " + ", ".join(s.video_id for s in result.sources))
    return "\n".join(parts)


@tool(context=True)
def consultar_transcricoes(
    question: str,
    video_id: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> str:
    """Consulta as transcrições das sessões de RPG.
    Use quando a pergunta for sobre o conteúdo dos vídeos/sessões transcritas (personagens, eventos, cenas).
    Opcionalmente informe video_id para restringir à sessão indicada.

    Args:
        question: Pergunta em linguagem natural sobre o que aconteceu nas sessões.
        video_id: ID do vídeo/sessão para filtrar (opcional). Se não informado, pode ser usado o contexto da conversa.
    """
    vid = _resolve_video_id(video_id, tool_context)
    stream_queue = _get_stream_queue(tool_context)

    if stream_queue is not None:
        accumulated = []
        sources_list = []
        for event in rag_query_stream(question, video_id=vid):
            stream_queue.put(event)
            if event.get("type") == "token":
                accumulated.append(event.get("content", ""))
            elif event.get("type") == "sources":
                sources_list = event.get("sources") or []
        sources = [
            RAGSource(
                video_id=s.get("video_id", ""),
                content="",
                chunk_index=s.get("chunk_index", 0),
                relevance_score=float(s.get("relevance_score", 0)),
            )
            for s in sources_list
        ]
        result = RAGResult(answer="".join(accumulated), sources=sources)
        return _format_rag_response(result)

    result = rag_query(question, video_id=vid)
    return _format_rag_response(result)
