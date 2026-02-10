from typing import Optional

from strands import tool
from strands.types.tools import ToolContext

from src.rag.models import RAGResult
from src.rag.pipeline import rag_query


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
    result = rag_query(question, video_id=vid)
    return _format_rag_response(result)
