"""
Agent runner - Cronista Arcano com Strands Agents.
Usa a tool RAG (consultar_transcricoes) quando o modelo decide.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from strands import Agent
from strands.models.ollama import OllamaModel

from config.settings import AGENT_MODEL
from src.rag import rag_tool

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "agent_system.txt"
OLLAMA_HOST = "http://localhost:11434"
ANSWER_FALLBACK = "Sem resposta."
DEFAULT_SYSTEM_PROMPT = (
    """
    Você é o Cronista Arcano, assistente de RPG. 
    Responda direto a saudações e perguntas gerais.
    Use a ferramenta de consulta às transcrições quando a pergunta for sobre o conteúdo das sessões/transcrições.
    """
)


@dataclass
class AgentResult:
    """Resultado da consulta ao agente. used_rag=True quando a tool RAG foi invocada."""

    answer: str
    used_rag: bool
    sources: list[dict[str, Any]]


def _load_system_prompt() -> str:
    """Carrega o system prompt do arquivo ou retorna o padrão."""
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT


def _looks_like_tool_call_json(text: str) -> bool:
    """True se o texto parece JSON de chamada de tool (não deve ser exibido como resposta)."""
    if not text or not text.strip().startswith("{"):
        return False
    t = text.strip()
    return ("consultar_transcricoes" in t or '"name"' in t) and ("parameters" in t or "arguments" in t)


def _extract_answer(strands_result: Any) -> str:
    """Extrai apenas blocos de texto da mensagem final; ignora tool-call JSON."""
    msg = getattr(strands_result, "message", None) if strands_result else None
    if msg is None:
        return ""
    content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
    blocks = content if isinstance(content, list) else [content] if content else []
    parts = []
    for block in blocks:
        text = None
        if isinstance(block, dict) and "text" in block:
            text = block["text"]
        elif hasattr(block, "text"):
            text = block.text
        if text and not _looks_like_tool_call_json(text):
            parts.append(text)
    return "\n".join(parts).strip() if parts else ""


def _is_rag_tool_name(name: str) -> bool:
    """True se o nome da tool indica consulta às transcrições."""
    lower = name.lower()
    return "consultar" in lower or "transcri" in lower or "rag" in lower


def _extract_used_rag_and_sources(strands_result: Any) -> tuple[bool, list[dict[str, Any]]]:
    """Indica se a tool RAG foi usada e extrai fontes a partir do estado do event loop."""
    state = getattr(strands_result, "state", None) if strands_result else None
    if state is None:
        return False, []
    event_loop_state = getattr(state, "event_loop_state", state)
    if not isinstance(event_loop_state, dict):
        return False, []
    messages = event_loop_state.get("messages", [])
    used_rag = False
    sources: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", []) if isinstance(msg, dict) else []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if "toolUse" in block:
                name = block.get("toolUse", {}).get("name", "")
                if _is_rag_tool_name(name):
                    used_rag = True
            if "toolResult" in block and used_rag:
                result_content = block.get("toolResult", {}).get("content", [])
                for c in result_content if isinstance(result_content, list) else []:
                    if isinstance(c, dict) and "text" in c and "Fontes:" in str(c.get("text", "")):
                        sources.append({"source": "transcricoes"})
                        break
    return used_rag, sources


def agent_query(
    pergunta: str,
    video_id: Optional[str] = None,
    model: str = AGENT_MODEL,
) -> AgentResult:
    """
    Executa o agente Cronista Arcano: responde direto ou invoca a tool RAG conforme a pergunta.
    """
    system_prompt = _load_system_prompt()
    if video_id:
        system_prompt += f'\n\nContexto atual: o usuário está no vídeo/sessão "{video_id}". Use esse video_id ao consultar transcrições quando fizer sentido.'

    ollama_model = OllamaModel(host=OLLAMA_HOST, model_id=model)
    agent = Agent(
        model=ollama_model,
        tools=[rag_tool],
        system_prompt=system_prompt,
    )
    invocation_state = {"video_id": video_id} if video_id is not None else {}

    strands_result = agent(pergunta.strip(), invocation_state=invocation_state)
    answer = _extract_answer(strands_result) or ANSWER_FALLBACK
    used_rag, sources = _extract_used_rag_and_sources(strands_result)

    return AgentResult(answer=answer, used_rag=used_rag, sources=sources)
