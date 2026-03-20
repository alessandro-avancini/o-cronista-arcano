"""
Chunk Summary - Resumo denso de chunks via Ollama (modelo local) para enriquecimento de metadados.
"""

import logging

import ollama

from config.settings import AGENT_MODEL

logger = logging.getLogger(__name__)

_CHUNK_SUMMARY_PROMPT = """
Resuma o trecho abaixo em 1 ou 2 frases densas, em português.
Foque em: quem participou, ação principal e local/contexto.
Responda apenas com o resumo, sem introdução.

Trecho:
"""


def generate_chunk_summary(text: str, *, max_fallback_chars: int = 200) -> str:
    """
    Gera resumo denso de um chunk via Ollama (granite4:3b-h).

    Em falha (Ollama indisponível, exceção, resposta vazia) retorna o texto truncado
    em max_fallback_chars para não quebrar o pipeline.
    """
    if not (text or "").strip():
        return ""

    try:
        prompt = f"{_CHUNK_SUMMARY_PROMPT}{text}"
        response = ollama.chat(
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = (response.get("message", {}).get("content") or "").strip()
        if not summary:
            return _fallback_summary(text, max_fallback_chars)
        return summary
    except Exception as e:
        logger.warning("Falha ao sumarizar chunk: %s; usando fallback", e)
        return _fallback_summary(text, max_fallback_chars)


def _fallback_summary(text: str, max_chars: int) -> str:
    """Retorna início do texto truncado quando a LLM falha."""
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"
