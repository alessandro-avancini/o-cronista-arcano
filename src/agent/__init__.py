"""
Agent package - Cronista Arcano agêntico.
Decide entre resposta direta e invocação da tool RAG.
"""

from src.agent.runner import AgentResult, agent_query

__all__ = ["agent_query", "AgentResult"]
