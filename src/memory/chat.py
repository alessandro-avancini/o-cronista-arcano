"""
RAG Chat - Chat com Modelo Local e Busca Vetorial
==================================================

Chat interativo usando Ollama como LLM local e ChromaDB 
para recuperar contexto relevante das transcrições.

Uso:
    python -m src.memory.chat
    python -m src.memory.chat --model llama3.2
"""

import logging
from typing import List, Optional
import ollama

from config.settings import RAG_TOP_K_DEFAULT
from src.memory.search import semantic_search, search_by_video, SearchResult

logger = logging.getLogger(__name__)

# Modelo padrão do Ollama
DEFAULT_MODEL = "gemma3:4b"

# Template do prompt com contexto
RAG_PROMPT_TEMPLATE = """Você é o Cronista Arcano, um assistente especializado em sessões de RPG de mesa.
Use o contexto abaixo para responder à pergunta do usuário.

CONTEXTO DAS TRANSCRIÇÕES:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def format_context(results: List[SearchResult]) -> str:
    """Formata os resultados da busca como contexto para o LLM."""
    if not results:
        return "Nenhum contexto relevante encontrado."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[Trecho {i} - {result.video_id}]\n{result.content}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def chat_with_context(
    question: str,
    model: str = DEFAULT_MODEL,
    top_k: int = RAG_TOP_K_DEFAULT,
    video_id: Optional[str] = None,
) -> str:
    """
    Responde uma pergunta usando RAG (Retrieval-Augmented Generation).
    Se video_id for informado, a busca é restrita ao contexto daquele vídeo.
    """
    logger.info(f"🔍 Buscando contexto para: '{question}'")
    if video_id:
        results = search_by_video(question, video_id=video_id, top_k=top_k)
    else:
        results = semantic_search(question, top_k=top_k)
    
    # 2. Formata o contexto
    context = format_context(results)
    
    # 3. Monta o prompt completo
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    
    # 4. Gera resposta com Ollama
    logger.info(f"🤖 Gerando resposta com {model}...")
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]


def interactive_chat(model: str = DEFAULT_MODEL):
    """Interface de chat interativo."""
    print("\n" + "=" * 60)
    print("🧙 O CRONISTA ARCANO - Chat RAG")
    print("=" * 60)
    print(f"📦 Modelo: {model}")
    print("💡 Digite 'sair' para encerrar\n")
    
    while True:
        try:
            question = input("👤 Você: ").strip()
            
            if question.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Até a próxima aventura!")
                break
            
            if not question:
                continue
            
            response = chat_with_context(question, model=model)
            print(f"\n🧙 Cronista: {response}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat encerrado.")
            break
        except ollama.ResponseError as e:
            print(f"\n❌ Erro do Ollama: {e}")
            print("💡 Verifique se o Ollama está rodando e o modelo está instalado.\n")
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Chat RAG com Ollama e busca vetorial"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Modelo Ollama (padrão: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Pergunta única (sem modo interativo)"
    )
    
    args = parser.parse_args()
    
    if args.question:
        # Modo pergunta única
        response = chat_with_context(args.question, model=args.model)
        print(f"\n🧙 Cronista: {response}\n")
    else:
        # Modo interativo
        interactive_chat(model=args.model)
