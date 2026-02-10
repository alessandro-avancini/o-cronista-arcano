"""
API FastAPI do Cronista Arcano
==============================
Endpoints: GET /videos, POST /processar, POST /perguntar
"""

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config.helper as config
from src.extraction.run_pipeline import run_pipeline
from src.memory.ingest import ingest_transcript
from src.memory.vector_store import list_video_ids
from src.agent.runner import agent_query

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(
    title="Cronista Arcano",
    description="API RAG para mestres de RPG",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _web_path(*parts: str) -> Path:
    return WEB_DIR.joinpath(*parts)


def _serve_static_file(relative_parts: tuple[str, ...], media_type: str, not_found_detail: str):
    """Serve um arquivo estático ou levanta 404."""
    path = _web_path(*relative_parts)
    if path.exists():
        return FileResponse(path, media_type=media_type)
    raise HTTPException(status_code=404, detail=not_found_detail)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    logger.info("Interface: pasta web em %s", WEB_DIR)
else:
    logger.warning("Pasta 'web' não encontrada em %s; arquivos estáticos não serão servidos.", WEB_DIR)


@app.on_event("startup")
def startup():
    if WEB_DIR.exists():
        logger.info("Cronista Arcano: interface em http://localhost:8000/ (web em %s)", WEB_DIR)


@app.get("/js/app.js")
def serve_js():
    """Serve o script da interface para garantir que sempre carregue."""
    return _serve_static_file(("js", "app.js"), "application/javascript", "app.js não encontrado")


@app.get("/css/style.css")
def serve_css():
    """Serve o CSS da interface."""
    return _serve_static_file(("css", "style.css"), "text/css", "style.css não encontrado")


class ProcessarBody(BaseModel):
    url: str

    class Config:
        json_schema_extra = {"example": {"url": "https://www.youtube.com/watch?v=..."}}


class PerguntarBody(BaseModel):
    video_id: str
    pergunta: str

    class Config:
        json_schema_extra = {"example": {"video_id": "Nome do Vídeo", "pergunta": "Quem era o vilão?"}}


@app.get("/videos")
def get_videos():
    """Retorna a lista de nomes/IDs dos vídeos já presentes no banco vetorial."""
    try:
        ids = list_video_ids()
        return {"videos": [{"id": vid, "nome": vid} for vid in ids]}
    except Exception as e:
        logger.exception("Erro ao listar vídeos")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/processar")
async def processar(body: ProcessarBody):
    """
    Recebe URL do YouTube, baixa, transcreve, extrai conhecimento e ingere no banco vetorial.
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL é obrigatória")

    config.setup_directories()
    try:
        result = await asyncio.to_thread(run_pipeline, url)
    except Exception as e:
        logger.exception("Erro no pipeline")
        raise HTTPException(status_code=500, detail=str(e))

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=result.error_message or "Processamento falhou",
        )

    transcript_path = config.get_final_transcript_path(result.video_id)
    if not transcript_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Transcrição não encontrada após o pipeline",
        )

    try:
        ingest_result = await asyncio.to_thread(ingest_transcript, transcript_path)
    except Exception as e:
        logger.exception("Erro na ingestão")
        raise HTTPException(status_code=500, detail=str(e))

    if not ingest_result.success:
        raise HTTPException(
            status_code=500,
            detail=ingest_result.error_message or "Falha na ingestão",
        )

    return {
        "success": True,
        "video_id": result.video_id,
        "chunks_ingeridos": ingest_result.chunks_created,
    }


@app.post("/perguntar")
async def perguntar(body: PerguntarBody):
    """Recebe ID do vídeo e pergunta; executa o agente (com tool RAG quando aplicável) e retorna a resposta."""
    video_id = body.video_id.strip() or None
    pergunta = body.pergunta.strip()
    if not pergunta:
        raise HTTPException(status_code=400, detail="Pergunta é obrigatória")

    try:
        result = await asyncio.to_thread(
            agent_query,
            pergunta,
            video_id=video_id,
        )
        return {"resposta": result.answer, "used_rag": result.used_rag, "sources": result.sources}
    except Exception as e:
        logger.exception("Erro ao perguntar")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """Serve a interface estática."""
    index = _web_path("index.html")
    if index.exists():
        return FileResponse(index)
    return {"message": "Cronista Arcano API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
