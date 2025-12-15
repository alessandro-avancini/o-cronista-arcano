"""
O Cronista Arcano - Pipeline Principal
=======================================

Script principal que orquestra todo o processamento de vídeos do YouTube:
1. Download do áudio (ingestion)
2. Divisão em chunks (audio_processor)
3. Transcrição via Gemini (transcription)
4. Extração de conhecimento e geração do JSON final (knowledge_extraction)

Uso:
    python src/main.py <URL_DO_VIDEO_YOUTUBE>
    
Exemplo:
    python src/main.py "https://www.youtube.com/watch?v=qdR4Hq-cj2U"
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Adiciona o diretório raiz ao path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Imports dos módulos do projeto
import config.helper as config
from config.settings import MASTER_PROMPT
from src.ingestion import download_audio
from src.audio_processor import split_audio_into_chunks
from src.transcription import process_episode_transcription
from src.knowledge_extraction import (
    read_text_file,
    save_text_to_file,
    knowledge_extract,
    clean_and_validate_json
)

# Carrega variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Resultado do processamento do pipeline."""
    success: bool
    video_id: str
    audio_path: Optional[Path] = None
    chunks_count: int = 0
    transcript_path: Optional[Path] = None
    json_path: Optional[Path] = None
    error_message: Optional[str] = None


def validate_environment() -> bool:
    """Valida se as variáveis de ambiente necessárias estão configuradas."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ GEMINI_API_KEY não está configurada no arquivo .env")
        return False
    logger.info("✅ Variáveis de ambiente validadas")
    return True


def extract_video_id_from_path(audio_path: str) -> str:
    """Extrai o ID/título do vídeo a partir do caminho do arquivo de áudio."""
    filename = os.path.basename(audio_path)
    # Remove a extensão .mp3
    video_id = os.path.splitext(filename)[0]
    return video_id


def step_1_download_audio(video_url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Etapa 1: Download do áudio do YouTube.
    
    Returns:
        tuple: (caminho_do_audio, video_id) ou (None, None) em caso de erro
    """
    logger.info("=" * 60)
    logger.info("📥 ETAPA 1: Download do Áudio do YouTube")
    logger.info("=" * 60)
    
    audio_dir = config.get_audio_dir()
    audio_path = download_audio(video_url, str(audio_dir))
    
    if audio_path and os.path.exists(audio_path):
        video_id = extract_video_id_from_path(audio_path)
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"✅ Áudio baixado com sucesso: {audio_path}")
        logger.info(f"   Tamanho: {file_size_mb:.2f} MB")
        logger.info(f"   Video ID: {video_id}")
        return audio_path, video_id
    else:
        logger.error("❌ Falha no download do áudio")
        return None, None


def step_2_split_audio(audio_path: str, video_id: str) -> list[str]:
    """
    Etapa 2: Divisão do áudio em chunks menores.
    
    Returns:
        list: Lista de caminhos dos chunks criados
    """
    logger.info("=" * 60)
    logger.info("✂️  ETAPA 2: Divisão do Áudio em Chunks")
    logger.info("=" * 60)
    
    chunks_dir = config.get_episode_chunks_dir(video_id)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_paths = split_audio_into_chunks(audio_path, str(chunks_dir))
    
    if chunk_paths:
        logger.info(f"✅ Áudio dividido em {len(chunk_paths)} chunks")
    else:
        logger.warning("⚠️ Nenhum chunk foi criado")
    
    return chunk_paths


def step_3_transcribe(video_id: str) -> Optional[Path]:
    """
    Etapa 3: Transcrição dos chunks de áudio.
    
    Returns:
        Path: Caminho do arquivo de transcrição final, ou None em caso de erro
    """
    logger.info("=" * 60)
    logger.info("📝 ETAPA 3: Transcrição do Áudio")
    logger.info("=" * 60)
    
    try:
        process_episode_transcription(video_id)
        transcript_path = config.get_final_transcript_path(video_id)
        
        if transcript_path.exists():
            file_size_kb = transcript_path.stat().st_size / 1024
            logger.info(f"✅ Transcrição concluída: {transcript_path}")
            logger.info(f"   Tamanho: {file_size_kb:.2f} KB")
            return transcript_path
        else:
            logger.error("❌ Arquivo de transcrição não foi criado")
            return None
    except Exception as e:
        logger.error(f"❌ Erro durante a transcrição: {e}")
        return None


def step_4_extract_knowledge(video_id: str) -> Optional[Path]:
    """
    Etapa 4: Extração de conhecimento e geração do JSON final.
    
    Returns:
        Path: Caminho do arquivo JSON gerado, ou None em caso de erro
    """
    logger.info("=" * 60)
    logger.info("🧠 ETAPA 4: Extração de Conhecimento")
    logger.info("=" * 60)
    
    transcript_path = config.get_final_transcript_path(video_id)
    json_output_path = config.get_knowledge_extraction_path(video_id)
    
    # Lê a transcrição
    transcription_text = read_text_file(str(transcript_path))
    if not transcription_text:
        logger.error("❌ Não foi possível ler o arquivo de transcrição")
        return None
    
    # Lê o prompt mestre
    master_prompt = read_text_file(MASTER_PROMPT)
    if not master_prompt:
        logger.error(f"❌ Não foi possível ler o prompt mestre: {MASTER_PROMPT}")
        return None
    
    # Realiza a extração de conhecimento
    logger.info("🔄 Enviando transcrição para o modelo Gemini...")
    json_response = knowledge_extract(transcription_text, master_prompt)
    
    if not json_response:
        logger.error("❌ Falha na extração de conhecimento")
        return None
    
    # Limpa e valida o JSON
    clean_json = clean_and_validate_json(json_response)
    
    # Salva o resultado
    success = save_text_to_file(clean_json, str(json_output_path))
    
    if success:
        file_size_kb = json_output_path.stat().st_size / 1024
        logger.info(f"✅ JSON gerado com sucesso: {json_output_path}")
        logger.info(f"   Tamanho: {file_size_kb:.2f} KB")
        return json_output_path
    else:
        logger.error("❌ Falha ao salvar o arquivo JSON")
        return None


def run_pipeline(video_url: str) -> ProcessingResult:
    """
    Executa o pipeline completo de processamento.
    
    Args:
        video_url: URL do vídeo do YouTube
        
    Returns:
        ProcessingResult: Resultado do processamento
    """
    logger.info("🚀 Iniciando O Cronista Arcano - Pipeline de Processamento")
    logger.info(f"   URL: {video_url}")
    logger.info("")
    
    # Validação inicial
    if not validate_environment():
        return ProcessingResult(
            success=False,
            video_id="",
            error_message="Variáveis de ambiente não configuradas"
        )
    
    # Configuração de diretórios
    config.setup_directories()
    
    # Etapa 1: Download
    audio_path, video_id = step_1_download_audio(video_url)
    if not audio_path or not video_id:
        return ProcessingResult(
            success=False,
            video_id="",
            error_message="Falha no download do áudio"
        )
    
    # Etapa 2: Divisão em chunks
    chunk_paths = step_2_split_audio(audio_path, video_id)
    if not chunk_paths:
        return ProcessingResult(
            success=False,
            video_id=video_id,
            audio_path=Path(audio_path),
            error_message="Falha na divisão do áudio em chunks"
        )
    
    # Etapa 3: Transcrição
    transcript_path = step_3_transcribe(video_id)
    if not transcript_path:
        return ProcessingResult(
            success=False,
            video_id=video_id,
            audio_path=Path(audio_path),
            chunks_count=len(chunk_paths),
            error_message="Falha na transcrição"
        )
    
    # Etapa 4: Extração de conhecimento
    json_path = step_4_extract_knowledge(video_id)
    if not json_path:
        return ProcessingResult(
            success=False,
            video_id=video_id,
            audio_path=Path(audio_path),
            chunks_count=len(chunk_paths),
            transcript_path=transcript_path,
            error_message="Falha na extração de conhecimento"
        )
    
    # Sucesso completo!
    return ProcessingResult(
        success=True,
        video_id=video_id,
        audio_path=Path(audio_path),
        chunks_count=len(chunk_paths),
        transcript_path=transcript_path,
        json_path=json_path
    )


def print_summary(result: ProcessingResult) -> None:
    """Imprime um resumo do processamento."""
    logger.info("")
    logger.info("=" * 60)
    
    if result.success:
        logger.info("🎉 PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    else:
        logger.info("❌ PROCESSAMENTO FALHOU")
    
    logger.info("=" * 60)
    logger.info("")
    logger.info("📊 RESUMO:")
    logger.info(f"   • Video ID: {result.video_id}")
    
    if result.audio_path:
        logger.info(f"   • Áudio: {result.audio_path}")
    
    if result.chunks_count > 0:
        logger.info(f"   • Chunks criados: {result.chunks_count}")
    
    if result.transcript_path:
        logger.info(f"   • Transcrição: {result.transcript_path}")
    
    if result.json_path:
        logger.info(f"   • JSON Final: {result.json_path}")
    
    if result.error_message:
        logger.info(f"   • Erro: {result.error_message}")
    
    logger.info("")


def main():
    """Função principal - ponto de entrada do script."""
    parser = argparse.ArgumentParser(
        description="O Cronista Arcano - Processa vídeos do YouTube e gera JSON com informações extraídas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python src/main.py "https://www.youtube.com/watch?v=abc123"
  python src/main.py "https://youtu.be/abc123"
        """
    )
    
    parser.add_argument(
        "video_url",
        type=str,
        help="URL do vídeo do YouTube para processar"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Ativa modo verboso com mais detalhes de log"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Executa o pipeline
    result = run_pipeline(args.video_url)
    
    # Imprime resumo
    print_summary(result)
    
    # Retorna código de saída apropriado
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
