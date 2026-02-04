from pathlib import Path

from config.settings import (
    AUDIO_DIR_NAME,
    CHUNKS_DIR_NAME,
    DOWNLOADS_DIR_NAME,
    FULL_TRANSCRIPT_FILENAME,
    JSON_DIR_NAME,
    KNOWLEDGE_JSON_PREFIX,
    KNOWLEDGE_JSON_SUFFIX,
    MASTER_PROMPT_FILENAME,
    TRANSCRIPTS_DIR_NAME,
    TRANSCRIPTION_PROMPT_FILENAME,
    VECTOR_DB_DIR_NAME,
)

# --- Directory paths (built from PROJECT_ROOT + names from settings) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = PROJECT_ROOT / DOWNLOADS_DIR_NAME
AUDIO_DIR = DOWNLOADS_DIR / AUDIO_DIR_NAME
CHUNKS_ROOT_DIR = DOWNLOADS_DIR / CHUNKS_DIR_NAME
TRANSCRIPTS_ROOT_DIR = DOWNLOADS_DIR / TRANSCRIPTS_DIR_NAME
JSON_ROOT_DIR = DOWNLOADS_DIR / JSON_DIR_NAME
VECTOR_DB_DIR = DOWNLOADS_DIR / VECTOR_DB_DIR_NAME

PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts"


def _ensure_dir(path: Path) -> Path:
    """Cria o diretório se não existir e retorna o path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_audio_dir() -> Path:
    """
    Retorna o caminho para a PASTA que contém o áudio.
    Ex: .../downloads/audio/
    """
    return _ensure_dir(AUDIO_DIR)


def get_episode_chunks_dir(video_id: str) -> Path:
    """
    Retorna o caminho para a PASTA que contém os chunks de áudio.
    Ex: .../downloads/chunks/RPG Ninja S01E01.../
    """
    return CHUNKS_ROOT_DIR / video_id


def get_episode_transcripts_dir(video_id: str) -> Path:
    """
    Retorna o caminho para a PASTA onde salvaremos os .txt individuais.
    Ex: .../transcripts/RPG Ninja S01E01.../
    """
    return _ensure_dir(TRANSCRIPTS_ROOT_DIR / video_id)


def get_final_transcript_path(video_id: str) -> Path:
    """
    Retorna o caminho para o ARQUIVO .txt final e combinado.
    Ex: .../transcripts/RPG Ninja S01E01.../full_transcript.txt
    """
    return get_episode_transcripts_dir(video_id) / FULL_TRANSCRIPT_FILENAME


def get_knowledge_extraction_path(video_id: str) -> Path:
    """
    Retorna o caminho para o ARQUIVO .json de extração de conhecimento.
    Ex: .../downloads/json/full_{video_id}.json
    """
    _ensure_dir(JSON_ROOT_DIR)
    return JSON_ROOT_DIR / f"{KNOWLEDGE_JSON_PREFIX}{video_id}{KNOWLEDGE_JSON_SUFFIX}"


def get_vector_db_dir() -> Path:
    """
    Retorna o caminho para o diretório do banco vetorial ChromaDB.
    Ex: .../downloads/vectordb/
    """
    return _ensure_dir(VECTOR_DB_DIR)


def get_master_prompt_path() -> Path:
    """Retorna o path absoluto do arquivo do prompt mestre."""
    return PROMPTS_DIR / MASTER_PROMPT_FILENAME


def get_transcription_prompt_path() -> Path:
    """Retorna o path absoluto do arquivo do prompt de transcrição."""
    return PROMPTS_DIR / TRANSCRIPTION_PROMPT_FILENAME


def read_master_prompt() -> str | None:
    """Lê e retorna o conteúdo do prompt mestre. Retorna None em caso de erro."""
    try:
        return get_master_prompt_path().read_text(encoding="utf-8")
    except OSError:
        return None


def setup_directories() -> None:
    """Cria a estrutura de diretórios base."""
    print("Verificando e configurando diretórios...")
    try:
        for dir_path in [
            DOWNLOADS_DIR,
            AUDIO_DIR,
            CHUNKS_ROOT_DIR,
            TRANSCRIPTS_ROOT_DIR,
            JSON_ROOT_DIR,
            VECTOR_DB_DIR,
        ]:
            _ensure_dir(dir_path)
        print("Estrutura de diretórios pronta.")
    except OSError as e:
        print(f"Erro ao criar diretórios: {e}")
        raise


if __name__ == "__main__":
    print(PROJECT_ROOT)
    print(DOWNLOADS_DIR)
    print(CHUNKS_ROOT_DIR)
    print(TRANSCRIPTS_ROOT_DIR)
