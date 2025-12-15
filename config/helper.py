from pathlib import Path

# --- Directory Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
AUDIO_DIR = DOWNLOADS_DIR / "audio"
CHUNKS_ROOT_DIR = DOWNLOADS_DIR / "chunks"
TRANSCRIPTS_ROOT_DIR = DOWNLOADS_DIR / "transcripts"
JSON_ROOT_DIR = DOWNLOADS_DIR / "json"
VECTOR_DB_DIR = DOWNLOADS_DIR / "vectordb"

# --- Helper Functions ---

def get_audio_dir() -> Path:
    """
    Retorna o caminho para a PASTA que contém o áudio.
    Ex: .../downloads/audio/RPG Ninja S01E01.../"""
    path = AUDIO_DIR
    path.mkdir(parents=True, exist_ok=True) # Cria se não existir
    return path

def get_episode_chunks_dir(video_id: str) -> Path:
    """
    Retorna o caminho para a PASTA que contém os chunks de áudio.
    Ex: .../downloads/chunks/RPG Ninja S01E01.../
    """
    path = CHUNKS_ROOT_DIR / video_id
    # Não precisamos criar, pois o audio_processor.py já deve ter criado.
    return path

def get_episode_transcripts_dir(video_id: str) -> Path:
    """
    Retorna o caminho para a PASTA onde salvaremos os .txt individuais.
    Ex: .../transcripts/RPG Ninja S01E01.../
    """
    path = TRANSCRIPTS_ROOT_DIR / video_id
    path.mkdir(parents=True, exist_ok=True) # Cria se não existir
    return path

def get_final_transcript_path(video_id: str) -> Path:
    """
    Retorna o caminho para o ARQUIVO .txt final e combinado.
    Ex: .../transcripts/RPG Ninja S01E01.../full_transcript.txt
    """
    # Salva o .txt final dentro da pasta de transcrições do episódio
    return get_episode_transcripts_dir(video_id) / "full_transcript.txt"

def get_knowledge_extraction_path(video_id: str) -> Path:
    """
    Retorna o caminho para o ARQUIVO .json final e combinado.
    Ex: .../knowledge_extraction/RPG Ninja S01E01.../full_transcript.json
    """
    # Salva o .txt final dentro da pasta de transcrições do episódio
    JSON_ROOT_DIR.mkdir(parents=True, exist_ok=True) # Cria se não existir
    return JSON_ROOT_DIR / f"full_{video_id}.json"

def get_vector_db_dir() -> Path:
    """
    Retorna o caminho para o diretório do banco vetorial ChromaDB.
    Ex: .../downloads/vectordb/
    """
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    return VECTOR_DB_DIR

def setup_directories():
    """
    Cria a estrutura de diretórios base.
    """
    print("Verificando e configurando diretórios...")
    try:
        # Adiciona o novo diretório de transcrições
        for dir_path in [DOWNLOADS_DIR, CHUNKS_ROOT_DIR, TRANSCRIPTS_ROOT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print("Estrutura de diretórios pronta.")
    except OSError as e:
        print(f"Erro ao criar diretórios: {e}")
        raise

if __name__ == "__main__":
    print(PROJECT_ROOT)
    print(DOWNLOADS_DIR)
    print(CHUNKS_ROOT_DIR)
    print(TRANSCRIPTS_ROOT_DIR)