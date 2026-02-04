"""
Config package: settings and path helpers.
Use: from config import get_audio_dir, CHUNK_SIZE, get_master_prompt_path, ...
"""

from config.helper import (
    AUDIO_DIR,
    CHUNKS_ROOT_DIR,
    DOWNLOADS_DIR,
    JSON_ROOT_DIR,
    PROJECT_ROOT,
    PROMPTS_DIR,
    TRANSCRIPTS_ROOT_DIR,
    VECTOR_DB_DIR,
    get_audio_dir,
    get_episode_chunks_dir,
    get_episode_transcripts_dir,
    get_final_transcript_path,
    get_knowledge_extraction_path,
    get_master_prompt_path,
    get_transcription_prompt_path,
    get_vector_db_dir,
    read_master_prompt,
    setup_directories,
)
from config.settings import (
    AUDIO_CHUNK_EXTENSION,
    AUDIO_CHUNK_PREFIX,
    CHROMA_ADD_BATCH_SIZE,
    CHROMA_COLLECTION_NAME,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_AUDIO_CHUNK_LENGTH_MS,
    EMBEDDING_MODEL,
    FULL_TRANSCRIPT_FILENAME,
    MODEL_NAME,
    RAG_TOP_K_DEFAULT,
    SEARCH_DEFAULT_TOP_K,
    SEARCH_PREVIEW_MAX_CHARS,
    TRANSCRIPT_CHUNK_SUFFIX,
)

__all__ = [
    # Helper paths
    "PROJECT_ROOT",
    "DOWNLOADS_DIR",
    "AUDIO_DIR",
    "CHUNKS_ROOT_DIR",
    "TRANSCRIPTS_ROOT_DIR",
    "JSON_ROOT_DIR",
    "VECTOR_DB_DIR",
    "PROMPTS_DIR",
    # Helper functions
    "get_audio_dir",
    "get_episode_chunks_dir",
    "get_episode_transcripts_dir",
    "get_final_transcript_path",
    "get_knowledge_extraction_path",
    "get_vector_db_dir",
    "get_master_prompt_path",
    "get_transcription_prompt_path",
    "read_master_prompt",
    "setup_directories",
    # Settings
    "AUDIO_CHUNK_EXTENSION",
    "AUDIO_CHUNK_PREFIX",
    "CHROMA_ADD_BATCH_SIZE",
    "CHROMA_COLLECTION_NAME",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "DEFAULT_AUDIO_CHUNK_LENGTH_MS",
    "EMBEDDING_MODEL",
    "FULL_TRANSCRIPT_FILENAME",
    "MODEL_NAME",
    "RAG_TOP_K_DEFAULT",
    "SEARCH_DEFAULT_TOP_K",
    "SEARCH_PREVIEW_MAX_CHARS",
    "TRANSCRIPT_CHUNK_SUFFIX",
]
