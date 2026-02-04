# --- Directory and file names (used by config.helper) ---
DOWNLOADS_DIR_NAME = "downloads"
AUDIO_DIR_NAME = "audio"
CHUNKS_DIR_NAME = "chunks"
TRANSCRIPTS_DIR_NAME = "transcripts"
JSON_DIR_NAME = "json"
VECTOR_DB_DIR_NAME = "vectordb"
FULL_TRANSCRIPT_FILENAME = "full_transcript.txt"
TRANSCRIPT_CHUNK_SUFFIX = "_transcript.txt"
AUDIO_CHUNK_EXTENSION = ".mp3"
AUDIO_CHUNK_PREFIX = "chunk_"
KNOWLEDGE_JSON_PREFIX = "full_"
KNOWLEDGE_JSON_SUFFIX = ".json"
MASTER_PROMPT_FILENAME = "master_prompt.txt"
TRANSCRIPTION_PROMPT_FILENAME = "transcription_prompt.txt"

# --- Gemini / LLM ---
MODEL_NAME = "gemini-2.5-flash-lite"

# --- Audio pipeline ---
DEFAULT_AUDIO_CHUNK_MINUTES = 20
DEFAULT_AUDIO_CHUNK_LENGTH_MS = (
    DEFAULT_AUDIO_CHUNK_MINUTES * 60 * 1000
)

# --- Vectorial memory ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
CHROMA_COLLECTION_NAME = "cronista_arcano_transcripts"
CHROMA_ADD_BATCH_SIZE = 100

# --- RAG chat ---
RAG_TOP_K_DEFAULT = 3

# --- Search ---
SEARCH_DEFAULT_TOP_K = 5
SEARCH_PREVIEW_MAX_CHARS = 500
