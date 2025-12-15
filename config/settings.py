MODEL_NAME = 'gemini-2.5-flash-lite'
MASTER_PROMPT = 'src/prompts/master_prompt.txt'
DEFAULT_AUDIO_CHUNK_LENGTH_MS = 20 * 60 * 1000 # Minutes * Seconds * Milliseconds -> 20 minutes

# --- Vectorial Memory Settings ---
CHUNK_SIZE = 500  # Tamanho máximo de cada chunk em caracteres (reduzido para melhor granularidade)
CHUNK_OVERLAP = 100  # Sobreposição entre chunks para manter contexto
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Modelo otimizado para retrieval multilíngue
CHROMA_COLLECTION_NAME = "cronista_arcano_transcripts"
