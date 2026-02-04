import json
import logging
import os

from dotenv import load_dotenv
from google import genai

import config.helper as config
from config.settings import MODEL_NAME
from src.extraction.io_utils import read_text_file, save_text_to_file

load_dotenv()
logger = logging.getLogger(__name__)


def clean_and_validate_json(raw_text: str) -> tuple[str, bool]:
    """
    Tenta limpar e validar o texto de resposta para garantir que é um JSON.

    Às vezes os LLMs podem envolver a resposta em markdown (```json ... ```).
    Retorna (conteúdo_limpo, is_valid).
    """

    logger.info("Cleaning and validating JSON")
    clean_text = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```")

    try:
        json.loads(clean_text)
        logger.info("JSON validation successful. The text is valid JSON.")
        return clean_text, True
    except json.JSONDecodeError as e:
        logger.error("JSON Decode Error: %s", e)
        logger.warning("Saving raw text for debugging.")
        return raw_text, False

def knowledge_extract(transcription_text: str, query_prompt: str) -> str | None:
    """
    Envia a transcrição e o prompt mestre para a API Gemini.

    Graças à grande janela de contexto do Gemini 1.5, podemos enviar
    a transcrição inteira de uma vez.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set.")
            return None

        client = genai.Client(api_key=api_key)

        logger.info("Sending transcription to Gemini model: %s", MODEL_NAME)
        logger.info("Length of transcription text: %d characters", len(transcription_text))

        final_prompt = f"{query_prompt}\n{transcription_text}"

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[final_prompt]
        )
        return response.text
    except Exception as e:
        logger.error("Error during API call: %s", e, exc_info=True)
        return None


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    video_id = "RPG Ensino Magico Ep. 02 - Os Pilares do Heroísmo"
    TRANSCRIPT_INPUT_FILE = config.get_final_transcript_path(video_id)
    KNOWLEDGE_OUTPUT_FILE = config.get_knowledge_extraction_path(video_id)

    transcription_text = read_text_file(TRANSCRIPT_INPUT_FILE)
    master_prompt = config.read_master_prompt()
    if not master_prompt:
        logger.error("Could not load master prompt.")
        sys.exit(1)
    json_response_text = knowledge_extract(transcription_text, master_prompt)
    if not json_response_text:
        logger.error("Knowledge extraction returned no content.")
        sys.exit(1)
    clean_json_response, _ = clean_and_validate_json(json_response_text)
    success = save_text_to_file(clean_json_response, KNOWLEDGE_OUTPUT_FILE)
    if success:
        logger.info("Pipeline executed successfully. Knowledge saved at: %s", KNOWLEDGE_OUTPUT_FILE)
    else:
        logger.error("Pipeline failed.")
        sys.exit(1)