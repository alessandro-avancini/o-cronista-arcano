import os
import logging
import json
from google import genai
from dotenv import load_dotenv
from config.settings import (
    MODEL_NAME,
    MASTER_PROMPT
)

load_dotenv()

def read_text_file(file_path: str) -> str | None:
    """Reads the content of a text file."""
    logging.info(f"Reading file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Error - file path not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error - error when reading file: {e}", exc_info=True)
        return None

def save_text_to_file(text: str, file_path: str) -> bool:
    """Saves text content to a file."""
    logging.info(f"Saving output to file: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info("File saved successfully.")
        return True
    except Exception as e:
        logging.error(f"Error - could not save file: {e}", exc_info=True)
        return False


def clean_and_validate_json(raw_text: str) -> str | None:
    """
    Tenta limpar e validar o texto de resposta para garantir que é um JSON.

    Às vezes, os LLMs podem envolver a resposta em "markdown" (```json ... ```).
    Esta função tenta limpar isso.
    """

    logging.info(f"Cleaning and validation JSON")
    clean_text = raw_text.strip().lstrip('```json').lstrip('```').rstrip('```')

    # 2. Validação: Tenta carregar o texto como um objeto JSON
    try:
        json.loads(clean_text)
        logging.info('JSON validation successful! The text is valid JSON.')
        return clean_text
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}")
        logging.warning("Saving raw text for debugging.")
        return raw_text

def knowledge_extract(transcription_text: str, query_prompt: str) -> str | None:
    """
    Envia a transcrição e o prompt mestre para a API Gemini.

    Graças à grande janela de contexto do Gemini 1.5, podemos enviar
    a transcrição inteira de uma vez.
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY environment variable not set.")
            return None

        client = genai.Client(api_key=api_key)

        logging.info(f"Sending transcription to Gemini model: {MODEL_NAME}")
        logging.info(f"Length of transcription text: {len(transcription_text)} characters")

        final_prompt = f"{query_prompt}\n{transcription_text}"

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[final_prompt]
        )
        return response.text
    except Exception as e:
        logging.error(f"Error during API call: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    TRANSCRIPT_INPUT_FILE = "long_transcription.txt"
    KNOWLEDGE_OUTPUT_FILE = "long_session_knowledge.json"

    transcription_text = read_text_file(TRANSCRIPT_INPUT_FILE)
    master_prompt = read_text_file(MASTER_PROMPT)
    json_response_text = knowledge_extract(transcription_text, master_prompt)
    clean_json_response = clean_and_validate_json(json_response_text)
    success = save_text_to_file(clean_json_response, KNOWLEDGE_OUTPUT_FILE)
    if success:
        logging.info(f"--- Pipeline executed sucessfully! Knowledge saved at: {KNOWLEDGE_OUTPUT_FILE} ---")
    else:
        logging.error("--- Pipeline failed. ---")