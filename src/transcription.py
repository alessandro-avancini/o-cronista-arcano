import os
from google import genai
import logging
from dotenv import load_dotenv
import time

load_dotenv()
logging.basicConfig(level=logging.INFO)


def transcribe_audio_gemini(audio_file_path: str,
                            output_file_path: str,
                            model = 'gemini-2.5-flash') -> bool:
    """
    Transcribes an audio file using Google Gemini's model.

    Args:
        audio_file_path (str): The path to the audio file to be transcribed.
        output_file_path (str): The path where the transcription will be saved.
        model (str): The Gemini model to use for transcription.
    Returns:
        str: The transcribed text.
    """

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        logging.info("Transcribing audio file...")
        audio_file = client.files.upload(file=audio_file_path)

        with open('prompts/transcription_prompt.txt', "r") as f:
            prompt = f.read()

        response = client.models.generate_content(
            model=model,
            contents=[prompt, audio_file]
        )
        if response.text:
            logging.info(f"Transcription completed successfully. Saved at: {output_file_path}")
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            logging.info("Transcription saved successfully.")
            return True
        else:
            logging.error("Transcription did not contain any text. Check the audio file and try again.")
            return False

    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    audio_path = "downloads/RPG Ninja S01E01 - O Pauzinho da Magia.mp3"
    output_path = "transcription.txt"
    transcribe_audio_gemini(audio_path, output_path)