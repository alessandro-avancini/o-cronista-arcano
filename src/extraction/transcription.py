import logging
import os
import pathlib

from dotenv import load_dotenv
from google import genai

import config.helper as config
from config.settings import MODEL_NAME, TRANSCRIPT_CHUNK_SUFFIX

load_dotenv()
logger = logging.getLogger(__name__)


def transcribe_audio_gemini(
    audio_file_path: pathlib.Path,
    model: str = MODEL_NAME,
) -> str:
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
        logger.info("Transcribing audio file...")
        audio_file = client.files.upload(file=audio_file_path)

        prompt_path = config.get_transcription_prompt_path()
        prompt = prompt_path.read_text(encoding="utf-8")

        response = client.models.generate_content(
            model=model,
            contents=[prompt, audio_file]
        )
        if response.text:
            logger.info("Transcription completed successfully")
            return response.text
        logger.error("Transcription did not contain any text. Check the audio file and try again.")
        return ""

    except Exception as e:
        logger.error("An error occurred during transcription: %s", e)
        return ""

def process_episode_transcription(video_id: str) -> pathlib.Path | None:
    """
    Processes the transcription of an episode audio file and saves the result.

    Args:
        video_id: Identifier of the episode (used for paths).

    Returns:
        Path to the final transcript file, or None on failure.
    """
    chunks_dir = config.get_episode_chunks_dir(video_id)
    transcripts_dir = config.get_episode_transcripts_dir(video_id)
    final_transcript_path = config.get_final_transcript_path(video_id)

    chunk_files = sorted(chunks_dir.glob("*.mp3"))

    if not chunk_files:
        logger.error("No audio chunks found in directory: %s", chunks_dir)
        return None

    logger.info("Processing %d audio chunks for transcription...", len(chunk_files))

    individual_transcripts = []
    for chunk_file in chunk_files:

        transcript_filename = f"{chunk_file.stem}{TRANSCRIPT_CHUNK_SUFFIX}"
        transcript_path = transcripts_dir / transcript_filename
        individual_transcripts.append(transcript_path)

        if transcript_path.exists():
            logger.info("Transcript already exists for chunk: %s, skipping transcription.", chunk_file.name)
            continue
        transcript_text = transcribe_audio_gemini(chunk_file)

        if transcript_text:
            try:
                transcript_path.write_text(transcript_text, encoding='utf-8')
            except Exception as e:
                logger.error("Error saving transcript for chunk %s: %s", chunk_file.name, e)

    logger.info("Combining individual transcripts into final transcript...")
    full_text = ""

    for transcript_path in individual_transcripts:
        if transcript_path.exists():
            try:
                chunk_text = transcript_path.read_text(encoding='utf-8')
                full_text += chunk_text + "\n\n"
            except Exception as e:
                logger.error("Error reading transcript file %s: %s", transcript_path.name, e)
        else:
            logger.warning("Transcript file not found: %s", transcript_path.name)

    try:
        final_transcript_path.write_text(full_text, encoding="utf-8")
        logger.info("Final transcription saved at: %s", final_transcript_path)
        return final_transcript_path
    except Exception as e:
        logger.error("Error when saving full transcription: %s", e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config.setup_directories()
    episode_id = "RPG Ensino Magico Ep. 02 - Os Pilares do Heroísmo"
    process_episode_transcription(episode_id)

