import os
from google import genai
import logging
from dotenv import load_dotenv
import config.helper as config
import pathlib

load_dotenv()
logging.basicConfig(level=logging.INFO)


def transcribe_audio_gemini(audio_file_path: pathlib.Path,
                            model = 'gemini-2.5-flash') -> str:
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
            logging.info(f"Transcription completed successfully")
            return response.text
        else:
            logging.error("Transcription did not contain any text. Check the audio file and try again.")
            return ""

    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}")
        return ""

def process_episode_transcription(video_id: str):
    """
    Processes the transcription of an episode audio file and saves the result.

    Args:
        audio_file_path (str): The path to the audio file to be transcribed.
        output_file_path (str): The path where the transcription will be saved.
        model (str): The Gemini model to use for transcription.
    """


    chunks_dir = config.get_episode_chunks_dir(video_id)
    transcripts_dir = config.get_episode_transcripts_dir(video_id)
    final_transcript_path = config.get_final_transcript_path(video_id)

    chunk_files = sorted(chunks_dir.glob("*.mp3"))

    if not chunk_files:
        logging.error(f"No audio chunks found in directory: {chunks_dir}")
        return

    logging.info(f"Processing {len(chunk_files)} audio chunks for transcription...")

    individual_transcripts = []
    for chunk_file in chunk_files:

        transcript_filename = f"{chunk_file.stem}_transcript.txt"
        transcript_path = transcripts_dir / transcript_filename
        individual_transcripts.append(transcript_path)

        if transcript_path.exists():
            logging.info(f"Transcript already exists for chunk: {chunk_file.name}, skipping transcription.")
            continue
        transcript_text = transcribe_audio_gemini(chunk_file)

        if transcript_text:
            try:
                transcript_path.write_text(transcript_text, encoding='utf-8')
            except Exception as e:
                logging.error(f"Error saving transcript for chunk {chunk_file.name}: {e}")

    logging.info("Combining individual transcripts into final transcript...")
    full_text = ""

    for transcript_path in individual_transcripts:
        if transcript_path.exists():
            try:
                chunk_text = transcript_path.read_text(encoding='utf-8')
                full_text += chunk_text + "\n\n"
            except Exception as e:
                logging.error(f"Error reading transcript file {transcript_path.name}: {e}")
        else:
            logging.warning(f"Transcript file not found: {transcript_path.name}")

    try:
        final_transcript_path.write_text(full_text, encoding='utf-8')
        print(f"Final transcription saved at: {final_transcript_path}")
    except Exception as e:
        print(f"Error when saving full transcription: {e}")


if __name__ == "__main__":
    # Example usage
    config.setup_directories()
    episode_id = "RPG Ninja S01E01 - O Pauzinho da Magia"
    process_episode_transcription(episode_id)

