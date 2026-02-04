import logging
import math
import os
from pathlib import Path

from pydub import AudioSegment

import config.helper as config
from config.settings import (
    AUDIO_CHUNK_EXTENSION,
    AUDIO_CHUNK_PREFIX,
    DEFAULT_AUDIO_CHUNK_LENGTH_MS,
)

logger = logging.getLogger(__name__)


def split_audio_into_chunks(
    audio_file_path: str,
    output_dir: str,
    chunk_length_ms: int = DEFAULT_AUDIO_CHUNK_LENGTH_MS,
) -> list[str]:
    """
    Divides an audio file into smaller chunks of fixed size.

    Arguments:
        audio_file_path: File path to the input audio file.
        output_dir: Directory where the chunks will be saved.
        chunk_length_ms: Duration of each chunk in milliseconds.

    Returns:
        List of file paths of the created chunks.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting audio splitting: %s", audio_file_path)
    try:
        audio = AudioSegment.from_file(audio_file_path)
        logger.info(
            "Audio loaded successfully. Total duration: %.2f seconds.",
            len(audio) / 1000,
        )
    except Exception as e:
        logger.error("Error loading audio file %s: %s", audio_file_path, e)
        return []

    total_length_ms = len(audio)
    total_chunks = math.ceil(total_length_ms / chunk_length_ms)
    logger.info(
        "Dividing into %d chunks of ~%.0f minutes...",
        total_chunks,
        chunk_length_ms / 1000 / 60,
    )

    chunk_paths = []
    for i in range(total_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, total_length_ms)
        chunk = audio[start_ms:end_ms]
        chunk_filename = f"{AUDIO_CHUNK_PREFIX}{i:04d}{AUDIO_CHUNK_EXTENSION}"
        chunk_path = os.path.join(output_dir, chunk_filename)
        try:
            chunk.export(chunk_path, format="mp3")
            chunk_paths.append(chunk_path)
            logger.info("Exported: %s", chunk_path)
        except Exception as e:
            logger.error("Error exporting chunk %s: %s", chunk_path, e)

    logger.info(
        "Splitting concluded successfully. %d chunks saved at %s",
        len(chunk_paths),
        output_dir,
    )
    return chunk_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    video_id = "RPG Ensino Magico Ep. 02 - Os Pilares do Heroísmo"
    INPUT_AUDIO = config.get_audio_dir() / f"{video_id}.mp3"
    OUTPUT_FOLDER = config.get_episode_chunks_dir(video_id)

    if not os.path.exists(INPUT_AUDIO):
        logger.error("Example entry file not found: %s", INPUT_AUDIO)
        logger.info("Please download an audio from YouTube and place it in the downloads folder.")
    else:
        created_chunks = split_audio_into_chunks(str(INPUT_AUDIO), str(OUTPUT_FOLDER))
        if created_chunks:
            logger.info("Next step: Send these files to transcription!")
            logger.info("%s", created_chunks)
