import logging
import os

import yt_dlp

logger = logging.getLogger(__name__)


def download_audio(video_url: str, output_dir: str = "downloads/audio") -> str | None:
    """
    Downloads the audio from a YouTube video and saves it as an MP3 file.

    Args:
        video_url: The URL of the YouTube video.
        output_dir: The directory where the MP3 file will be saved.

    Returns:
        Full path to the saved MP3 file, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": False,
        "logger": logging.getLogger("yt_dlp"),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("Downloading audio from YouTube video.")
            info_dict = ydl.extract_info(video_url, download=False)
            base_filename = ydl.prepare_filename(info_dict)
            name_without_ext = os.path.splitext(os.path.basename(base_filename))[0]
            mp3_path = os.path.join(output_dir, name_without_ext + ".mp3")

            error_code = ydl.download([video_url])

            if error_code != 0:
                logger.error("Failed to download the audio.")
                return None

            if os.path.exists(mp3_path):
                logger.info("Audio downloaded successfully: %s", mp3_path)
                return mp3_path
            logger.error("MP3 file was not created.")
            return None

    except yt_dlp.utils.DownloadError as de:
        logger.error("Erro de Download: %s", de)
        return None
    except Exception as e:
        logger.error("An unexpected exception interrupted the program: %s", e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_url = "https://youtu.be/qdR4Hq-cj2U?si=Zl1AUxMuwCP3nkc7"
    audio_file = download_audio(test_url)

    if audio_file:
        logger.info("[SUCESS] Download finished.")
        logger.info("File saved at: %s", audio_file)
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        logger.info("Size: %.2f MB", file_size_mb)
    else:
        logger.error("[FALHA] Audio not downloaded.")
