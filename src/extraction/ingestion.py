import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yt_dlp
import logging
import config.helper as config 

logging.basicConfig(level=logging.INFO)

def download_audio(video_url: str, output_dir: str = "downloads/audio") -> str | None:
    """
    Downloads the audio from a YouTube video and saves it as an MP3 file.
    Args:
        video_url (str): The URL of the YouTube video.
        output_dir (str): The directory where the MP3 file will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'logger': logging.getLogger('yt_dlp'),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info("Downloading audio from YouTube video.")
            info_dict = ydl.extract_info(video_url, download=False)
            title = info_dict.get('title', 'video')            

            base_filename = ydl.prepare_filename(info_dict)
            mp3_filename = f'{os.path.splitext(base_filename)[0]}.mp3'
            error_code = ydl.download([video_url])

            if error_code != 0:
                logging.error("Failed to download the audio.")
                return None

            if os.path.exists(mp3_filename):
                logging.info(f"Audio downloaded successfully: {mp3_filename}")
                return mp3_filename
            else:
                logging.error("MP3 file was not created.")
                return None


    except yt_dlp.utils.DownloadError as de:
        logging.error(f"Erro de Download: {de}")
        return None
    except Exception as e:
        logging.error(f"An unexpected exception interrupted the program: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    test_url = "https://youtu.be/qdR4Hq-cj2U?si=Zl1AUxMuwCP3nkc7"
    audio_file = download_audio(test_url, )

    if audio_file:
        logging.info(f"\n[SUCESS] Download finished.")
        logging.info(f"File saved at: {audio_file}")
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        logging.info(f"Size: {file_size_mb:.2f} MB")
    else:
        logging.error("\n[FALHA] Audio not downloaded.")