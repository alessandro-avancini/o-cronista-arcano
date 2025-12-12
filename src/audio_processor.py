import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydub import AudioSegment
import math
from config.settings import DEFAULT_AUDIO_CHUNK_LENGTH_MS
import config.helper as config

def split_audio_into_chunks(audio_file_path: str,
                            output_dir: str,
                            chunk_length_ms: int = DEFAULT_AUDIO_CHUNK_LENGTH_MS) -> list[str]:
    """
    Divides an audio file into smaller chunks of fixed size.

    Arguments:
        audio_file_path (str): File path to the input audio file.
        output_dir (str): Directory where the chunks will be saved.
        chunk_length_ms (int): Duration of each chunk in milliseconds.

    Retorn:
        list[str]: List containing the file paths of the created chunks.
    """
    print(f"Starting audion splitting: {audio_file_path}")
    #os.makedirs(output_dir, exist_ok=True)

    try:
        audio = AudioSegment.from_file(audio_file_path)
        print(f"Audio loaded successfully. Total duration: {len(audio) / 1000:.2f} seconds.")
    except Exception as e:
        print(f"Error loading audio file {audio_file_path}: {e}")
        return []

    total_length_ms = len(audio)
    total_chunks = math.ceil(total_length_ms / chunk_length_ms)
    print(f"Dividing into {total_chunks} chunks of ~{chunk_length_ms / 1000 / 60:.0f} minutes...")

    chunk_paths = []

    for i in range(total_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, total_length_ms)
        chunk = audio[start_ms:end_ms]
        chunk_filename = f"chunk_{i:04d}.mp3"
        chunk_path = os.path.join(output_dir, chunk_filename)
        try:
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)
            print(f"  -> Exported: {chunk_path}")
        except Exception as e:
            print(f"Error exporting chunk {chunk_path}: {e}")

    print(f"Splitting concluded sucessfully! {len(chunk_paths)} chunks saved at {output_dir}")
    return chunk_paths


if __name__ == "__main__":
    video_id = 'RPG Ensino Magico Ep. 02 - Os Pilares do Heroísmo'
    INPUT_AUDIO = config.get_audio_dir() / f"{video_id}.mp3"
    OUTPUT_FOLDER = config.get_episode_chunks_dir(video_id)

    if not os.path.exists(INPUT_AUDIO):
        print(f"Example entry file not found: {INPUT_AUDIO}")
        print("Please, download an audio from YouTube and place it in the 'downloads' folder.")
    else:
        created_chunks = split_audio_into_chunks(INPUT_AUDIO, OUTPUT_FOLDER)

        if created_chunks:
            print("\nNext step: Send these files to transcription!")
            print(created_chunks)