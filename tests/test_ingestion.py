import pytest
import os
from src.extraction.ingestion import download_audio


@pytest.fixture
def short_video_url():
    """URL from a short YouTube video for quick testing."""
    return "https://www.youtube.com/watch?v=kJ2uKnjioeQ"


@pytest.fixture
def bad_video_url():
    """An invalid YouTube video URL."""
    return "https://www.youtube.com/watch?v=videoquenaoexiste"


# --- Test Functions ---

def test_download_audio_success(short_video_url, tmp_path):
    """
    Tests the successful download of audio from a valid YouTube URL.
    """
    print(f"Saving test files at: {tmp_path}")

    # Chama nossa função, mas direciona a saída para o diretório temporário
    audio_file = download_audio(short_video_url, output_dir=str(tmp_path))

    # 1. Verifica se a função retornou um caminho
    assert audio_file is not None

    # 2. Verifica se o caminho é uma string
    assert isinstance(audio_file, str)

    # 3. Verifica se o arquivo realmente existe no disco
    assert os.path.exists(audio_file)

    # 4. Verifica se a extensão está correta
    assert audio_file.endswith(".mp3")

    # 5. Verifica se o arquivo não está vazio (tem mais de 1KB)
    assert os.path.getsize(audio_file) > 1024

    # 6. Verifica se o arquivo foi salvo no diretório temporário esperado
    assert str(tmp_path) in audio_file


def test_download_audio_failure_bad_url(bad_video_url, tmp_path):
    """
    Tests the failure scenario with an invalid URL.
    We expect the function to handle the error and return None.
    """

    audio_file = download_audio(bad_video_url, output_dir=str(tmp_path))

    # O único 'assert' necessário: a função deve retornar None
    assert audio_file is None


def test_download_audio_failure_no_url(tmp_path):
    """
    Failure scenario when no URL is provided.
    We expect the function to handle the error and return None.
    """

    audio_file_empty = download_audio("", output_dir=str(tmp_path))
    assert audio_file_empty is None

    audio_file_none = download_audio(None, output_dir=str(tmp_path))
    assert audio_file_none is None