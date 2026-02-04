"""
I/O utilities for the extraction pipeline.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_text_file(file_path: str | Path) -> str | None:
    """Reads the content of a text file."""
    path = Path(file_path) if isinstance(file_path, str) else file_path
    logger.info("Reading file: %s", path)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("File path not found: %s", path)
        return None
    except Exception as e:
        logger.error("Error when reading file: %s", e, exc_info=True)
        return None


def save_text_to_file(text: str, file_path: str | Path) -> bool:
    """Saves text content to a file."""
    path = Path(file_path) if isinstance(file_path, str) else file_path
    logger.info("Saving output to file: %s", path)
    try:
        path.write_text(text, encoding="utf-8")
        logger.info("File saved successfully.")
        return True
    except Exception as e:
        logger.error("Could not save file: %s", e, exc_info=True)
        return False
