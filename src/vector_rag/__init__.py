"""RAG (Retrieval Augmented Generation) package."""
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from vector_rag.config import Config


def get_project_root() -> Path:
    """Get the root directory of the project.

    Returns:
        Path: Path object pointing to the project root directory
    """
    # Start from this file's directory and look upwards for the root marker
    current = Path(__file__).parent.parent
    while True:
        # Look for a marker file/directory that indicates the project root
        if (current / "extractor").exists() and (current / "extractor" / "pyproject.toml").exists():
            return current
        if current.parent == current:
            # We've reached the filesystem root
            raise FileNotFoundError("Could not determine project root directory")
        current = current.parent


filename = os.getenv('DOTENV_FILE', '.env')
dotenv_path = find_dotenv(filename)
load_dotenv(dotenv_path)

env_file = Path(dotenv_path).joinpath(filename)
config = Config(env_file=filename)
