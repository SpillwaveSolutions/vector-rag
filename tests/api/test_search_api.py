import numpy as np
import pytest

from vector_rag.chunking import LineChunker
from vector_rag.config import Config
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.db.dimension_utils import ensure_vector_dimension
from vector_rag.embeddings import MockEmbedder
from vector_rag.model import File

config = Config()
EMBEDDINGS_DIM = 384
