"""Embeddings package for handling text embeddings from various sources."""

import importlib.util

from .base import Embedder
from .mock_embedder import MockEmbedder
from .openai_embedder import OpenAIEmbedder

# Conditionally import SentenceTransformersEmbedder if dependencies are available
sentence_transformers_available = importlib.util.find_spec("sentence_transformers") is not None
torch_available = importlib.util.find_spec("torch") is not None

if sentence_transformers_available and torch_available:
    from .sentence_transformers_embedder import SentenceTransformersEmbedder
    __all__ = ["Embedder", "OpenAIEmbedder", "MockEmbedder", "SentenceTransformersEmbedder"]
else:
    __all__ = ["Embedder", "OpenAIEmbedder", "MockEmbedder"]
