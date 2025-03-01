"""SentenceTransformers embedder implementation.
Downloads and caches the model locally for offline use, running on the local CPU or GPU.
"""

import torch
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from ..config import Config
from ..model import Chunk
from .base import Embedder

class SentenceTransformersEmbedder(Embedder):
    """SentenceTransformers embedder implementation using all-MiniLM-L12-v2."""
    
    def __init__(self, config: Optional[Config] = None, batch_size: int = 16):
        """
        Initialize the SentenceTransformers embedder.
        
        Args:
            config: Configuration object. If not provided, a default Config will be used.
            batch_size: Number of texts to encode in a batch.
        """
        if config is None:
            config = Config()
        model_name = "sentence-transformers/all-MiniLM-L12-v2"
        dimension = config.get_or_default("EMBEDDINGS_DIM", 384)
        super().__init__(model_name, dimension)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dimension = self.model.get_sentence_embedding_dimension()

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings.
        """
        return self.dimension

    def embed_texts(self, chunks: List[Chunk]) -> List[List[float]]:
        """Embed a list of texts using SentenceTransformers.

        Args:
            chunks: List of text chunks to embed.

        Returns:
            List[List[float]]: List of embeddings, one per text.
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return embeddings.tolist()

    @classmethod
    def create(
        cls,
        config = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: int = 16,
    ):
        if config is None:
            config = Config(
                SENTENCE_TRANSFORMERS_MODEL=model_name or "sentence-transformers/all-MiniLM-L12-v2",
                EMBEDDINGS_DIM=dimension or 384,
            )
        return SentenceTransformersEmbedder(config, batch_size=batch_size)
