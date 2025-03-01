import os
import time
from typing import List, Optional
from ..config import Config
from ..model import Chunk
from .base import Embedder
import os


class OpenAIEmbedder(Embedder):
    """OpenAI embedder implementation."""

    def __init__(
        self,
        config: Optional[Config] = None,
        batch_size: int = 16,
    ):
        """Initialize OpenAI embedder.

        Args:
           config: Configuration

        Raises:
            ValueError: If no API key is provided or if config is None
        """
        if config is None:
            config = Config()

        super().__init__(config.EMBEDDINGS_MODEL, config.EMBEDDINGS_DIM)
        os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key must be provided")
        import openai

        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.batch_size = batch_size
        self.dimension = config.EMBEDDINGS_DIM

    def get_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            int: Dimension of the embeddings
        """
        return self.dimension

    def embed_texts(self, chunks: List[Chunk]) -> List[List[float]]:
        """Embed a list of texts using OpenAI's API.

        Args:
            chunks: List of text chunks to embed

        Returns:
            List[List[float]]: List of embeddings, one per text
        """
        embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = [chunk.content for chunk in chunks[i : i + self.batch_size]]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            # Getting weird rate-limit timeouts here. Cheap sleep for now.
            time.sleep(0.02) # @Rick take a look here...can we handle rate limit response better?
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

    @classmethod
    def create(
        cls,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size=16,
    ):
        config = Config(
            OPENAI_API_KEY=api_key,
            EMBEDDINGS_MODEL=model_name or "text-embedding-3-small",
            EMBEDDINGS_DIM=dimension or 384,  # Match SentenceTransformers dimension
        )

        return OpenAIEmbedder(config, batch_size=batch_size)
