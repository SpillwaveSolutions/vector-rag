import pytest
from vector_rag.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder
from vector_rag.model import Chunk
from vector_rag.config import Config


def test_sentence_transformer_dimension(sentence_transformers_embedder):
    """Test that the embedder returns the correct dimension (384 for all-MiniLM-L12-v2)."""
    assert sentence_transformers_embedder.get_dimension() == 384

def test_embed_single_text(module_db_handler):
    """Test embedding a single text chunk."""
    chunk = Chunk(content="This is a test sentence.", meta_data={})
    embeddings = module_db_handler.embedder.embed_texts([chunk])
    
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384  # all-MiniLM-L12-v2 dimension

def test_embed_multiple_texts(module_db_handler):
    """Test embedding multiple text chunks."""
    chunks = [
        Chunk(content="First test sentence.", meta_data={}),
        Chunk(content="Second test sentence.", meta_data={}),
        Chunk(content="Third test sentence.", meta_data={})
    ]
    embeddings = module_db_handler.embedder.embed_texts(chunks)
    
    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)

def test_batch_processing(module_db_handler):
    """Test that batching works correctly."""
    chunks = [Chunk(content=f"Test sentence {i}.", meta_data={}) for i in range(20)]
    embeddings = module_db_handler.embedder.embed_texts(chunks)
    
    assert len(embeddings) == 20
    assert all(len(emb) == 384 for emb in embeddings)
