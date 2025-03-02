import numpy as np
import pytest
from sqlalchemy import text

from vector_rag.chunking import LineChunker
from vector_rag.config import Config
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.db.dimension_utils import ensure_vector_dimension
from vector_rag.embeddings import MockEmbedder
from vector_rag.model import File

config = Config()
EMBEDDINGS_DIM = 384

@pytest.fixture
def mock_embedder():
    """Create a mock embedder with consistent embeddings for testing."""
    class TestEmbedder(MockEmbedder):
        def embed_texts(self, texts):
            # Return predictable embeddings with correct dtype and shape
            return [
                np.array(
                    [float(len(t.content)) / 100] * self.dimension, dtype=">f4"
                )
                for t in texts
            ]
    return TestEmbedder(dimension=384)


@pytest.fixture
def test_files_with_metadata():
    """Create test files with different metadata."""
    return [
        File(
            name=f"test{i}.txt",
            path=f"/path/to/test{i}.txt",
            crc=f"crc{i}",
            content=f"Test content {'x' * (i * 10)}\n" * (i + 1),
            meta_data={
                "type": "test",
                "category": "technical" if i % 2 == 0 else "non-technical",
                "source": "manual" if i < 3 else "auto-generated",
                "priority": str(i % 3 + 1)  # "1", "2", or "3"
            },
        )
        for i in range(6)
    ]


@pytest.fixture
def populated_handler_with_metadata(test_db, mock_embedder, test_files_with_metadata):
    """Create a handler with test data containing metadata."""
    # Ensure vector dimensions match
    ensure_vector_dimension(test_db, 384)

    handler = DBFileHandler.create(
        config.TEST_DB_NAME, mock_embedder, chunker=LineChunker.create(5, 0)
    )
    project = handler.create_project("Test Project with Metadata")

    # Add test files and verify metadata was stored
    for file in test_files_with_metadata:
        file_record = handler.add_file(project.id, file)
        assert file_record is not None
        
        # Verify chunks were created with correct metadata
        with handler.session_scope() as session:
            chunks = session.query(handler.Chunk).filter_by(file_id=file_record.id).all()
            assert len(chunks) > 0, "No chunks were created for the file"
            for chunk in chunks:
                assert chunk.chunk_metadata == file.meta_data, \
                    f"Chunk metadata mismatch: {chunk.chunk_metadata} != {file.meta_data}"

    return handler, project.id


def test_search_by_text_with_metadata(populated_handler_with_metadata):
    """Test text search with metadata filtering."""
    handler, project_id = populated_handler_with_metadata

    # Search for technical documents
    tech_results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        metadata_filter={"category": "technical"}
    )
    
    # Search for manual sources
    manual_results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        metadata_filter={"source": "manual"}
    )

    # Search with multiple metadata filters
    combined_results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        metadata_filter={
            "category": "technical",
            "source": "manual"
        }
    )

    assert len(tech_results.results) > 0
    assert len(manual_results.results) > 0
    assert len(combined_results.results) > 0
    
    # Verify meta_data filtering worked
    assert len(tech_results.results) == 3  # Should be 3 technical documents
    for result in tech_results.results:
        assert result.chunk.meta_data["category"] == "technical", \
            f"Expected technical but got {result.chunk.meta_data['category']}"
    
    assert len(manual_results.results) == 3  # Should be 3 manual sources
    for result in manual_results.results:
        assert result.chunk.meta_data["source"] == "manual", \
            f"Expected manual but got {result.chunk.meta_data['source']}"
    
    assert len(combined_results.results) == 2  # Should be 2 technical manual sources
    for result in combined_results.results:
        assert result.chunk.meta_data["category"] == "technical", \
            f"Expected technical but got {result.chunk.meta_data['category']}"
        assert result.chunk.meta_data["source"] == "manual", \
            f"Expected manual but got {result.chunk.meta_data['source']}"


def test_search_by_embedding_with_metadata(populated_handler_with_metadata, mock_embedder):
    """Test embedding search with metadata filtering."""
    handler, project_id = populated_handler_with_metadata

    # Create a test embedding with correct dtype
    test_embedding = np.array([0.5] * mock_embedder.get_dimension(), dtype=">f4")

    # Search for high priority documents
    high_priority_results = handler.search_chunks_by_embedding(
        project_id=project_id,
        embedding=test_embedding,
        metadata_filter={"priority": 3}
    )

    # Search for auto-generated technical documents
    auto_tech_results = handler.search_chunks_by_embedding(
        project_id=project_id,
        embedding=test_embedding,
        metadata_filter={
            "category": "technical",
            "source": "auto-generated"
        }
    )

    assert len(high_priority_results.results) > 0
    assert len(auto_tech_results.results) > 0
    
    # Verify metadata filtering worked
    for result in high_priority_results.results:
        assert result.chunk.meta_data["priority"] == "3"
    
    for result in auto_tech_results.results:
        assert result.chunk.meta_data["category"] == "technical"
        assert result.chunk.meta_data["source"] == "auto-generated"


def test_search_with_invalid_metadata(populated_handler_with_metadata):
    """Test search with invalid metadata filters."""
    handler, project_id = populated_handler_with_metadata

    # Search with non-existent metadata field
    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        metadata_filter={"non_existent_field": "value"}
    )
    
    assert len(results.results) == 0
    assert results.total_count == 0


def test_search_with_empty_metadata(populated_handler_with_metadata):
    """Test search with empty metadata filter."""
    handler, project_id = populated_handler_with_metadata

    # Search with empty metadata filter
    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        metadata_filter={}
    )
    
    # Should return all results without filtering
    unfiltered_results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query"
    )
    
    assert len(results.results) == len(unfiltered_results.results)
    assert results.total_count == unfiltered_results.total_count


def test_search_with_multiple_metadata_values(populated_handler_with_metadata):
    """Test search with multiple possible values for a metadata field."""
    handler, project_id = populated_handler_with_metadata

    # Search for documents with priority 1 or 2
    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        metadata_filter={"priority": [1, 2]}
    )
    
    assert len(results.results) > 0
    
    # Verify metadata filtering worked
    for result in results.results:
        assert result.chunk.meta_data["priority"] in ["1", "2"]

def test_metadata_filtering_consistency(populated_handler_with_metadata, mock_embedder):
    """Verify text and embedding searches return consistent results with metadata filters."""
    handler, project_id = populated_handler_with_metadata
    
    # Create test embedding
    test_embedding = np.array([0.5] * mock_embedder.get_dimension(), dtype=">f4")
    
    # Define test cases
    test_cases = [
        {"filter": {"category": "technical"}, "expected_count": 3},
        {"filter": {"source": "manual"}, "expected_count": 3},
        {"filter": {"priority": "2"}, "expected_count": 2},
        {"filter": {"category": "technical", "source": "manual"}, "expected_count": 2},
        {"filter": {"category": "non-technical", "priority": "3"}, "expected_count": 1},
    ]
    
    for case in test_cases:
        # Test text search
        text_results = handler.search_chunks_by_text(
            project_id=project_id,
            query_text="Test query",
            metadata_filter=case["filter"]
        )
        
        # Test embedding search
        embedding_results = handler.search_chunks_by_embedding(
            project_id=project_id,
            embedding=test_embedding,
            metadata_filter=case["filter"]
        )
        
        # Verify counts match expectations
        assert len(text_results.results) == case["expected_count"], \
            f"Text search failed for filter {case['filter']}"
        assert len(embedding_results.results) == case["expected_count"], \
            f"Embedding search failed for filter {case['filter']}"
        
        # Verify metadata matches in both result sets
        for result in text_results.results + embedding_results.results:
            for key, value in case["filter"].items():
                if isinstance(value, list):
                    assert result.chunk.meta_data[key] in value, \
                        f"Metadata mismatch in {key}: {result.chunk.meta_data[key]} not in {value}"
                else:
                    assert result.chunk.meta_data[key] == value, \
                        f"Metadata mismatch in {key}: {result.chunk.meta_data[key]} != {value}"
