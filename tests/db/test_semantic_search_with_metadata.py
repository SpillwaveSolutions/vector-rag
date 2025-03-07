import numpy as np
import pytest

from vector_rag.config import Config

config = Config()
EMBEDDINGS_DIM = 384

@pytest.mark.db_reset(scope="module")
def test_search_by_text_with_metadata(module_populated_handler_with_metadata):
    """Test text search with metadata filtering."""
    handler, project_id = module_populated_handler_with_metadata

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

    # Verify metadata filtering worked
    assert len(tech_results.results) == 3  # Should be 3 technical documents
    for result in tech_results.results:
        assert result.chunk.metadata["category"] == "technical", \
            f"Expected technical but got {result.chunk.metadata['category']}"

    assert len(manual_results.results) == 3  # Should be 3 manual sources
    for result in manual_results.results:
        assert result.chunk.metadata["source"] == "manual", \
            f"Expected manual but got {result.chunk.metadata['source']}"

    assert len(combined_results.results) == 2  # Should be 2 technical manual sources
    for result in combined_results.results:
        assert result.chunk.metadata["category"] == "technical", \
            f"Expected technical but got {result.chunk.metadata['category']}"
        assert result.chunk.metadata["source"] == "manual", \
            f"Expected manual but got {result.chunk.metadata['source']}"


def test_search_by_embedding_with_metadata(module_populated_handler_with_metadata, mock_embedder):
    """Test embedding search with metadata filtering."""
    handler, project_id = module_populated_handler_with_metadata

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
        assert result.chunk.metadata["priority"] == "3"

    for result in auto_tech_results.results:
        assert result.chunk.metadata["category"] == "technical"
        assert result.chunk.metadata["source"] == "auto-generated"


def test_search_with_invalid_metadata(module_populated_handler_with_metadata):
    """Test search with invalid metadata filters."""
    handler, project_id = module_populated_handler_with_metadata

    # Search with non-existent metadata field
    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        metadata_filter={"non_existent_field": "value"}
    )

    assert len(results.results) == 0
    assert results.total_count == 0


def test_search_with_empty_metadata(module_populated_handler_with_metadata):
    """Test search with empty metadata filter."""
    handler, project_id = module_populated_handler_with_metadata

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


def test_search_with_multiple_metadata_values(module_populated_handler_with_metadata):
    """Test search with multiple possible values for a metadata field."""
    handler, project_id = module_populated_handler_with_metadata

    # Search for documents with priority 1 or 2
    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        metadata_filter={"priority": [1, 2]}
    )

    assert len(results.results) > 0

    # Verify metadata filtering worked
    for result in results.results:
        assert result.chunk.metadata["priority"] in ["1", "2"]


def test_metadata_filtering_consistency(module_populated_handler_with_metadata, mock_embedder):
    """Verify text and embedding searches return consistent results with metadata filters."""
    handler, project_id = module_populated_handler_with_metadata

    # Create test embedding
    test_embedding = np.array([0.5] * mock_embedder.get_dimension(), dtype=">f4")

    # Define test cases
    test_cases = [
        {"filter": {"category": "technical"}, "expected_count": 3},
        {"filter": {"source": "manual"}, "expected_count": 3},
        {"filter": {"priority": "2"}, "expected_count": 2},
        {"filter": {"category": "technical", "source": "manual"}, "expected_count": 2},
        {"filter": {"category": "non-technical", "priority": "3"}, "expected_count": 2},
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
                    assert result.chunk.metadata[key] in value, \
                        f"Metadata mismatch in {key}: {result.chunk.metadata[key]} not in {value}"
                else:
                    assert result.chunk.metadata[key] == value, \
                        f"Metadata mismatch in {key}: {result.chunk.metadata[key]} != {value}"
