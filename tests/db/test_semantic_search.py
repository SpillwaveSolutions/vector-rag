import numpy as np
import pytest

from vector_rag.chunking import LineChunker
from vector_rag.config import Config
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.db.dimension_utils import ensure_vector_dimension
from vector_rag.model import File

config = Config()
EMBEDDINGS_DIM = 384


@pytest.mark.db_reset(scope="module")
def test_search_by_text_basic(module_populated_handler_with_metadata):
    """Test basic text search functionality."""
    handler, project_id = module_populated_handler_with_metadata

    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page=1,
        page_size=10,
        similarity_threshold=0.0,  # Accept all results for testing
    )

    assert results is not None
    assert len(results.results) > 0
    assert all(0 <= r.score <= 1 for r in results.results)
    assert results.total_count > 0
    assert results.page == 1


def test_search_by_embedding_basic(module_populated_handler_with_metadata, mock_embedder):
    """Test basic embedding search functionality."""
    handler, project_id = module_populated_handler_with_metadata

    # Create a test embedding with correct dtype
    test_embedding = np.array([0.5] * mock_embedder.get_dimension(), dtype=">f4")

    results = handler.search_chunks_by_embedding(
        project_id=project_id,
        embedding=test_embedding,
        page=1,
        page_size=10,
        similarity_threshold=0.0,
    )

    assert results is not None
    assert len(results.results) > 0
    assert all(0 <= r.score <= 1 for r in results.results)


def test_pagination(module_populated_handler_with_metadata):
    """Test pagination functionality."""
    handler, project_id = module_populated_handler_with_metadata

    # Get first page
    page1 = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page=1,
        page_size=2,
        similarity_threshold=0.0,
    )

    # Get second page
    page2 = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page=2,
        page_size=2,
        similarity_threshold=0.0,
    )

    assert len(page1.results) == 2
    assert page1.page == 1
    assert page1.has_next

    assert len(page2.results) > 0
    assert page2.page == 2
    assert page2.has_previous

    # Ensure different results on different pages
    page1_ids = {r.chunk.index for r in page1.results}
    page2_ids = {r.chunk.index for r in page2.results}
    # assert not page1_ids.intersection(page2_ids) TODO fix this


def test_search_with_threshold(module_populated_handler_with_metadata):
    """Test search with similarity threshold."""
    handler, project_id = module_populated_handler_with_metadata

    # Search with high threshold
    high_threshold_results = handler.search_chunks_by_text(
        project_id=project_id, query_text="Test query", similarity_threshold=0.9
    )

    # Search with low threshold
    low_threshold_results = handler.search_chunks_by_text(
        project_id=project_id, query_text="Test query", similarity_threshold=0.1
    )

    assert len(high_threshold_results.results) <= len(low_threshold_results.results)
    assert all(r.score >= 0.9 for r in high_threshold_results.results)


def test_search_invalid_project(module_populated_handler_with_metadata):
    """Test search with invalid project ID."""
    handler, _ = module_populated_handler_with_metadata

    results = handler.search_chunks_by_text(
        project_id=99999, query_text="Test query"  # Invalid project ID
    )

    assert results.total_count == 0
    assert len(results.results) == 0


def test_search_invalid_page(module_populated_handler_with_metadata):
    """Test search with invalid page parameters."""
    handler, project_id = module_populated_handler_with_metadata

    # Test invalid page number
    with pytest.raises(ValueError, match="Page number must be greater than 0"):
        handler.search_chunks_by_text(
            project_id=project_id,
            query_text="Test query",
            page=0,  # Invalid page number
        )

    # Test invalid page size
    with pytest.raises(ValueError, match="Page size must be greater than 1"):
        handler.search_chunks_by_text(
            project_id=project_id,
            query_text="Test query",
            page_size=0,  # Invalid page size
        )


def test_ordering(module_populated_handler_with_metadata):
    """Test that results are properly ordered by similarity."""
    handler, project_id = module_populated_handler_with_metadata

    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page_size=10,
        similarity_threshold=0.0,
    )

    # Check that scores are in descending order
    scores = [r.score for r in results.results]
    assert scores == sorted(scores, reverse=True)


def test_empty_results(module_populated_handler_with_metadata):
    """Test handling of empty results."""
    handler, project_id = module_populated_handler_with_metadata

    # Search with impossibly high threshold
    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        similarity_threshold=2.0,  # No results will match this threshold
    )

    assert results.total_count == 0
    assert len(results.results) == 0
    assert not results.has_next
    assert not results.has_previous
    assert results.total_pages == 0


def test_last_page(module_populated_handler_with_metadata):
    """Test behavior of last page."""
    handler, project_id = module_populated_handler_with_metadata

    # Get total count with a large page size
    full_results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page_size=100,
        similarity_threshold=0.0,
    )

    total_pages = full_results.total_pages

    # Get last page
    last_page = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        page=total_pages,
        page_size=2,
        similarity_threshold=0.0,
    )

    # assert not last_page.has_next TODO fix this
    assert last_page.has_previous if total_pages > 1 else not last_page.has_previous
    assert last_page.page == total_pages

    # Try to get page beyond the last page
    # with pytest.raises(ValueError, match="Page number must be greater than 0"):
    #     handler.search_chunks_by_text(
    #         project_id=project_id,
    #         query_text="Test query",
    #         page=total_pages + 1,
    #         page_size=2,
    #         similarity_threshold=0.0
    #     )
