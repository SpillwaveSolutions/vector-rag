import logging
import numpy as np
import pytest
import sys

from vector_rag.config import Config
from vector_rag.model import File
from vector_rag.logging_config import configure_logging

# Configure logging at the start of the test
configure_logging(logging.DEBUG)

# Get loggers
logger = logging.getLogger(__name__)
db_logger = logging.getLogger("vector_rag.db.db_file_handler")

# Force output to be unbuffered
sys.stdout.reconfigure(line_buffering=True)

config = Config()
EMBEDDINGS_DIM = 384

@pytest.mark.db_reset(scope="module")
def test_search_by_text_with_metadata(module_populated_handler_with_metadata):
    """Test text search with metadata filtering."""
    handler, project_id = module_populated_handler_with_metadata
    
    logger.info("Starting test_search_by_text_with_metadata")
    logger.debug(f"Using project_id: {project_id}")

    # Search for technical documents
    logger.info("Searching for technical documents")
    tech_results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test query",
        metadata_filter={"category": "technical"}
    )
    logger.debug(f"Found {len(tech_results.results)} technical documents")

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


def test_nested_metadata_filtering(module_populated_handler_with_metadata):
    """Test search with nested metadata structure."""
    handler, project_id = module_populated_handler_with_metadata

    # Add a document with nested metadata
    nested_metadata = {
        "doc_metadata": {
            "filename": "West Dallas Light Industrial Portfolio - OM (5.23.24).pdf",
            "filetype": "application/pdf",
            "project_id": "6",
            "date_modified": "2025-03-05T14:45:41",
            "document_type": "OFFERING_MEMORANDUM",
            "source_document_id": "6"
        }
    }
    
    # Add test file with nested metadata
    content = "Test content for nested metadata"
    test_file = handler.add_file(
        project_id=project_id,
        file_model=File(
            name="West Dallas Light Industrial Portfolio - OM (5.23.24).pdf",
            crc=str(hash(content)),
            path="/test/nested.pdf",
            filename="nested.pdf",
            content=content,
            metadata=nested_metadata
        )
    )
    
    # Print for debugging
    print(f"DEBUG: Added file with nested metadata: {nested_metadata}")
    
    # Search using nested metadata filter
    results = handler.search_chunks_by_text(
        project_id=project_id,
        query_text="Test",
        metadata_filter={"doc_metadata": {"source_document_id": "6"}}
    )

    print(f"DEBUG: Found {len(results.results)} results with nested metadata filter")
    
    # Verify results
    assert len(results.results) > 0, "No results found with nested metadata filter"
    for result in results.results:
        print(f"DEBUG: Result metadata: {result.chunk.metadata}")
        assert "doc_metadata" in result.chunk.metadata, "doc_metadata not found in chunk metadata"
        assert "source_document_id" in result.chunk.metadata["doc_metadata"], "source_document_id not found in doc_metadata"
        assert result.chunk.metadata["doc_metadata"]["source_document_id"] == "6", "Incorrect source_document_id value"
        assert result.chunk.metadata["doc_metadata"]["document_type"] == "OFFERING_MEMORANDUM", "Incorrect document_type value"


def test_jsonb_containment_operator(module_populated_handler_with_metadata):
    """Test the JSONB containment operator directly."""
    handler, project_id = module_populated_handler_with_metadata
    
    # Add a file with complex nested metadata
    complex_metadata = {
        "properties": {
            "location": {
                "city": "Dallas",
                "state": "TX",
                "zip": "75201"
            },
            "details": {
                "size": "10000",
                "units": "5",
                "amenities": ["pool", "gym", "parking"]
            },
            "pricing": {
                "price": "2500000",
                "cap_rate": "5.2"
            }
        },
        "tags": ["commercial", "industrial", "investment"]
    }
    
    # Add test file with complex nested metadata
    content = "Test content for complex nested metadata"
    test_file = handler.add_file(
        project_id=project_id,
        file_model=File(
            name="complex_metadata.txt",
            crc=str(hash(content)),
            path="/test/complex.txt",
            filename="complex.txt",
            content=content,
            metadata=complex_metadata
        )
    )
    
    print(f"DEBUG: Added file with complex nested metadata: {complex_metadata}")
    
    # Test cases for nested metadata queries
    test_cases = [
        {
            "filter": {"properties": {"location": {"city": "Dallas"}}},
            "description": "Deep nested object (3 levels)"
        },
        {
            "filter": {"properties": {"details": {"amenities": ["pool"]}}},
            "description": "Nested array within object"
        },
        {
            "filter": {"tags": ["commercial"]},
            "description": "Array at top level"
        },
        {
            "filter": {"properties": {"pricing": {"price": "2500000"}}},
            "description": "Nested numeric value as string"
        }
    ]
    
    # Run test cases
    for case in test_cases:
        print(f"DEBUG: Testing {case['description']}: {case['filter']}")
        results = handler.search_chunks_by_text(
            project_id=project_id,
            query_text="Test",
            metadata_filter=case["filter"]
        )
        
        print(f"DEBUG: Found {len(results.results)} results for {case['description']}")
        assert len(results.results) > 0, f"No results found for {case['description']}"
        
        # Verify first result has the expected metadata
        if len(results.results) > 0:
            result_metadata = results.results[0].chunk.metadata
            print(f"DEBUG: Result metadata for {case['description']}: {result_metadata}")


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
