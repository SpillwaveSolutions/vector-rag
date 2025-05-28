"""Tests for BM25 and hybrid search functionality."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from vector_rag.model import Chunk, ChunkResult, ChunkResults, File, Project
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.db.db_model import DbBase, ProjectDB, FileDB, ChunkDB
from vector_rag.embeddings.base import Embedder
from vector_rag.config import Config


class MockEmbedder(Embedder):
    """Mock embedder for tests."""

    def embed_texts(self, chunks):
        # Return deterministic embeddings based on content length
        return [np.array([0.1, 0.2, 0.3, len(chunk.content) / 100], dtype=">f4") for chunk in chunks]

    def get_dimension(self):
        return 4


@pytest.fixture
def db_handler(tmp_path):
    """Create a DBFileHandler with test database."""
    db_path = tmp_path / "test.db"
    config = Config()
    config.DB_URL = f"sqlite:///{db_path}"
    config.EMBEDDINGS_DIM = 4
    
    handler = DBFileHandler(config=config, embedder=MockEmbedder())
    
    # Apply the migration to add tsvector support
    with handler.engine.connect() as conn:
        # SQLite doesn't support tsvector, so we'll simulate it
        conn.execute(text("""
            ALTER TABLE chunks 
            ADD COLUMN content_tsv TEXT
        """))
        conn.commit()
    
    return handler


@pytest.fixture
def sample_project(db_handler):
    """Create a sample project with test data."""
    with db_handler.session_scope() as session:
        project = ProjectDB(name="Test Project", description="Test project for search")
        session.add(project)
        session.flush()
        
        # Add files with different content
        file1 = FileDB(
            project_id=project.id,
            filename="technical_doc.txt",
            file_path="/test/technical_doc.txt",
            crc="abc123",
            file_size=1000
        )
        file2 = FileDB(
            project_id=project.id,
            filename="general_doc.txt",
            file_path="/test/general_doc.txt",
            crc="def456",
            file_size=2000
        )
        session.add_all([file1, file2])
        session.flush()
        
        # Add chunks with specific content for testing
        chunks = [
            ChunkDB(
                file_id=file1.id,
                content="BM25 is a ranking function used for information retrieval systems.",
                embedding=[0.1, 0.2, 0.3, 0.4],
                chunk_index=0,
                chunk_metadata={"type": "technical"},
                content_tsv="BM25 rank function use information retrieval system"  # Simulated tsvector
            ),
            ChunkDB(
                file_id=file1.id,
                content="The BM25 algorithm considers term frequency and document length.",
                embedding=[0.2, 0.3, 0.4, 0.5],
                chunk_index=1,
                chunk_metadata={"type": "technical"},
                content_tsv="BM25 algorithm consider term frequency document length"
            ),
            ChunkDB(
                file_id=file2.id,
                content="Machine learning models can improve search relevance significantly.",
                embedding=[0.3, 0.4, 0.5, 0.6],
                chunk_index=0,
                chunk_metadata={"type": "general"},
                content_tsv="machine learn model improve search relevance significantly"
            ),
            ChunkDB(
                file_id=file2.id,
                content="Vector embeddings capture semantic meaning of text content.",
                embedding=[0.4, 0.5, 0.6, 0.7],
                chunk_index=1,
                chunk_metadata={"type": "general"},
                content_tsv="vector embed capture semantic mean text content"
            ),
        ]
        session.add_all(chunks)
        session.commit()
        
        return project.id


class TestBM25Search:
    """Test BM25 search functionality."""

    @patch('vector_rag.db.db_file_handler.func.plainto_tsquery')
    @patch('vector_rag.db.db_file_handler.func.ts_rank_cd')
    def test_search_chunks_by_bm25_basic(self, mock_ts_rank, mock_tsquery, db_handler, sample_project):
        """Test basic BM25 search."""
        # Mock the PostgreSQL functions
        mock_tsquery.return_value = "BM25"
        mock_ts_rank.return_value = 0.8
        
        # Since we're using SQLite for tests, we need to mock the actual search
        with patch.object(db_handler, 'session_scope') as mock_session_scope:
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__.return_value = mock_session
            
            # Mock the query results
            mock_chunk = MagicMock()
            mock_chunk.content = "BM25 is a ranking function"
            mock_chunk.chunk_index = 0
            mock_chunk.chunk_metadata = {"type": "technical"}
            
            mock_session.execute.return_value.all.return_value = [(mock_chunk, 0.8)]
            mock_session.execute.return_value.scalar.return_value = 1  # Total count
            
            results = db_handler.search_chunks_by_bm25(
                project_id=sample_project,
                query_text="BM25",
                page=1,
                page_size=10
            )
            
            assert len(results.results) == 1
            assert results.results[0].score == 0.8
            assert "BM25" in results.results[0].chunk.content
            assert results.total_count == 1

    def test_search_chunks_by_bm25_with_filters(self, db_handler, sample_project):
        """Test BM25 search with metadata filters."""
        with patch.object(db_handler, 'session_scope') as mock_session_scope:
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__.return_value = mock_session
            
            # Mock empty results
            mock_session.execute.return_value.all.return_value = []
            mock_session.execute.return_value.scalar.return_value = 0
            
            results = db_handler.search_chunks_by_bm25(
                project_id=sample_project,
                query_text="BM25",
                metadata_filter={"type": "technical"},
                rank_threshold=0.5
            )
            
            assert len(results.results) == 0
            assert results.total_count == 0


class TestHybridSearch:
    """Test hybrid search functionality."""

    @patch('vector_rag.db.db_file_handler.func.plainto_tsquery')
    @patch('vector_rag.db.db_file_handler.func.ts_rank_cd')
    def test_search_chunks_hybrid_balanced(self, mock_ts_rank, mock_tsquery, db_handler, sample_project):
        """Test hybrid search with balanced weights."""
        # Mock the PostgreSQL functions
        mock_tsquery.return_value = "BM25"
        mock_ts_rank.return_value = 0.6
        
        with patch.object(db_handler, 'session_scope') as mock_session_scope:
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__.return_value = mock_session
            
            # Mock the query results with both scores
            mock_chunk = MagicMock()
            mock_chunk.content = "BM25 is a ranking function"
            mock_chunk.chunk_index = 0
            mock_chunk.chunk_metadata = {"type": "technical"}
            
            # Return chunk, vector_score, bm25_score, hybrid_score
            mock_session.execute.return_value.all.return_value = [
                (mock_chunk, 0.8, 0.6, 0.7)  # (0.8 + 0.6) / 2 = 0.7
            ]
            mock_session.execute.return_value.scalar.return_value = 1
            
            results = db_handler.search_chunks_hybrid(
                project_id=sample_project,
                query_text="BM25",
                vector_weight=0.5,
                bm25_weight=0.5
            )
            
            assert len(results.results) == 1
            assert results.results[0].score == 0.7
            assert "_scores" in results.results[0].chunk.metadata
            assert results.results[0].chunk.metadata["_scores"]["vector"] == 0.8
            assert results.results[0].chunk.metadata["_scores"]["bm25"] == 0.6
            assert results.results[0].chunk.metadata["_scores"]["hybrid"] == 0.7

    def test_search_chunks_hybrid_vector_only(self, db_handler, sample_project):
        """Test hybrid search with vector weight only."""
        with patch.object(db_handler, 'session_scope') as mock_session_scope:
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__.return_value = mock_session
            
            mock_chunk = MagicMock()
            mock_chunk.content = "Vector search result"
            mock_chunk.chunk_index = 0
            mock_chunk.chunk_metadata = {}
            
            # When bm25_weight=0, only vector score matters
            mock_session.execute.return_value.all.return_value = [
                (mock_chunk, 0.9, 0.0, 0.9)
            ]
            mock_session.execute.return_value.scalar.return_value = 1
            
            results = db_handler.search_chunks_hybrid(
                project_id=sample_project,
                query_text="vector",
                vector_weight=1.0,
                bm25_weight=0.0
            )
            
            assert len(results.results) == 1
            assert results.results[0].score == 0.9

    def test_search_chunks_hybrid_bm25_only(self, db_handler, sample_project):
        """Test hybrid search with BM25 weight only."""
        with patch.object(db_handler, 'session_scope') as mock_session_scope:
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__.return_value = mock_session
            
            mock_chunk = MagicMock()
            mock_chunk.content = "BM25 search result"
            mock_chunk.chunk_index = 0
            mock_chunk.chunk_metadata = {}
            
            # When vector_weight=0, only BM25 score matters
            mock_session.execute.return_value.all.return_value = [
                (mock_chunk, 0.0, 0.85, 0.85)
            ]
            mock_session.execute.return_value.scalar.return_value = 1
            
            results = db_handler.search_chunks_hybrid(
                project_id=sample_project,
                query_text="BM25",
                vector_weight=0.0,
                bm25_weight=1.0
            )
            
            assert len(results.results) == 1
            assert results.results[0].score == 0.85

    def test_search_chunks_hybrid_invalid_weights(self, db_handler, sample_project):
        """Test hybrid search with invalid weights."""
        with pytest.raises(ValueError, match="At least one weight must be non-zero"):
            db_handler.search_chunks_hybrid(
                project_id=sample_project,
                query_text="test",
                vector_weight=0.0,
                bm25_weight=0.0
            )


class TestAPIIntegration:
    """Test API integration for BM25 and hybrid search."""

    def test_api_search_bm25(self):
        """Test BM25 search through API."""
        from vector_rag.api import VectorRAGAPI
        
        with patch('vector_rag.api.DBFileHandler') as mock_handler_class:
            mock_handler = MagicMock()
            mock_handler_class.return_value = mock_handler
            
            mock_results = ChunkResults(
                results=[ChunkResult(score=0.8, chunk=Chunk(target_size=1, content="BM25 result", index=0))],
                total_count=1,
                page=1,
                page_size=10
            )
            mock_handler.search_chunks_by_bm25.return_value = mock_results
            
            api = VectorRAGAPI()
            results = api.search_bm25(
                project_id=1,
                query_text="BM25"
            )
            
            assert len(results.results) == 1
            assert results.results[0].score == 0.8
            mock_handler.search_chunks_by_bm25.assert_called_once()

    def test_api_search_hybrid(self):
        """Test hybrid search through API."""
        from vector_rag.api import VectorRAGAPI
        
        with patch('vector_rag.api.DBFileHandler') as mock_handler_class:
            mock_handler = MagicMock()
            mock_handler_class.return_value = mock_handler
            
            mock_results = ChunkResults(
                results=[ChunkResult(score=0.75, chunk=Chunk(target_size=1, content="Hybrid result", index=0))],
                total_count=1,
                page=1,
                page_size=10
            )
            mock_handler.search_chunks_hybrid.return_value = mock_results
            
            api = VectorRAGAPI()
            results = api.search_hybrid(
                project_id=1,
                query_text="test",
                vector_weight=0.6,
                bm25_weight=0.4
            )
            
            assert len(results.results) == 1
            assert results.results[0].score == 0.75
            mock_handler.search_chunks_hybrid.assert_called_once_with(
                project_id=1,
                query_text="test",
                page=1,
                page_size=10,
                vector_weight=0.6,
                bm25_weight=0.4,
                similarity_threshold=0.0,
                rank_threshold=0.0,
                file_id=None,
                metadata_filter=None
            )