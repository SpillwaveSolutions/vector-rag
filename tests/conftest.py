import os
import sys
import uuid
from pathlib import Path
import subprocess
import time
from random import random

from vector_rag.chunking import LineChunker
from vector_rag.db import ensure_vector_dimension, DBFileHandler
from vector_rag.model import File

# Add src directory to Python path
src_path = str(Path(__file__).parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from vector_rag.config import Config
import numpy as np
from vector_rag.embeddings import MockEmbedder
# Check if SentenceTransformersEmbedder is available
try:
    from vector_rag.embeddings import SentenceTransformersEmbedder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Load environment variables from .env file
env_path = Path('../../.env')
if env_path.exists():
    load_dotenv(env_path)

def get_config():
    global config
    config = Config(env_file=env_path)

def setup_test_env():
    """Set up test environment."""
    os.environ["POSTGRES_DB"] = os.getenv("TEST_DB_NAME", "vectordb_test")
    os.environ["POSTGRES_PASSWORD"] = "postgres"
    get_config()

def ensure_db_running():
    """Helper function to ensure DB container is running."""
    # Check if container is running
    proc = subprocess.run(
        ["docker", "compose", "ps", "-q", "db"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not proc.stdout:
        # Container not running, start it
        subprocess.run(["docker", "compose", "up", "-d", "db"], check=True)

        # Wait for the DB container to be ready
        for i in range(30):
            proc = subprocess.run(
                ["docker", "compose", "exec", "db", "pg_isready"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode == 0:
                break
            time.sleep(1)
        else:
            raise Exception("Database container did not become ready in time")


def perform_db_reset():
    """Helper function to reset test database state."""
    setup_test_env()
    ensure_db_running()
    # Connect to default postgres database
    default_engine = create_engine(config.DB_URL)

    try:
        # Drop and recreate test database
        with default_engine.connect() as conn:
            # Terminate connections to test database
            conn.execute(text(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity 
                WHERE datname = '{config.TEST_DB_NAME}' 
                AND pid <> pg_backend_pid()
            """))
            conn.execute(text("commit"))
            conn.execute(text(f"DROP DATABASE IF EXISTS {config.TEST_DB_NAME}"))
            conn.execute(text("commit"))
            conn.execute(text(f"CREATE DATABASE {config.TEST_DB_NAME}"))
            conn.execute(text("commit"))
            
        # Now connect to the NEW test database to create the extension
        test_engine = create_engine(config.TEST_DB_URL)
        with test_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("commit"))
    except Exception as e:
        pytest.fail(f"Failed to create vector extension: {str(e)}")


def pytest_addoption(parser):
    parser.addoption(
        "--embedding",
        action="store",
        default="sentence",
        choices=["openai", "sentence"],
        help="Choose embedding type: openai (1536 dimensions) or sentence (384 dimensions)"
    )
    parser.addoption(
        "--db-reset",
        action="store",
        default="module",
        choices=["module", "function"],
        help="Control database reset frequency: module (faster) or function (more isolated)"
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "db_reset(scope): control database reset frequency for test group"
    )

@pytest.fixture
def embedding_group(request):
    """Return the embedding group for the current test based on markers."""
    if request.node.get_closest_marker("openai"):
        return "openai"
    return "sentence"

@pytest.fixture(scope="module")
def mock_embedder(request):
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
    
    # Determine dimension based on markers
    if request.node.get_closest_marker("openai"):
        return TestEmbedder(dimension=1536)
    return TestEmbedder(dimension=384)

@pytest.fixture
def sentence_transformers_embedder():
    """Create a SentenceTransformers embedder for testing."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence_transformers package not installed")
    config = Config(LOCAL_EMBEDDING=True)
    return SentenceTransformersEmbedder(config)

@pytest.fixture(scope="session")
def session_test_db(request):
    """Create test database with appropriate vector dimension based on test markers."""
    setup_test_env()
    
    # Determine vector dimension from markers
    if request.node.get_closest_marker("openai"):
        vector_dim = 1536
    elif request.node.get_closest_marker("no_embeddings"):
        vector_dim = None
    else:
        # Default to sentence transformers
        vector_dim = 384
        
    if vector_dim is not None:
        config.EMBEDDINGS_DIM = vector_dim

    # Create fresh test database
    perform_db_reset()

    # Create engine for test database
    test_engine = create_engine(config.TEST_DB_URL)

    # Create all tables using our test vector type
    with test_engine.connect() as conn:
        try:
            # Set the embedding dimension directly in the model
            from vector_rag.db.db_model import DbBase, ChunkDB
            ChunkDB.embedding.type.dimension = vector_dim
            
            # Now create tables with correct dimension
            DbBase.metadata.create_all(test_engine)
        except Exception as e:
            pytest.fail(f"Failed to create tables: {str(e)}")

    yield test_engine

    # Only cleanup once at end of session
    if request.session.testsfailed:
        print("\nSome tests failed - preserving test database for debugging")
    else:
        perform_db_reset()

@pytest.fixture(scope="module")
def module_db_handler(request, session_test_db):
    """Create a DBFileHandler with appropriate configuration."""
    from vector_rag.db.db_file_handler import DBFileHandler
    
    # Determine vector dimension from markers
    if request.node.get_closest_marker("openai"):
        vector_dim = 1536
    elif request.node.get_closest_marker("no_embeddings"):
        vector_dim = None
    else:
        # Default to sentence transformers
        vector_dim = 384
        
    # Create appropriate embedder
    if vector_dim is not None:
        from vector_rag.embeddings.mock_embedder import MockEmbedder
        embedder = MockEmbedder(dimension=vector_dim)
    else:
        embedder = None
        
    return DBFileHandler(config, embedder=embedder)


@pytest.fixture
def test_files():
    """Create test files with different content lengths."""
    return [
        File(
            name=f"test{i}.txt",
            path=f"/path/to/test{i}.txt",
            crc=f"crc{i}",
            content=f"Test content {'x' * (i * 10)}\n" * (i + 1),
            metadata={"type": "test"},
        )
        for i in range(5)
    ]


@pytest.fixture(scope="module")
def test_files_with_metadata():
    """Create test files with different metadata."""
    return [
        File(
            name=f"test{i}.txt",
            path=f"/path/to/test{i}.txt",
            crc=f"crc{i}",
            content=f"Test content {'x' * (i * 10)}\n" * (i + 1),
            metadata={
                "type": "test",
                "category": "technical" if i % 2 == 0 else "non-technical",
                "source": "manual" if i < 3 else "auto-generated",
                "priority": str(i % 3 + 1)  # "1", "2", or "3"
            },
        )
        for i in range(6)
    ]

@pytest.fixture(scope="module")
def module_populated_handler_with_metadata(session_test_db, mock_embedder, test_files_with_metadata):
    """Create a handler with test data containing metadata."""
    # Ensure vector dimensions match
    ensure_vector_dimension(session_test_db, 384)

    handler = DBFileHandler.create(
        config.TEST_DB_NAME, mock_embedder, chunker=LineChunker.create(5, 0)
    )
    project = handler.create_project(f"Test Project with Metadata {str(uuid.uuid4())}")

    # Add test files and verify metadata was stored
    for file in test_files_with_metadata:
        file_record = handler.add_file(project.id, file)
        assert file_record is not None

        # Verify chunks were created with correct metadata
        with handler.session_scope() as session:
            chunks = session.query(handler.Chunk).filter_by(file_id=file_record.id).all()
            assert len(chunks) > 0, "No chunks were created for the file"
            for chunk in chunks:
                assert chunk.chunk_metadata == file.metadata, \
                    f"Chunk metadata mismatch: {chunk.chunk_metadata} != {file.metadata}"

    return handler, project.id

@pytest.fixture(scope="module")
def db_reset():
    """
    Reset the DB container by stopping it, removing its data folder, and restarting it.
    This fixture can be used explicitly if desired.
    """
    perform_db_reset()
    yield
