import importlib
import os
import sys
from pathlib import Path
import subprocess
import time

from vector_rag.db.db_model import DbBase, ChunkDB

# Add src directory to Python path
src_path = str(Path(__file__).parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from vector_rag.config import Config
from vector_rag.embeddings import MockEmbedder

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

@pytest.fixture
def embedding_group(request):
    """Return the embedding group for the current test based on markers."""
    if request.node.get_closest_marker("openai"):
        return "openai"
    return "sentence"

@pytest.fixture
def mock_embedder(request):
    """
    Create a mock embedder for testing.
    If the test is marked with 'openai', returns an embedder with 1536 dimensions;
    otherwise, returns one with 384 dimensions.
    """
    if request.node.get_closest_marker("openai"):
        return MockEmbedder(dimension=1536)
    else:
        return MockEmbedder(dimension=384)

@pytest.fixture(scope="function")
def test_db(request):
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

    # Cleanup using perform_db_reset
    perform_db_reset()

@pytest.fixture
def db_handler(request, test_db):
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

@pytest.fixture(scope="module")
def db_reset():
    """
    Reset the DB container by stopping it, removing its data folder, and restarting it.
    This fixture can be used explicitly if desired.
    """
    perform_db_reset()
    yield
