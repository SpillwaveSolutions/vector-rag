import uuid
from datetime import datetime
from vector_rag.config import Config
from vector_rag.model import Chunk, File

config = Config()


def test_add_chunk(module_db_handler):
    # Create a project with unique name
    unique_id = str(uuid.uuid4())[:8]
    project = module_db_handler.create_project(f"Test Project test_add_chunk {unique_id}")

    # Create a file
    file_model = File(name="test.txt", path="/test/test.txt", crc="123", content="Test")
    file = module_db_handler.add_file(project.id, file_model)

    # Create a chunk
    chunk = Chunk(target_size=100, content="Test chunk content", index=0)

    # Add the chunk
    result = module_db_handler.add_chunk(file.id, chunk)

    # Assert the chunk was added successfully
    assert result is not None
    assert isinstance(result, Chunk)
    assert result.content == chunk.content
    assert result.index == chunk.index


def test_add_chunk_nonexistent_file(module_db_handler):
    # Try to add a chunk to a non-existent file
    chunk = Chunk(target_size=100, content="Test chunk content", index=0)
    result = module_db_handler.add_chunk(999, chunk)

    # Assert the operation failed
    assert result is None


def test_add_chunk_with_metadata(module_db_handler):
    # Create a project with unique name
    unique_id = str(uuid.uuid4())[:8]
    project = module_db_handler.create_project(f"Test Project with metadata {unique_id}")

    # Create a file
    file_model = File(
        name="test.txt", path="/test/test.txt", crc="123", content="Test content"
    )
    file = module_db_handler.add_file(project.id, file_model)

    # Create a chunk with metadata
    metadata = {"source": "test_file", "page": 1, "importance": "high"}
    chunk = Chunk(
        target_size=100, content="Test chunk content", index=0, meta_data=metadata
    )

    # Add the chunk
    result = module_db_handler.add_chunk(file.id, chunk)

    # Assert the chunk was added successfully
    assert result is not None
    assert isinstance(result, Chunk)
    assert result.content == chunk.content
    assert result.index == chunk.index
    assert result.meta_data == metadata
