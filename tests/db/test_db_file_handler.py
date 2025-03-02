"""Test database file handler."""
import hashlib
import os
import tempfile
from logging import debug
from pathlib import Path

import pytest

from vector_rag.config import Config
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.embeddings.mock_embedder import MockEmbedder
from vector_rag.model import File as FileModel

config = Config()
TEST_DB_NAME = config.TEST_DB_NAME

@pytest.fixture
def embedder(request):
    """Create a mock embedder for testing.
    Uses 1536 dimensions if the test is marked with 'openai', otherwise 384.
    """
    if request.node.get_closest_marker("openai"):
        return MockEmbedder(dimension=1536)
    else:
        return MockEmbedder(dimension=384)

def create_test_file(content="Test content", name="test.txt", path="/path/to/test.txt"):
    """Create a test file model."""
    return FileModel(
        name=name,
        path=path,
        crc=str(hash(content)),  # Simple hash for testing
        content=content,
        meta_data={},
    )

@pytest.mark.sentence
def test_add_duplicate_file_same_crc(test_db):
    """Test adding the same file twice with same CRC."""
    handler = DBFileHandler.create(TEST_DB_NAME, MockEmbedder(dimension=384))
    project = handler.create_project("Test Project")
    file_model = create_test_file()
    file1 = handler.add_file(project.id, file_model)
    assert file1 is not None
    file2 = handler.add_file(project.id, file_model)
    assert file2 is not None
    assert file2.id == file1.id  # Should return same file
    with handler.session_scope() as session:
        file_count = session.query(handler.File).count()
        assert file_count == 1
        chunk_count = session.query(handler.Chunk).count()
        assert chunk_count > 0  # Should have original chunks

@pytest.mark.sentence
def test_add_duplicate_file_different_project(test_db):
    """Test adding same file to different projects."""
    handler = DBFileHandler.create(TEST_DB_NAME, MockEmbedder(dimension=384))
    project1 = handler.create_project("Project 1")
    project2 = handler.create_project("Project 2")
    file_model = create_test_file()
    file1 = handler.add_file(project1.id, file_model)
    assert file1 is not None
    file2 = handler.add_file(project2.id, file_model)
    assert file2 is not None
    assert file2.id != file1.id  # Should be different files
    with handler.session_scope() as session:
        file_count = session.query(handler.File).count()
        assert file_count == 2
        chunks1 = session.query(handler.Chunk).filter_by(file_id=file1.id).count()
        chunks2 = session.query(handler.Chunk).filter_by(file_id=file2.id).count()
        assert chunks1 > 0
        assert chunks2 > 0
        assert chunks1 == chunks2  # Same content, so same number of chunks

@pytest.mark.sentence
def test_add_duplicate_file_different_crc(test_db):
    """Test adding same file with different content (different CRC)."""
    handler = DBFileHandler.create(TEST_DB_NAME, MockEmbedder(dimension=384))
    project = handler.create_project("Test Project")
    original_content = "Original content"
    file1 = handler.add_file(project.id, create_test_file(original_content))
    assert file1 is not None
    file1_id = file1.id
    with handler.session_scope() as session:
        original_chunk_count = session.query(handler.Chunk).filter_by(file_id=file1_id).count()
    modified_content = "Modified content"
    handler.add_file(
        project.id,
        create_test_file(modified_content, name="test.txt", path="/path/to/test.txt"),
    )
    with handler.session_scope() as session:
        file_count = session.query(handler.File).count()
        assert file_count == 1
        current_file = session.query(handler.File).first()
        assert current_file is not None
        current_chunks = session.query(handler.Chunk).all()
        assert len(current_chunks) > 0
        assert any(modified_content in chunk.content for chunk in current_chunks)

def test_file_versioning_workflow(test_db):
    """Integration test for complete file versioning workflow."""
    handler = DBFileHandler.create(TEST_DB_NAME, MockEmbedder(dimension=384))
    project = handler.create_project("Test Project")
    content1 = "Version 1\nThis is the first version of the file."
    file1 = handler.add_file(project.id, create_test_file(content1))
    assert file1 is not None
    with handler.session_scope() as session:
        chunks1 = session.query(handler.Chunk).filter_by(file_id=file1.id).all()
        chunk_count1 = len(chunks1)
        assert chunk_count1 > 0
    content2 = "Version 2\nThis is the modified version with more content.\nExtra line."
    file2 = handler.add_file(
        project.id,
        create_test_file(content2, name="test.txt", path="/path/to/test.txt"),
    )
    assert file2 is not None
    assert file2.id != file1.id
    with handler.session_scope() as session:
        files = session.query(handler.File).all()
        assert len(files) == 1
        assert files[0].id == file2.id
        assert files[0].crc == file2.crc
        chunks2 = session.query(handler.Chunk).filter_by(file_id=file2.id).all()
        assert len(chunks2) > 0
        assert len(chunks2) >= chunk_count1
        old_chunks = session.query(handler.Chunk).filter_by(file_id=file1.id).all()
        assert len(old_chunks) == 0

def test_create_project(test_db):
    """Test creating a project."""
    handler = DBFileHandler.create(TEST_DB_NAME)
    project = handler.create_project("Test Project", "Test Description")
    assert project.id is not None
    assert project.name == "Test Project"
    assert project.description == "Test Description"
    with handler.session_scope() as session:
        db_project = session.get(handler.Project, project.id)
        assert db_project is not None
        assert db_project.name == "Test Project"
        assert db_project.description == "Test Description"

def test_get_project(test_db):
    """Test retrieving a project."""
    handler = DBFileHandler.create(TEST_DB_NAME)
    created = handler.create_project("Test Project", "Test Description")
    project = handler.get_project(created.id)
    assert project is not None
    assert project.id == created.id
    assert project.name == created.name
    assert project.description == created.description
    assert handler.get_project(999) is None

def test_delete_project(test_db):
    """Test deleting a project."""
    handler = DBFileHandler.create(TEST_DB_NAME)
    project = handler.create_project("Test Project")
    assert project.id is not None
    with handler.session_scope() as session:
        assert session.get(handler.Project, project.id) is not None
    assert handler.delete_project(project.id) is True
    with handler.session_scope() as session:
        assert session.get(handler.Project, project.id) is None

def test_delete_nonexistent_project(test_db):
    """Test deleting a project that doesn't exist."""
    handler = DBFileHandler.create(TEST_DB_NAME)
    assert handler.delete_project(999) is False

@pytest.mark.sentence
def test_add_file(test_db, embedder):
    """Test adding a file to a project."""
    handler = DBFileHandler.create(TEST_DB_NAME, embedder)
    project = handler.create_project("Test Project")
    file_model = create_test_file()
    debug(file_model)
    success = handler.add_file(project.id, file_model)
    assert success is not None
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        assert file is not None
        assert file.project_id == project.id
        assert file.filename == file_model.name
        assert file.file_path == file_model.path
        assert file.created_at is not None
        chunks = session.query(handler.Chunk).filter_by(file_id=file.id).all()
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.embedding is not None
            assert len(chunk.embedding) == embedder.get_dimension()

@pytest.mark.sentence
def test_add_file_to_nonexistent_project(test_db, embedder):
    """Test adding a file to a non-existent project."""
    handler = DBFileHandler.create(TEST_DB_NAME, embedder)
    file_model = create_test_file()
    success = handler.add_file(999, file_model)
    assert success is None

@pytest.mark.sentence
def test_remove_file(test_db, embedder):
    """Test removing a file from a project."""
    handler = DBFileHandler.create(TEST_DB_NAME, embedder)
    project = handler.create_project("Test Project")
    file_model = create_test_file()
    success = handler.add_file(project.id, file_model)
    assert success is not None
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        assert file is not None
        file_id = file.id
    assert handler.remove_file(project.id, file_id) is True
    with handler.session_scope() as session:
        assert session.get(handler.File, file_id) is None

@pytest.mark.sentence
def test_remove_nonexistent_file(test_db, embedder):
    """Test removing a non-existent file."""
    handler = DBFileHandler.create(TEST_DB_NAME, embedder)
    project = handler.create_project("Test Project")
    assert handler.remove_file(project.id, 999) is False

@pytest.mark.sentence
def test_remove_file_wrong_project(test_db, embedder):
    """Test removing a file from the wrong project."""
    handler = DBFileHandler.create(TEST_DB_NAME, embedder)
    project1 = handler.create_project("Project 1")
    project2 = handler.create_project("Project 2")
    file_model = create_test_file()
    file = handler.add_file(project1.id, file_model)
    assert file is not None
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(filename=file_model.name).first()
        assert file is not None
        file_id = file.id
    assert handler.remove_file(project2.id, file_id) is False

@pytest.mark.sentence
def test_delete_file_success(test_db, embedder):
    """Test successful deletion of a file and its chunks."""
    handler = DBFileHandler.create(TEST_DB_NAME, embedder)
    project = handler.create_project("Test Project")
    file_model = create_test_file()
    handler.add_file(project.id, file_model)
    with handler.session_scope() as session:
        file = session.query(handler.File).filter(handler.File.file_path == file_model.path).first()
        file_id = file.id
        chunks = session.query(handler.Chunk).filter(handler.Chunk.file_id == file_id).all()
        assert len(chunks) > 0
    result = handler.delete_file(file_id)
    assert result is True
    with handler.session_scope() as session:
        file = session.get(handler.File, file_id)
        assert file is None
        chunks = session.query(handler.Chunk).filter(handler.Chunk.file_id == file_id).all()
        assert len(chunks) == 0

@pytest.mark.sentence
def test_delete_nonexistent_file(test_db):
    """Test attempting to delete a non-existent file."""
    handler = DBFileHandler.create(TEST_DB_NAME)
    result = handler.delete_file(999999)
    assert result is False

@pytest.mark.sentence
def test_get_file(test_db, embedder):
    """Test getting a file by project ID, path and name."""
    handler = DBFileHandler.create(TEST_DB_NAME, embedder)
    project = handler.create_project("Test Project")
    file_model = create_test_file("Test content")
    added_file = handler.add_file(project.id, file_model)
    assert added_file is not None
    found_file = handler.get_file(
        project_id=project.id, file_path=file_model.path, filename=file_model.name
    )
    assert found_file is not None
    assert found_file.id == added_file.id
    assert found_file.name == file_model.name
    assert found_file.path == file_model.path
    wrong_project = handler.create_project("Wrong Project")
    not_found = handler.get_file(
        project_id=wrong_project.id, file_path=file_model.path, filename=file_model.name
    )
    assert not_found is None
    not_found = handler.get_file(
        project_id=project.id, file_path="/wrong/path.txt", filename=file_model.name
    )
    assert not_found is None
    not_found = handler.get_file(
        project_id=project.id, file_path=file_model.path, filename="wrong.txt"
    )
    assert not_found is None

def teardown_module(module):
    """Clean up temporary files after tests."""
    for item in Path().glob("*.txt"):
        if item.is_file() and item.suffix == ".txt":
            try:
                item.unlink()
            except OSError:
                pass

@pytest.mark.sentence
def test_get_projects(test_db):
    """Test getting project listings."""
    handler = DBFileHandler.create(TEST_DB_NAME)
    project_names = ["Project A", "Project B", "Project C"]
    created_projects = []
    for name in project_names:
        project = handler.create_project(name, f"Description for {name}")
        created_projects.append(project)
    all_projects = handler.get_projects()
    assert len(all_projects) == len(project_names)
    for i in range(len(all_projects) - 1):
        assert all_projects[i].created_at >= all_projects[i + 1].created_at
    limited_projects = handler.get_projects(limit=2)
    assert len(limited_projects) == 2
    assert limited_projects[0].name == project_names[-1]
    offset_projects = handler.get_projects(offset=1)
    assert len(offset_projects) == 2
    assert offset_projects[0].name == project_names[-2]
    paged_projects = handler.get_projects(limit=1, offset=1)
    assert len(paged_projects) == 1
    assert paged_projects[0].name == project_names[-2]

@pytest.mark.sentence
def test_get_projects_empty(test_db):
    """Test getting project listings when there are no projects."""
    handler = DBFileHandler.create(TEST_DB_NAME)
    projects = handler.get_projects()
    assert len(projects) == 0
    assert len(handler.get_projects(limit=10)) == 0
    assert len(handler.get_projects(offset=5)) == 0
