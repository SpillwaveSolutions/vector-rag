"""Test database file handler."""
import hashlib
import os
import tempfile
import uuid
from logging import debug
from pathlib import Path

import pytest

from vector_rag.config import Config
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.embeddings.mock_embedder import MockEmbedder
from vector_rag.model import File as FileModel

config = Config()

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
        metadata={},
    )

@pytest.mark.sentence
def test_add_duplicate_file_same_crc():
    """Test adding the same file twice with same CRC."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, MockEmbedder(dimension=384))
    project = handler.create_project(f"Test Project {str(uuid.uuid4())}")
    file_model = create_test_file()
    file1 = handler.add_file(project.id, file_model)
    assert file1 is not None
    file2 = handler.add_file(project.id, file_model)
    assert file2 is not None
    assert file2.id == file1.id  # Should return same file
    with handler.session_scope() as session:
        # Check that the file exists in the database
        db_file = session.query(handler.File).filter_by(id=file1.id).first()
        assert db_file is not None
        # Check that chunks were created
        chunks = session.query(handler.Chunk).filter_by(file_id=file1.id).all()
        assert len(chunks) > 0  # Should have original chunks

@pytest.mark.sentence
def test_add_duplicate_file_different_project():
    """Test adding same file to different projects."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, MockEmbedder(dimension=384))
    project1 = handler.create_project(f"Project 1 {str(uuid.uuid4())}")
    project2 = handler.create_project(f"Project 2 {str(uuid.uuid4())}")
    file_model = create_test_file()
    file1 = handler.add_file(project1.id, file_model)
    assert file1 is not None
    file2 = handler.add_file(project2.id, file_model)
    assert file2 is not None
    assert file2.id != file1.id  # Should be different files
    with handler.session_scope() as session:
        # Verify both files exist
        db_file1 = session.query(handler.File).filter_by(id=file1.id).first()
        db_file2 = session.query(handler.File).filter_by(id=file2.id).first()
        assert db_file1 is not None
        assert db_file2 is not None
        # Verify they belong to different projects
        assert db_file1.project_id == project1.id
        assert db_file2.project_id == project2.id
        # Check chunks
        chunks1 = session.query(handler.Chunk).filter_by(file_id=file1.id).count()
        chunks2 = session.query(handler.Chunk).filter_by(file_id=file2.id).count()
        assert chunks1 > 0
        assert chunks2 > 0
        assert chunks1 == chunks2  # Same content, so same number of chunks

@pytest.mark.sentence
def test_add_duplicate_file_different_crc():
    """Test adding same file with different content (different CRC)."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, MockEmbedder(dimension=384))
    project = handler.create_project(f"Test Project {str(uuid.uuid4())}")
    original_content = "Original content"
    file1 = handler.add_file(project.id, create_test_file(original_content))
    assert file1 is not None
    file1_id = file1.id
    with handler.session_scope() as session:
        original_chunk_count = session.query(handler.Chunk).filter_by(file_id=file1_id).count()
    
    # Add the same file with different content
    modified_content = "Modified content"
    file2 = handler.add_file(
        project.id,
        create_test_file(modified_content, name="test.txt", path="/path/to/test.txt"),
    )
    assert file2 is not None
    
    with handler.session_scope() as session:
        # Verify the file exists in the project
        file_in_project = session.query(handler.File).filter_by(
            project_id=project.id, 
            filename="test.txt", 
            file_path="/path/to/test.txt"
        ).first()
        assert file_in_project is not None
        
        # Verify the content was updated
        chunks = session.query(handler.Chunk).filter_by(file_id=file_in_project.id).all()
        assert len(chunks) > 0
        assert any(modified_content in chunk.content for chunk in chunks)

def test_file_versioning_workflow():
    """Integration test for complete file versioning workflow."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, MockEmbedder(dimension=384))
    project = handler.create_project(f"Test Project {str(uuid.uuid4())}")
    
    # Add first version
    content1 = "Version 1\nThis is the first version of the file."
    file1 = handler.add_file(project.id, create_test_file(content1))
    assert file1 is not None
    file1_id = file1.id
    
    with handler.session_scope() as session:
        chunks1 = session.query(handler.Chunk).filter_by(file_id=file1.id).all()
        chunk_count1 = len(chunks1)
        assert chunk_count1 > 0
    
    # Add second version
    content2 = "Version 2\nThis is the modified version with more content.\nExtra line."
    file2 = handler.add_file(
        project.id,
        create_test_file(content2, name="test.txt", path="/path/to/test.txt"),
    )
    assert file2 is not None
    
    with handler.session_scope() as session:
        # Verify the file in the project has been updated
        current_file = session.query(handler.File).filter_by(
            project_id=project.id,
            filename="test.txt",
            file_path="/path/to/test.txt"
        ).first()
        assert current_file is not None
        assert current_file.crc == file2.crc
        
        # Verify new chunks exist
        chunks2 = session.query(handler.Chunk).filter_by(file_id=current_file.id).all()
        assert len(chunks2) > 0
        
        # Verify content of chunks contains the new version
        assert any("Version 2" in chunk.content for chunk in chunks2)
        
        # Verify old chunks are gone or replaced
        if file1_id != current_file.id:  # If file was replaced rather than updated
            old_chunks = session.query(handler.Chunk).filter_by(file_id=file1_id).all()
            assert len(old_chunks) == 0

def test_create_project():
    """Test creating a project."""
    handler = DBFileHandler.create(config.TEST_DB_NAME)
    project_name = f"Test Project {str(uuid.uuid4())}"
    project = handler.create_project(project_name, "Test Description")
    assert project.id is not None
    assert project.name == project_name
    assert project.description == "Test Description"
    with handler.session_scope() as session:
        db_project = session.get(handler.Project, project.id)
        assert db_project is not None
        assert db_project.name == project_name
        assert db_project.description == "Test Description"

def test_get_project():
    """Test retrieving a project."""
    handler = DBFileHandler.create(config.TEST_DB_NAME)
    project_name = f"Test Project {str(uuid.uuid4())}"
    created = handler.create_project(project_name, "Test Description")
    project = handler.get_project(created.id)
    assert project is not None
    assert project.id == created.id
    assert project.name == created.name
    assert project.description == created.description
    assert handler.get_project(999) is None

def test_delete_project():
    """Test deleting a project."""
    handler = DBFileHandler.create(config.TEST_DB_NAME)
    project = handler.create_project(f"Test Project {str(uuid.uuid4())}")
    assert project.id is not None
    with handler.session_scope() as session:
        assert session.get(handler.Project, project.id) is not None
    assert handler.delete_project(project.id) is True
    with handler.session_scope() as session:
        assert session.get(handler.Project, project.id) is None

def test_delete_nonexistent_project():
    """Test deleting a project that doesn't exist."""
    handler = DBFileHandler.create(config.TEST_DB_NAME)
    assert handler.delete_project(999) is False

@pytest.mark.sentence
def test_add_file(embedder):
    """Test adding a file to a project."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, embedder)
    project = handler.create_project(f"Test Project {str(uuid.uuid4())}")
    file_model = create_test_file()
    debug(file_model)
    success = handler.add_file(project.id, file_model)
    assert success is not None
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(
            project_id=project.id, 
            filename=file_model.name
        ).first()
        assert file is not None
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
def test_add_file_to_nonexistent_project(embedder):
    """Test adding a file to a non-existent project."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, embedder)
    file_model = create_test_file()
    success = handler.add_file(999, file_model)
    assert success is None

@pytest.mark.sentence
def test_remove_file(embedder):
    """Test removing a file from a project."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, embedder)
    project = handler.create_project(f"Test Project {str(uuid.uuid4())}")
    file_model = create_test_file()
    success = handler.add_file(project.id, file_model)
    assert success is not None
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(
            project_id=project.id,
            filename=file_model.name
        ).first()
        assert file is not None
        file_id = file.id
    assert handler.remove_file(project.id, file_id) is True
    with handler.session_scope() as session:
        assert session.get(handler.File, file_id) is None

@pytest.mark.sentence
def test_remove_nonexistent_file(embedder):
    """Test removing a non-existent file."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, embedder)
    project = handler.create_project(f"Test Project {str(uuid.uuid4())}")
    assert handler.remove_file(project.id, 999) is False

@pytest.mark.sentence
def test_remove_file_wrong_project(embedder):
    """Test removing a file from the wrong project."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, embedder)
    project1 = handler.create_project(f"Project 1 {str(uuid.uuid4())}")
    project2 = handler.create_project(f"Project 2 {str(uuid.uuid4())}")
    file_model = create_test_file()
    file = handler.add_file(project1.id, file_model)
    assert file is not None
    with handler.session_scope() as session:
        file = session.query(handler.File).filter_by(
            project_id=project1.id,
            filename=file_model.name
        ).first()
        assert file is not None
        file_id = file.id
    assert handler.remove_file(project2.id, file_id) is False

@pytest.mark.sentence
def test_delete_file_success(embedder):
    """Test successful deletion of a file and its chunks."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, embedder)
    project = handler.create_project(f"Test Project {str(uuid.uuid4())}")
    file_model = create_test_file()
    added_file = handler.add_file(project.id, file_model)
    assert added_file is not None
    file_id = added_file.id
    
    # Verify file and chunks exist before deletion
    with handler.session_scope() as session:
        file = session.get(handler.File, file_id)
        assert file is not None
        chunks = session.query(handler.Chunk).filter(handler.Chunk.file_id == file_id).all()
        assert len(chunks) > 0
    
    # Delete the file
    result = handler.delete_file(file_id)
    assert result is True
    
    # Verify file and chunks are gone
    with handler.session_scope() as session:
        file = session.get(handler.File, file_id)
        assert file is None
        chunks = session.query(handler.Chunk).filter(handler.Chunk.file_id == file_id).all()
        assert len(chunks) == 0

@pytest.mark.sentence
def test_delete_nonexistent_file():
    """Test attempting to delete a non-existent file."""
    handler = DBFileHandler.create(config.TEST_DB_NAME)
    result = handler.delete_file(999999)
    assert result is False

@pytest.mark.sentence
def test_get_file(embedder):
    """Test getting a file by project ID, path and name."""
    handler = DBFileHandler.create(config.TEST_DB_NAME, embedder)
    project = handler.create_project(f"Test Project {str(uuid.uuid4())}")
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
    wrong_project = handler.create_project(f"Wrong Project {str(uuid.uuid4())}")
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
def test_get_projects():
    """Test getting project listings."""
    handler = DBFileHandler.create(config.TEST_DB_NAME)
    unique_id = str(uuid.uuid4())[:8]
    project_names = [f"Project A {unique_id}", f"Project B {unique_id}", f"Project C {unique_id}"]
    created_projects = []
    for name in project_names:
        project = handler.create_project(name, f"Description for {name}")
        created_projects.append(project)
    
    # Filter projects to only those with our unique ID
    with handler.session_scope() as session:
        our_projects = session.query(handler.Project).filter(
            handler.Project.name.like(f"%{unique_id}%")
        ).order_by(handler.Project.created_at.desc()).all()
        
        assert len(our_projects) == len(project_names)
        
        # Test ordering by created_at
        for i in range(len(our_projects) - 1):
            assert our_projects[i].created_at >= our_projects[i + 1].created_at
        
        # Test limit
        limited = session.query(handler.Project).filter(
            handler.Project.name.like(f"%{unique_id}%")
        ).order_by(handler.Project.created_at.desc()).limit(2).all()
        assert len(limited) == 2
        assert limited[0].name in project_names
        
        # Test offset
        offset = session.query(handler.Project).filter(
            handler.Project.name.like(f"%{unique_id}%")
        ).order_by(handler.Project.created_at.desc()).offset(1).all()
        assert len(offset) == 2
        assert offset[0].name in project_names
        
        # Test paging
        paged = session.query(handler.Project).filter(
            handler.Project.name.like(f"%{unique_id}%")
        ).order_by(handler.Project.created_at.desc()).limit(1).offset(1).all()
        assert len(paged) == 1
        assert paged[0].name in project_names

@pytest.mark.sentence
def test_get_projects_empty():
    """Test getting project listings when there are no projects."""
    # Use the existing database but with a unique project prefix
    unique_prefix = f"empty_test_{str(uuid.uuid4())[:8]}"
    handler = DBFileHandler.create(config.TEST_DB_NAME)
    
    # Check that no projects exist with our unique prefix
    with handler.session_scope() as session:
        projects = session.query(handler.Project).filter(
            handler.Project.name.like(f"{unique_prefix}%")
        ).all()
        assert len(projects) == 0
        
        # Test with limit
        limited = session.query(handler.Project).filter(
            handler.Project.name.like(f"{unique_prefix}%")
        ).limit(10).all()
        assert len(limited) == 0
        
        # Test with offset
        offset = session.query(handler.Project).filter(
            handler.Project.name.like(f"{unique_prefix}%")
        ).offset(5).all()
        assert len(offset) == 0
