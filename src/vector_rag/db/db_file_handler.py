"""Database file handler for managing projects and files."""

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Union

import numpy as np
from sqlalchemy import Float, create_engine, func, literal, select, text
from sqlalchemy.orm import sessionmaker

from vector_rag.model import Chunk, ChunkResult, ChunkResults, File, Project
from .. import config

from ..chunking import LineChunker
from ..chunking.base_chunker import Chunker
from ..config import Config
from ..embeddings import Embedder, OpenAIEmbedder
from .base_file_handler import FileHandler
from .db_model import ChunkDB, DbBase, FileDB, ProjectDB
from .dimension_utils import ensure_vector_dimension
from ..embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder

logger = logging.getLogger(__name__)

class DBFileHandler(FileHandler):
    """Handler for managing files in the database."""

    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    # Only enable SQL logging if SQL_DEBUG_LOGGING is set to true
    if config.get_or_default('SQL_DEBUG_LOGGING', '').lower() == 'true':
        @event.listens_for(Engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault('query_start_time', []).append(time.time())
            print(f"SQL: {statement}")
            print(f"Parameters: {parameters}")

        @event.listens_for(Engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - conn.info['query_start_time'].pop(-1)
            print(f"Total time: {total:.2f}s")

    def __init__(
        self,
        config: Config = None,
        embedder: Optional[Embedder] = None,
        chunker: Optional[Chunker] = None,
    ):
        """Initialize the handler.

        Args:
            db_url: Database URL, defaults to config.DB_URL
            embedder: Embedder instance for generating embeddings

        Raises:
            ValueError: If db_url is None
        """

        if config.DB_URL is None:
            raise ValueError("Database URL must be provided")

        self.engine = create_engine(config.DB_URL)
        self.create_tables_and_indexes()
        self.embedder = embedder or SentenceTransformersEmbedder(config)
        self.Session = sessionmaker(bind=self.engine)
        self.chunker: Chunker

        if not chunker:
            self.chunker = LineChunker(config)
        else:
            self.chunker = chunker

        # Make models accessible
        self.Project = ProjectDB
        self.File = FileDB
        self.Chunk = ChunkDB



        # Ensure vector dimension matches embedder if provided
        if self.embedder:
            embedder_dim = self.embedder.get_dimension()
            if config.EMBEDDINGS_DIM and embedder_dim != config.EMBEDDINGS_DIM:
                raise ValueError(
                    f"Embedder dimension ({embedder_dim}) does not match "
                    f"configured dimension ({config.EMBEDDINGS_DIM})"
                )
            ensure_vector_dimension(self.engine, embedder_dim)

    def create_tables_and_indexes(self):
        DbBase.metadata.create_all(self.engine)

        # Create indexes for array fields and unique constraint for metadata within a chunk
        with self.engine.connect() as connection:
            # Enable required extensions for text search and array operations
            connection.execute(text("""
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                CREATE EXTENSION IF NOT EXISTS btree_gin;
            """))
            connection.commit()

        with self.engine.connect() as connection:
            # Check if indexes already exist before creating them
            connection.execute(text("""
                -- Create GIN indexes for JSONB metadata
                CREATE INDEX IF NOT EXISTS idx_chunk_metadata ON chunks USING gin (chunk_metadata);
                
                -- Create index for text search on metadata (no lower function needed)
                CREATE INDEX IF NOT EXISTS idx_chunk_metadata_gin_ops ON chunks USING gin (chunk_metadata jsonb_path_ops);
            """))
            connection.commit()

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_or_create_project(
        self, name: str, description: Optional[str] = None
    ) -> Project:
        """Get an existing project by name or create a new one.

        Args:
            name: Project name
            description: Optional project description

        Returns:
            Project instance
        """
        with self.session_scope() as session:
            project = session.query(ProjectDB).filter(ProjectDB.name == name).first()
            if project:
                if description and description != project.description:
                    project.description = description
                    project.updated_at = datetime.now(timezone.utc)
                session.flush()
                # Create a detached copy with all attributes loaded
                project_copy = Project(
                    name=project.name, description=project.description, id=project.id
                )
                return project_copy

            project = ProjectDB(name=name, description=description)
            session.add(project)
            session.flush()
            # Create a detached copy with all attributes loaded
            project_copy = Project(
                name=project.name, description=project.description, id=project.id
            )
            return project_copy

    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new project.

        Args:
            name: Project name
            description: Optional project description

        Returns:
            Project instance

        Raises:
            ValueError: If project with given name already exists
        """
        with self.session_scope() as session:
            existing = session.query(ProjectDB).filter(ProjectDB.name == name).first()
            if existing:
                raise ValueError(f"Project with name '{name}' already exists")

            project = ProjectDB(name=name, description=description)
            session.add(project)
            session.flush()
            # Create a detached copy with all attributes loaded
            project_copy = Project(
                name=project.name, description=project.description, id=project.id
            )
            return project_copy

    def get_project(self, project_id: int) -> Optional[Project]:
        """Get a project by ID.

        Args:
            project_id: ID of the project

        Returns:
            Project if found, None otherwise
        """
        with self.session_scope() as session:
            project = session.get(ProjectDB, project_id)
            if project:
                # Get a copy of the data
                return Project(
                    name=project.name, description=project.description, id=project.id
                )
            return None

    def delete_project(self, project_id: int) -> bool:
        """Delete a project and all its files.

        Args:
            project_id: ID of the project to delete

        Returns:
            bool: True if project was deleted, False if not found
        """
        with self.session_scope() as session:
            project = session.get(ProjectDB, project_id)
            if project:
                session.delete(project)
                return True
            return False

    def add_chunks(self, file_id: int, chunks: List[Chunk]) -> List[Optional[Chunk]]:
        """Add multiple chunks to the database for a given file in an optimized batch.

        Args:
            file_id: ID of the file these chunks belong to
            chunks: List of Chunk objects to be added

        Returns:
            List[Optional[Chunk]]: List of created chunk objects, with None for any failed chunks
        """
        with self.session_scope() as session:
            # Verify file exists
            file = session.get(FileDB, file_id)
            if not file:
                logger.error(f"File with id {file_id} not found")
                return [None] * len(chunks)

            try:
                # Generate embeddings for all chunks in one call
                embeddings = self.embedder.embed_texts(chunks)

                # Create and add all chunk records
                created_chunks = []
                for chunk, embedding in zip(chunks, embeddings):
                    chunk_db = ChunkDB(
                        file_id=file_id,
                        content=chunk.content,
                        embedding=embedding,
                        chunk_index=chunk.index,
                        chunk_metadata=chunk.metadata,
                    )
                    session.add(chunk_db)

                    # Create Chunk object for return
                    created_chunks.append(Chunk(
                        target_size=chunk.target_size,
                        content=chunk_db.content,
                        index=chunk_db.chunk_index,
                        metadata=chunk_db.chunk_metadata,
                    ))

                session.flush()
                return created_chunks

            except Exception as e:
                logger.error(f"Error adding chunks to file {file_id}: {str(e)}")
                return [None] * len(chunks)

    def add_chunk(self, file_id: int, chunk: Chunk) -> Optional[Chunk]:
        """Add a single chunk to the database for a given file.

        Args:
            file_id: ID of the file this chunk belongs to
            chunk: Chunk object to be added

        Returns:
            Chunk: Created chunk object
            None: If file doesn't exist or an error occurs
        """
        with self.session_scope() as session:
            # Verify file exists
            file = session.get(FileDB, file_id)
            if not file:
                logger.error(f"File with id {file_id} not found")
                return None

            try:
                # Generate embedding for the chunk
                embedding = self.embedder.embed_texts([chunk])[0]

                # Create new chunk record
                chunk_db = ChunkDB(
                    file_id=file_id,
                    content=chunk.content,
                    embedding=embedding,
                    chunk_index=chunk.index,
                    chunk_metadata=chunk.metadata,
                )
                session.add(chunk_db)
                session.flush()  # Get chunk_db.id

                # Create and return a Chunk object
                return Chunk(
                    target_size=chunk.target_size,
                    content=chunk_db.content,
                    index=chunk_db.chunk_index,
                    metadata=chunk_db.chunk_metadata,
                )

            except Exception as e:
                logger.error(f"Error adding chunk to file {file_id}: {str(e)}")
                return None

    def add_file(self, project_id: int, file_model: File) -> Optional[File]:
        """Add a file to a project with version checking.

        Args:
            project_id: ID of the project
            file_model: File model containing file information

        Returns:
            FileDB: Created or existing file record
            None: If project doesn't exist
        """
        with self.session_scope() as session:
            # Verify project exists
            project = session.get(ProjectDB, project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return None

            # Check if file already exists
            existing_file = self.get_file(project_id, file_model.path, file_model.name)

            if existing_file:
                if existing_file.crc == file_model.crc:
                    logger.info(
                        f"File {file_model.name} already exists with same CRC {file_model.crc}"
                    )
                    return existing_file
                else:
                    logger.info(
                        f"File {file_model.name} exists but CRC differs "
                        f"(old: {existing_file.crc}, new: {file_model.crc})"
                    )
                    logger.info("Deleting old version and creating new version")
                    if existing_file.id is not None:
                        self.delete_file(existing_file.id)

            # Create new file record
            file = FileDB(
                project_id=project_id,
                filename=file_model.name,
                file_path=file_model.path,
                crc=file_model.crc,
                file_size=file_model.size,
            )
            session.add(file)
            session.flush()  # Get file.id

            # Create chunks and preserve metadata
            chunks: List[Chunk] = self.chunker.chunk_text(file_model)
            # Copy file metadata to each chunk
            for chunk in chunks:
                chunk.metadata = file_model.metadata.copy()
            
            embeddings = self.embedder.embed_texts(chunks)

            for chunk, embedding in zip(chunks, embeddings):
                chunk_obj = ChunkDB(
                    file_id=file.id,
                    content=chunk.content,
                    embedding=embedding,
                    chunk_index=chunk.index,
                    chunk_metadata=chunk.metadata,
                )
                session.add(chunk_obj)

            # Get a copy of the file data
            file_data = File(
                id=file.id,
                name=file.filename,
                path=file.file_path,
                file_size=file.file_size,
                crc=file.crc,
            )

            # Commit to ensure the data is saved
            session.commit()
            logger.info(
                f"Successfully added file {file_model.name} to project {project_id}"
            )

            # Return a new instance with the data
            return file_data

    def remove_file(self, project_id: int, file_id: int) -> bool:
        """Remove a file from a project.

        Args:
            project_id: ID of the project
            file_id: ID of the file to remove

        Returns:
            bool: True if file was removed, False if not found
        """
        with self.session_scope() as session:
            file = session.get(FileDB, file_id)
            if file and file.project_id == project_id:
                session.delete(file)
                return True
            return False

    def delete_file(self, file_id: int) -> bool:
        """Delete a file and all its associated chunks from the database.

        Args:
            file_id: ID of the file to delete

        Returns:
            bool: True if file was deleted, False if not found
        """
        with self.session_scope() as session:
            file = session.get(self.File, file_id)
            if file is None:
                return False

            session.delete(file)
            session.flush()
            return True

    def get_file(
        self, project_id: int, file_path: str, filename: str
    ) -> Optional[File]:
        """Look up a file by project ID, path and name.

        Args:
            project_id: ID of the project containing the file
            file_path: Full path of the file
            filename: Name of the file

        Returns:
            File if found, None otherwise
        """
        with self.session_scope() as session:
            file = (
                session.query(FileDB)
                .filter(FileDB.project_id == project_id)
                .filter(FileDB.file_path == file_path)
                .filter(FileDB.filename == filename)
                .first()
            )

            if file:
                # Return a copy of the file data
                return File(
                    id=file.id,
                    name=file.filename,
                    path=file.file_path,
                    file_size=file.file_size,
                    crc=file.crc,
                )
            else:
                return None

    def get_projects(self, limit: int = -1, offset: int = -1) -> List[ProjectDB]:
        """Get a list of all projects.

        Args:
            limit: Maximum number of projects to return
            offset: Number of projects to skip

        Returns:
            List[ProjectDB]: List of projects ordered by creation date (newest first)
        """
        with self.session_scope() as session:
            query = session.query(ProjectDB).order_by(ProjectDB.created_at.desc())

            if limit != -1:
                query = query.limit(limit)
            if offset != -1:
                query = query.offset(offset)

            projects = query.all()

            # Create detached copies of the projects
            return [
                ProjectDB(
                    id=project.id,
                    name=project.name,
                    description=project.description,
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                )
                for project in projects
            ]

    def list_files(self, project_id: int) -> List[File]:
        """List all files in a project.

        Args:
            project_id: ID of the project

        Returns:
            List[FileModel]: List of file models, empty list if project doesn't exist
        """
        with self.session_scope() as session:
            # Verify project exists
            project = session.get(self.Project, project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return []

            # Query all files for the project
            db_files = (
                session.query(self.File)
                .filter(self.File.project_id == project_id)
                .all()
            )

            # Convert DB models to FileModel instances
            files = []
            for db_file in db_files:
                # Get all chunks for this file, ordered by chunk_index
                chunks = (
                    session.query(self.Chunk)
                    .filter(self.Chunk.file_id == db_file.id)
                    .order_by(self.Chunk.chunk_index)
                    .all()
                )

                # Reconstruct original content from chunks
                content = "\n".join(chunk.content for chunk in chunks)

                # Create FileModel instance
                file_model = File(
                    id=db_file.id,
                    name=db_file.filename,
                    path=db_file.file_path,
                    crc=db_file.crc,
                    content=content,
                    metadata={
                        "type": (
                            db_file.filename.split(".")[-1]
                            if "." in db_file.filename
                            else ""
                        )
                    },
                )
                files.append(file_model)

            return files

    def search_chunks_by_text(
        self,
        project_id: int,
        query_text: str,
        page: int = 1,
        page_size: int = 10,
        similarity_threshold: float = 0.7,
        file_id: int = None,
        metadata_filter: Optional[dict] = None,
    ) -> ChunkResults:
        """Search for chunks in a project using text query with pagination and metadata filtering."""
        if page < 1:
            raise ValueError("Page number must be greater than 0")
        if page_size < 1:
            raise ValueError("Page size must be greater than 1")

        # Get embedding for query text
        logger.info(f"Creating embedding for query: '{query_text}'")
        logger.debug(f"Metadata filter: {metadata_filter}")
        print(f"DEBUG: Creating embedding for query: '{query_text}' with metadata filter: {metadata_filter}")
        query_embedding = self.embedder.embed_texts(
            [Chunk(target_size=1, content=query_text, index=0)]
        )[0]

        return self.search_chunks_by_embedding(
            project_id, 
            query_embedding, 
            page, 
            page_size, 
            similarity_threshold,
            file_id,
            metadata_filter=metadata_filter
        )

    def search_chunks_by_embedding(
        self,
        project_id: int,
        embedding: Union[np.ndarray, Sequence[float]],
        page: int = 1,
        page_size: int = 10,
        similarity_threshold: float = 0.7,
        file_id: int = None,
        metadata_filter: Optional[dict] = None,
    ) -> ChunkResults:
        if page < 1:
            raise ValueError("Page number must be greater than 0")
        if page_size < 1:
            raise ValueError("Page size must be greater than 1")

        logger.info(f"Searching with embedding in project {project_id}")
        logger.debug(f"Metadata filter: {metadata_filter}")
        logger.debug(f"Similarity threshold: {similarity_threshold}")
        
        # Ensure `embedding` is a 1D float32 (big-endian) array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=">f4")
        elif embedding.dtype != ">f4":
            embedding = embedding.astype(">f4")
        embedding = embedding.ravel()

        with self.session_scope() as session:
            # distance_expr is chunks.embedding <=> your_query_embedding
            distance_expr = self.Chunk.embedding.op("<=>")(embedding)

            # Mark 1.0 as a float literal so that it doesn't become a vector
            similarity_expr = (literal(1.0, type_=Float) - distance_expr).label(
                "similarity"
            )

            # Also mark threshold as a float literal if needed
            threshold_expr = literal(similarity_threshold, type_=Float)

            # Build base query
            base_query = (
                select(self.Chunk, similarity_expr)
                .join(self.File)
                .where(self.File.project_id == project_id)
                .where(similarity_expr >= threshold_expr)  # numeric comparison
            )

            if file_id:
                base_query = base_query.where(self.File.id == file_id)

            # Add metadata filtering if provided
            if metadata_filter:
                logger.info(f"Applying metadata filter: {metadata_filter}")
                for key, value in metadata_filter.items():
                    # Handle nested JSON objects
                    if isinstance(value, dict):
                        logger.debug(f"Filtering for nested object {key}: {value}")
                        # Create a JSON object for containment check
                        json_obj = {key: value}
                        # Use JSONB containment operator @> for nested objects
                        base_query = base_query.where(
                            self.Chunk.chunk_metadata.op('@>')(json_obj)
                        )
                    # Handle lists of values
                    elif isinstance(value, list):
                        logger.debug(f"Filtering for multiple values of {key}: {value}")
                        
                        # Check if we're looking for a value in a list field
                        if len(value) == 1:
                            # We might be looking for a single value in a list field
                            # Use the JSONB containment operator @> for this
                            logger.debug(f"Checking if list field contains value: {value[0]}")
                            json_obj = {key: value}
                            base_query = base_query.where(
                                self.Chunk.chunk_metadata.op('@>')(json_obj)
                            )
                        else:
                            # We're looking for multiple possible values for this key
                            # Use OR conditions for multiple possible values
                            from sqlalchemy import or_
                            conditions = []
                            
                            # Try both direct equality and containment for arrays
                            for val in value:
                                # Direct equality check
                                conditions.append(self.Chunk.chunk_metadata[key].astext == str(val))
                                
                                # Check if the value is in a JSON array
                                json_obj = {key: [val]}
                                conditions.append(self.Chunk.chunk_metadata.op('@>')(json_obj))
                            
                            base_query = base_query.where(or_(*conditions))
                    # Handle simple key-value pairs
                    else:
                        logger.debug(f"Filtering for {key}={value}")
                        base_query = base_query.where(
                            self.Chunk.chunk_metadata[key].astext == str(value)
                        )
            
            # Print the query for debugging
            print(f"DEBUG: Performing search with metadata filter: {metadata_filter}")
            logger.debug(f"Performing search: base_query: {base_query}")

            # Count how many total rows match
            count_query = select(func.count()).select_from(base_query.subquery())
            total_count = session.execute(count_query).scalar() or 0
            
            # Direct print for debugging
            print(f"DEBUG: Found {total_count} total matching rows for query")

            # Pagination
            offset = (page - 1) * page_size
            results = session.execute(
                base_query.order_by(similarity_expr.desc())
                .offset(offset)
                .limit(page_size)
            ).all()

            logger.debug(f"Found {len(results)} results")
            # Convert to your Pydantic "ChunkResults"
            chunk_results = []
            for chunk_row, similarity in results:
                chunk_results.append(
                    ChunkResult(
                        score=float(similarity),
                        chunk=Chunk(
                            target_size=1,
                            content=chunk_row.content,
                            index=chunk_row.chunk_index,
                            metadata=chunk_row.chunk_metadata,
                        ),
                    )
                )

            return ChunkResults(
                results=chunk_results,
                total_count=total_count,
                page=page,
                page_size=page_size,
            )

    @classmethod
    def create(
        cls,
        db_name: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        chunker: Optional[Chunker] = None,
    ):
        config.set_override('DB_NAME', db_name)
        return DBFileHandler(config, embedder, chunker)
