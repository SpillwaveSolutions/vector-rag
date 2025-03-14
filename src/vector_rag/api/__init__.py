from typing import List, Optional
from vector_rag import config
from vector_rag.model import ChunkResults
from vector_rag.db.db_file_handler import DBFileHandler

class VectorRAGAPI:
    """Simplified API overlay for vector RAG search operations."""
    
    def __init__(self):
        self.config = config
        self.handler = DBFileHandler(config=self.config)
        
    def search_text(
        self,
        project_id: int,
        query_text: str,
        page: int = 1,
        page_size: int = 10,
        similarity_threshold: float = 0.7,
        file_id: int = None,
        metadata_filter: Optional[dict] = None
    ) -> ChunkResults:
        """
        Search for text chunks similar to the query text with optional metadata filtering.
        
        Args:
            project_id: ID of the project to search within
            query_text: Text to search for
            page: Page number for paginated results
            page_size: Number of results per page
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            file_id: ID of the file to search within
            metadata_filter: Optional dictionary of metadata key-value pairs to filter by
            
        Returns:
            ChunkResults containing matching chunks and metadata
        """
        return self.handler.search_chunks_by_text(
            project_id=project_id,
            query_text=query_text,
            page=page,
            page_size=page_size,
            similarity_threshold=similarity_threshold,
            file_id=file_id,
            metadata_filter=metadata_filter
        )
        
    def search_embedding(
        self,
        project_id: int,
        embedding: List[float],
        page: int = 1,
        page_size: int = 10,
        similarity_threshold: float = 0.7,
        file_id: int = None,
        metadata_filter: Optional[dict] = None
    ) -> ChunkResults:
        """
        Search for text chunks similar to the provided embedding vector with optional metadata filtering.
        
        Args:
            project_id: ID of the project to search within
            embedding: Embedding vector to search with
            page: Page number for paginated results
            page_size: Number of results per page
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            file_id: ID of the file to search within
            metadata_filter: Optional dictionary of metadata key-value pairs to filter by
            
        Returns:
            ChunkResults containing matching chunks and metadata
        """
        return self.handler.search_chunks_by_embeddng(
            project_id=project_id,
            embedding=embedding,
            page=page,
            page_size=page_size,
            similarity_threshold=similarity_threshold,
            file_id=file_id,
            metadata_filter=metadata_filter
        )
