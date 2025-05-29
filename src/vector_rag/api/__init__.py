from typing import List, Optional
from vector_rag import config, Config
from vector_rag.model import ChunkResults
from vector_rag.db.db_file_handler import DBFileHandler

class VectorRAGAPI:
    """Simplified API overlay for vector RAG search operations."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.handler = DBFileHandler(config=self.config)

    def query(self,
              project_id: int,
              file_id: int = None,
              query_text: str = None,
              metadata_filter: Optional[dict] = None
    ) -> ChunkResults:
        return self.handler.query(project_id, file_id, query_text, metadata_filter)

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

        return self.handler.search_chunks_by_embedding(
            project_id=project_id,
            embedding=embedding,
            page=page,
            page_size=page_size,
            similarity_threshold=similarity_threshold,
            file_id=file_id,
            metadata_filter=metadata_filter
        )

    def search_bm25(
        self,
        project_id: int,
        query_text: str,
        page: int = 1,
        page_size: int = 10,
        rank_threshold: float = 0.0,
        file_id: int = None,
        metadata_filter: Optional[dict] = None
    ) -> ChunkResults:
        """
        Search using PostgreSQL full-text search (BM25-style).
        
        Args:
            project_id: ID of the project to search within
            query_text: Text to search for
            page: Page number for paginated results
            page_size: Number of results per page
            rank_threshold: Minimum BM25 rank score
            file_id: Optional file ID to search within
            metadata_filter: Optional metadata filters
            
        Returns:
            ChunkResults containing BM25-ranked chunks
        """
        return self.handler.search_chunks_by_bm25(
            project_id=project_id,
            query_text=query_text,
            page=page,
            page_size=page_size,
            rank_threshold=rank_threshold,
            file_id=file_id,
            metadata_filter=metadata_filter
        )

    def search_hybrid(
        self,
        project_id: int,
        query_text: str,
        page: int = 1,
        page_size: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        similarity_threshold: float = 0.0,
        rank_threshold: float = 0.0,
        file_id: int = None,
        metadata_filter: Optional[dict] = None
    ) -> ChunkResults:
        """
        Hybrid search combining vector and BM25 scores.
        
        Args:
            project_id: ID of the project
            query_text: Query text
            page: Page number
            page_size: Results per page
            vector_weight: Weight for vector similarity (0-1)
            bm25_weight: Weight for BM25 score (0-1)
            similarity_threshold: Min vector similarity
            rank_threshold: Min BM25 rank
            file_id: Optional file ID
            metadata_filter: Optional metadata filters
            
        Returns:
            ChunkResults with hybrid-ranked chunks
        """
        return self.handler.search_chunks_hybrid(
            project_id=project_id,
            query_text=query_text,
            page=page,
            page_size=page_size,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            similarity_threshold=similarity_threshold,
            rank_threshold=rank_threshold,
            file_id=file_id,
            metadata_filter=metadata_filter
        )
