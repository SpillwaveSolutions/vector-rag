"""
Demo script showing BM25 and hybrid search capabilities.

This script demonstrates:
1. Traditional vector similarity search
2. BM25 (keyword/lexical) search
3. Hybrid search combining both approaches
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_rag.api import VectorRAGAPI
from vector_rag.config import Config
from vector_rag.db.db_file_handler import DBFileHandler
from vector_rag.model import Project, File, Chunk


def setup_demo_data(api: VectorRAGAPI):
    """Create sample project and add documents."""
    # Create a project
    handler = api.handler
    project = Project(name="BM25 Demo Project", description="Demo of BM25 and hybrid search")
    project_id = handler.create_project(project)
    
    # Sample documents with technical and non-technical content
    documents = [
        {
            "file": File(
                name="postgresql_fts.txt",
                path="/demo/postgresql_fts.txt",
                content="""PostgreSQL Full Text Search (FTS) with tsvector and tsquery.
                
PostgreSQL provides powerful full-text search capabilities through its tsvector data type.
The to_tsvector function converts text to a searchable format by:
- Lowercasing all words
- Removing stop words like 'the', 'and', 'is'
- Applying stemming (e.g., 'running' becomes 'run')
- Storing word positions for phrase searching

The ts_rank_cd function provides BM25-style ranking using cover density.
This rewards documents where query terms appear close together.
PostgreSQL FTS is highly optimized with GIN indexes for fast searches."""
            ),
            "metadata": {"category": "technical", "topic": "database"}
        },
        {
            "file": File(
                name="bm25_algorithm.txt",
                path="/demo/bm25_algorithm.txt",
                content="""BM25 Algorithm Overview and Implementation Details.

BM25 (Best Matching 25) is a probabilistic ranking function used in information retrieval.
Key components of BM25:
- Term Frequency (TF): How often a term appears in a document
- Inverse Document Frequency (IDF): How rare a term is across all documents
- Document length normalization: Prevents bias toward longer documents

The BM25 formula uses two tuning parameters:
- k1: Controls term frequency saturation (typically 1.2)
- b: Controls length normalization (typically 0.75)

BM25 excels at finding exact keyword matches and technical terminology."""
            ),
            "metadata": {"category": "technical", "topic": "algorithms"}
        },
        {
            "file": File(
                name="vector_embeddings.txt",
                path="/demo/vector_embeddings.txt",
                content="""Understanding Vector Embeddings for Semantic Search.

Vector embeddings transform text into high-dimensional numerical representations.
These vectors capture semantic meaning, allowing similarity comparisons.
Modern embedding models like BERT and GPT understand context and relationships.

Advantages of vector search:
- Finds semantically similar content even with different words
- Handles synonyms and paraphrases naturally
- Works across languages with multilingual models

Limitations include:
- May miss exact technical terms
- Requires significant computational resources
- Results can be less interpretable than keyword matches"""
            ),
            "metadata": {"category": "technical", "topic": "machine_learning"}
        },
        {
            "file": File(
                name="hybrid_search.txt",
                path="/demo/hybrid_search.txt",
                content="""Hybrid Search: Combining Vector and Keyword Approaches.

Hybrid search leverages both semantic understanding and exact matching.
By combining vector similarity scores with BM25 rankings, we get:
- Precise matching for technical terms and identifiers
- Semantic understanding for natural language queries
- Better handling of both short and long queries

Implementation strategies:
- Linear combination: weighted_score = α * vector_score + β * bm25_score
- Reciprocal rank fusion: combines rankings from both methods
- Re-ranking: use one method to filter, another to rank

Hybrid search significantly improves retrieval quality for technical documentation."""
            ),
            "metadata": {"category": "technical", "topic": "search"}
        }
    ]
    
    # Add files to project
    for doc in documents:
        handler.add_file(project_id, doc["file"], metadata=doc["metadata"])
    
    return project_id


def demonstrate_searches(api: VectorRAGAPI, project_id: int):
    """Run different search types and compare results."""
    
    # Test queries
    queries = [
        "PostgreSQL tsvector",  # Exact technical term
        "how to implement full text search",  # Natural language query
        "BM25 ranking algorithm k1 parameter",  # Mix of exact terms and context
        "finding similar documents",  # Semantic query
    ]
    
    print("=" * 80)
    print("SEARCH DEMONSTRATION: Vector vs BM25 vs Hybrid")
    print("=" * 80)
    
    for query in queries:
        print(f"\nQUERY: '{query}'")
        print("-" * 80)
        
        # 1. Vector Search (Semantic)
        print("\n1. VECTOR SEARCH (Semantic):")
        vector_results = api.search_text(
            project_id=project_id,
            query_text=query,
            page_size=3,
            similarity_threshold=0.3
        )
        
        for i, result in enumerate(vector_results.results):
            print(f"   #{i+1} (score: {result.score:.3f})")
            print(f"   File: {result.chunk.metadata.get('_file_name', 'Unknown')}")
            print(f"   Preview: {result.chunk.content[:100]}...")
        
        # 2. BM25 Search (Keyword/Lexical)
        print("\n2. BM25 SEARCH (Keyword):")
        bm25_results = api.search_bm25(
            project_id=project_id,
            query_text=query,
            page_size=3,
            rank_threshold=0.0
        )
        
        for i, result in enumerate(bm25_results.results):
            print(f"   #{i+1} (score: {result.score:.3f})")
            print(f"   File: {result.chunk.metadata.get('_file_name', 'Unknown')}")
            print(f"   Preview: {result.chunk.content[:100]}...")
        
        # 3. Hybrid Search (Combined)
        print("\n3. HYBRID SEARCH (50% Vector + 50% BM25):")
        hybrid_results = api.search_hybrid(
            project_id=project_id,
            query_text=query,
            page_size=3,
            vector_weight=0.5,
            bm25_weight=0.5
        )
        
        for i, result in enumerate(hybrid_results.results):
            print(f"   #{i+1} (score: {result.score:.3f})")
            if '_scores' in result.chunk.metadata:
                scores = result.chunk.metadata['_scores']
                print(f"   Scores - Vector: {scores['vector']:.3f}, BM25: {scores['bm25']:.3f}")
            print(f"   File: {result.chunk.metadata.get('_file_name', 'Unknown')}")
            print(f"   Preview: {result.chunk.content[:100]}...")
        
        print()


def demonstrate_hybrid_tuning(api: VectorRAGAPI, project_id: int):
    """Show how different weight configurations affect results."""
    
    print("\n" + "=" * 80)
    print("HYBRID SEARCH WEIGHT TUNING")
    print("=" * 80)
    
    query = "PostgreSQL full text search implementation"
    
    weight_configs = [
        (1.0, 0.0, "Vector only"),
        (0.7, 0.3, "Vector-heavy"),
        (0.5, 0.5, "Balanced"),
        (0.3, 0.7, "BM25-heavy"),
        (0.0, 1.0, "BM25 only"),
    ]
    
    print(f"\nQuery: '{query}'")
    
    for vector_w, bm25_w, description in weight_configs:
        print(f"\n{description} (Vector: {vector_w}, BM25: {bm25_w}):")
        
        results = api.search_hybrid(
            project_id=project_id,
            query_text=query,
            page_size=2,
            vector_weight=vector_w,
            bm25_weight=bm25_w
        )
        
        for i, result in enumerate(results.results):
            print(f"   #{i+1} Score: {result.score:.3f}")
            if '_scores' in result.chunk.metadata:
                scores = result.chunk.metadata['_scores']
                print(f"       Vector: {scores['vector']:.3f}, BM25: {scores['bm25']:.3f}")


def main():
    """Run the demonstration."""
    # Initialize API
    config = Config()
    api = VectorRAGAPI(config)
    
    print("Setting up demo data...")
    project_id = setup_demo_data(api)
    
    # Run demonstrations
    demonstrate_searches(api, project_id)
    demonstrate_hybrid_tuning(api, project_id)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("- Vector search excels at semantic similarity and paraphrasing")
    print("- BM25 search is superior for exact keyword matching")
    print("- Hybrid search combines the best of both approaches")
    print("- Weight tuning allows optimization for specific use cases")


if __name__ == "__main__":
    main()