# Vector RAG Complete Query Guide: Text, Semantic, BM25, and Hybrid Search

The Vector RAG system provides five powerful methods for querying your chunked documents. This guide covers all query methods with practical examples, performance tips, and real-world use cases.

## Table of Contents
- [Quick Start](#quick-start)
- [Query Methods Overview](#query-methods-overview)
- [Method 1: Direct Text Search](#method-1-direct-text-search-with-query)
- [Method 2: Semantic Text Search](#method-2-semantic-search-with-search_chunks_by_text)
- [Method 3: Direct Embedding Search](#method-3-direct-embedding-search-with-search_chunks_by_embedding)
- [Method 4: BM25 Full-Text Search](#method-4-bm25-full-text-search)
- [Method 5: Hybrid Search](#method-5-hybrid-search)
- [Advanced Metadata Filtering](#advanced-metadata-filtering)
- [Performance Guide](#performance-guide)
- [Common Patterns & Best Practices](#common-patterns--best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

```python
from vector_rag.api import VectorRAGAPI

# Initialize API
api = VectorRAGAPI()

# Method 1: Exact text search (fast, no pagination)
exact_results = api.query(
    project_id=1,
    query_text="machine learning"
)

# Method 2: Semantic search (finds related content)
semantic_results = api.search_text(
    project_id=1,
    query_text="AI algorithms",
    page=1,
    page_size=10,
    similarity_threshold=0.7
)

# Method 3: Pre-computed embedding search (fastest semantic)
embedding_results = api.search_embedding(
    project_id=1,
    embedding=my_embedding_vector,
    similarity_threshold=0.8
)

# Method 4: BM25 search (keyword/lexical matching)
bm25_results = api.search_bm25(
    project_id=1,
    query_text="PostgreSQL tsvector",
    page=1,
    page_size=10
)

# Method 5: Hybrid search (combines vector + BM25)
hybrid_results = api.search_hybrid(
    project_id=1,
    query_text="implement OAuth authentication",
    vector_weight=0.6,
    bm25_weight=0.4
)
```

## Query Methods Overview

| Method | Type | Speed | Use Case | Pagination | Similarity Score |
|--------|------|-------|----------|------------|------------------|
| `query()` | SQL ILIKE | ⚡⚡⚡ Fastest | Exact/substring matches | ❌ No | ❌ No |
| `search_text()` | Vector similarity | ⚡ Slowest | Semantic understanding | ✅ Yes | ✅ Yes |
| `search_embedding()` | Vector similarity | ⚡⚡ Fast | Cached/batch queries | ✅ Yes | ✅ Yes |
| `search_bm25()` | PostgreSQL FTS | ⚡⚡⚡ Very Fast | Keyword/technical terms | ✅ Yes | ✅ Yes (BM25) |
| `search_hybrid()` | Vector + BM25 | ⚡⚡ Fast | Best of both worlds | ✅ Yes | ✅ Yes (Combined) |

### Decision Tree
```
Need exact phrase matching? → Use query()
    ↓ No
Need exact keyword matching? → Use search_bm25()
    ↓ No
Need semantic understanding? → Use search_text()
    ↓ No
Need both keyword + semantic? → Use search_hybrid()
    ↓ No
Have pre-computed embeddings? → Use search_embedding()
```

## Method 1: Direct Text Search with `query()`

### When to Use
- 🎯 Searching for exact phrases, codes, or identifiers
- 🔍 Case-insensitive substring matching needed
- 📊 Need ALL results without pagination
- ⚡ Speed is critical

### Syntax
```python
def query(self,
          project_id: int,
          file_id: int = None,
          query_text: str = None,
          metadata_filter: Optional[dict] = None
          ) -> ChunkResults
```

### Real-World Examples

#### Example 1: Finding Error Codes
```python
# Search for specific error patterns
error_results = handler.query(
    project_id=1,
    query_text="ERROR_",
    metadata_filter={"log_level": "error"}
)

# Process all error chunks
for chunk in error_results.chunks:
    print(f"Found error in {chunk.file_name}: {chunk.content[:100]}...")
```

#### Example 2: Configuration Search
```python
# Find all configuration settings
config_results = handler.query(
    project_id=1,
    query_text="config.",
    metadata_filter={
        "file_type": "yaml",
        "environment": ["prod", "staging"]
    }
)
```

#### Example 3: Multi-Language Code Search
```python
# Find function definitions across languages
function_results = handler.query(
    project_id=1,
    query_text="def ",  # Python functions
    metadata_filter={"language": "python"}
)

# JavaScript/TypeScript functions
js_results = handler.query(
    project_id=1,
    query_text="function ",
    metadata_filter={"language": ["javascript", "typescript"]}
)
```

### ⚠️ Limitations
- No semantic understanding (won't find synonyms)
- Returns ALL matches (could be memory intensive)
- No relevance ranking

## Method 2: Semantic Search with `search_chunks_by_text()`

### When to Use
- 🧠 Need conceptually similar content
- 📄 Building user-facing search interfaces
- 🎨 Working with natural language queries
- 📊 Need ranked results with pagination

### Syntax
```python
def search_chunks_by_text(
    self,
    project_id: int,
    query_text: str,
    page: int = 1,
    page_size: int = 10,
    similarity_threshold: float = 0.7,
    file_id: int = None,
    metadata_filter: Optional[dict] = None,
) -> ChunkResults
```

### Real-World Examples

#### Example 1: Documentation Search
```python
# Find relevant documentation
docs = handler.search_chunks_by_text(
    project_id=1,
    query_text="how to deploy microservices to kubernetes",
    metadata_filter={
        "doc_type": "tutorial",
        "tags": ["deployment", "k8s", "microservices"]
    },
    page=1,
    page_size=5,
    similarity_threshold=0.75
)

# Display results with scores
for i, chunk in enumerate(docs.chunks, 1):
    print(f"{i}. [{chunk.similarity_score:.2f}] {chunk.file_name}")
    print(f"   {chunk.content[:150]}...")
```

#### Example 2: Multi-Page Results with Progress
```python
def search_all_pages(handler, project_id, query, threshold=0.7):
    """Search through all pages and collect results."""
    all_chunks = []
    page = 1
    
    while True:
        results = handler.search_chunks_by_text(
            project_id=project_id,
            query_text=query,
            page=page,
            page_size=50,
            similarity_threshold=threshold
        )
        
        all_chunks.extend(results.chunks)
        print(f"Page {page}/{results.total_pages}: Found {len(results.chunks)} chunks")
        
        if not results.has_next:
            break
        page += 1
    
    return all_chunks

# Usage
all_results = search_all_pages(handler, 1, "machine learning best practices")
```

#### Example 3: Adaptive Threshold Search
```python
def adaptive_search(handler, project_id, query, target_results=10):
    """Automatically adjust threshold to get desired number of results."""
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    
    for threshold in thresholds:
        results = handler.search_chunks_by_text(
            project_id=project_id,
            query_text=query,
            page=1,
            page_size=target_results,
            similarity_threshold=threshold
        )
        
        if results.total_count >= target_results:
            print(f"Found {results.total_count} results at threshold {threshold}")
            return results
    
    print(f"Only found {results.total_count} results even at threshold 0.5")
    return results
```

### 🎯 Similarity Threshold Guide

| Threshold | Description | Use Case |
|-----------|-------------|----------|
| 0.95+ | Nearly identical | Duplicate detection |
| 0.9-0.95 | Extremely similar | Finding variations of same content |
| 0.8-0.9 | Very similar | High-precision search |
| 0.7-0.8 | Related (default) | Balanced precision/recall |
| 0.6-0.7 | Loosely related | Exploratory search |
| <0.6 | Weak connection | Not recommended |

## Method 3: Direct Embedding Search with `search_chunks_by_embedding()`

### When to Use
- 🚀 Performance is critical
- 🔄 Repeated searches with same query
- 🤖 Building recommendation systems
- 📦 Batch processing queries

### Syntax
```python
def search_chunks_by_embedding(
    self,
    project_id: int,
    embedding: Union[np.ndarray, Sequence[float]],
    page: int = 1,
    page_size: int = 10,
    similarity_threshold: float = 0.7,
    file_id: int = None,
    metadata_filter: Optional[dict] = None,
) -> ChunkResults
```

### Real-World Examples

#### Example 1: Embedding Cache System
```python
class EmbeddingSearchCache:
    def __init__(self, handler, embedder):
        self.handler = handler
        self.embedder = embedder
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def search(self, project_id, query, **kwargs):
        # Check cache
        cache_key = f"{project_id}:{query}"
        
        if cache_key in self.cache:
            self.cache_hits += 1
            embedding = self.cache[cache_key]
        else:
            self.cache_misses += 1
            # Generate embedding
            chunks = [Chunk(content=query, index=0)]
            embedding = self.embedder.embed_texts(chunks)[0]
            self.cache[cache_key] = embedding
        
        # Search with embedding
        return self.handler.search_chunks_by_embedding(
            project_id=project_id,
            embedding=embedding,
            **kwargs
        )
    
    def stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "hit_rate": f"{hit_rate:.2%}",
            "hits": self.cache_hits,
            "misses": self.cache_misses
        }
```

#### Example 2: Batch Processing Pipeline
```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def batch_search_pipeline(handler, project_id, queries, embedder):
    """Process multiple queries efficiently."""
    
    # Step 1: Generate all embeddings in batch
    print("Generating embeddings...")
    chunks = [Chunk(content=q, index=i) for i, q in enumerate(queries)]
    embeddings = embedder.embed_texts(chunks)
    
    # Step 2: Search in parallel
    print("Searching...")
    results = {}
    
    def search_single(query_embedding_pair):
        query, embedding = query_embedding_pair
        return query, handler.search_chunks_by_embedding(
            project_id=project_id,
            embedding=embedding,
            page_size=5,
            similarity_threshold=0.8
        )
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = executor.map(search_single, zip(queries, embeddings))
        results = dict(futures)
    
    return results

# Usage
queries = [
    "machine learning optimization",
    "neural network architectures",
    "data preprocessing techniques"
]
results = batch_search_pipeline(handler, 1, queries, embedder)
```

#### Example 3: Document Similarity Matrix
```python
def build_similarity_matrix(handler, project_id, document_ids, embedder):
    """Build a similarity matrix between documents."""
    
    # Get embeddings for all documents
    doc_embeddings = {}
    for doc_id in document_ids:
        # Get document's average embedding
        doc_chunks = handler.query(
            project_id=project_id,
            file_id=doc_id
        )
        
        if doc_chunks.chunks:
            # Average all chunk embeddings
            embeddings = embedder.embed_texts(doc_chunks.chunks)
            doc_embeddings[doc_id] = np.mean(embeddings, axis=0)
    
    # Find similar documents for each
    similarity_matrix = {}
    
    for doc_id, embedding in doc_embeddings.items():
        similar_docs = handler.search_chunks_by_embedding(
            project_id=project_id,
            embedding=embedding,
            page_size=10,
            similarity_threshold=0.7,
            metadata_filter={"file_id": {"$ne": doc_id}}  # Exclude self
        )
        
        similarity_matrix[doc_id] = [
            (chunk.file_id, chunk.similarity_score) 
            for chunk in similar_docs.chunks
        ]
    
    return similarity_matrix
```

## Method 4: BM25 Full-Text Search

### When to Use
- 🔤 Searching for exact technical terms or keywords
- 📝 Code search (function names, variables, imports)
- 🏷️ Finding specific identifiers or acronyms
- ⚡ Need fast keyword-based retrieval
- 🎯 When exact terms matter more than semantic similarity

### Syntax
```python
def search_bm25(
    self,
    project_id: int,
    query_text: str,
    page: int = 1,
    page_size: int = 10,
    rank_threshold: float = 0.0,
    file_id: int = None,
    metadata_filter: Optional[dict] = None
) -> ChunkResults
```

### Real-World Examples

#### Example 1: Technical Documentation Search
```python
# Find PostgreSQL-specific documentation
pg_docs = api.search_bm25(
    project_id=1,
    query_text="PostgreSQL tsvector GIN index",
    page=1,
    page_size=10,
    metadata_filter={"doc_type": "technical"}
)

# BM25 excels at finding exact technical terminology
for result in pg_docs.results:
    print(f"Rank: {result.score:.3f} - {result.chunk.content[:100]}...")
```

#### Example 2: Code Search
```python
# Search for specific function usage
function_usage = api.search_bm25(
    project_id=1,
    query_text="search_chunks_by_embedding embedding parameter",
    metadata_filter={"file_type": "python"}
)

# Find import statements
imports = api.search_bm25(
    project_id=1,
    query_text="from vector_rag import",
    rank_threshold=0.5  # Only high-confidence matches
)
```

#### Example 3: Error and Log Search
```python
# Search for specific error patterns
errors = api.search_bm25(
    project_id=1,
    query_text="ConnectionError database timeout",
    metadata_filter={
        "log_level": ["error", "critical"],
        "service": "database"
    }
)

# Find configuration references
configs = api.search_bm25(
    project_id=1,
    query_text="EMBEDDING_DIM config environment",
    metadata_filter={"file_type": ["yaml", "env", "json"]}
)
```

### BM25 vs Vector Search Comparison

| Query Type | BM25 Result | Vector Result |
|------------|-------------|---------------|
| "ts_rank_cd" | ✅ Finds exact function name | ❌ May miss if not in training data |
| "implement authentication" | ❌ Only if exact words present | ✅ Finds auth-related content |
| "ERROR_CODE_404" | ✅ Exact match | ❌ No semantic meaning |
| "machine learning" | ✅ If exact phrase exists | ✅ Finds ML, AI, neural networks |

## Method 5: Hybrid Search

### When to Use
- 🎯 Need both exact matches AND semantic understanding
- 📚 Technical documentation with natural language
- 🔍 Comprehensive search results
- ⚖️ Want to balance precision and recall
- 🏆 Best overall search quality

### Syntax
```python
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
) -> ChunkResults
```

### Real-World Examples

#### Example 1: Technical Query with Context
```python
# Search for implementation details
results = api.search_hybrid(
    project_id=1,
    query_text="implement BM25 ranking algorithm PostgreSQL",
    vector_weight=0.4,  # 40% semantic
    bm25_weight=0.6,    # 60% keyword (technical focus)
    page_size=20
)

# Results include both exact BM25 references AND 
# conceptually related ranking implementations
for result in results.results:
    scores = result.chunk.metadata.get('_scores', {})
    print(f"Hybrid: {result.score:.3f} (Vector: {scores.get('vector', 0):.3f}, "
          f"BM25: {scores.get('bm25', 0):.3f})")
    print(f"Content: {result.chunk.content[:150]}...\n")
```

#### Example 2: Dynamic Weight Adjustment
```python
def adaptive_hybrid_search(api, project_id, query):
    """Adjust weights based on query characteristics."""
    
    # Analyze query
    technical_terms = ["API", "SQL", "HTTP", "JSON", "OAuth"]
    has_technical = any(term in query.upper() for term in technical_terms)
    
    # Longer queries often benefit from semantic search
    query_length = len(query.split())
    
    if has_technical and query_length < 5:
        # Short technical query - favor BM25
        vector_weight, bm25_weight = 0.3, 0.7
    elif query_length > 10:
        # Long natural language query - favor vector
        vector_weight, bm25_weight = 0.7, 0.3
    else:
        # Balanced approach
        vector_weight, bm25_weight = 0.5, 0.5
    
    print(f"Query: '{query}'")
    print(f"Weights: Vector={vector_weight}, BM25={bm25_weight}")
    
    return api.search_hybrid(
        project_id=project_id,
        query_text=query,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight
    )
```

#### Example 3: Hybrid Search Pipeline
```python
class HybridSearchPipeline:
    """Advanced hybrid search with fallback strategies."""
    
    def __init__(self, api):
        self.api = api
        
    def search(self, project_id, query, min_results=5):
        strategies = [
            # Start with balanced hybrid
            {"vector": 0.5, "bm25": 0.5, "name": "Balanced"},
            
            # Try keyword-heavy if few results
            {"vector": 0.3, "bm25": 0.7, "name": "Keyword-focused"},
            
            # Try semantic-heavy as fallback
            {"vector": 0.7, "bm25": 0.3, "name": "Semantic-focused"},
            
            # Pure strategies as last resort
            {"vector": 1.0, "bm25": 0.0, "name": "Pure Vector"},
            {"vector": 0.0, "bm25": 1.0, "name": "Pure BM25"}
        ]
        
        for strategy in strategies:
            results = self.api.search_hybrid(
                project_id=project_id,
                query_text=query,
                vector_weight=strategy["vector"],
                bm25_weight=strategy["bm25"],
                page_size=min_results * 2  # Get extra for filtering
            )
            
            print(f"{strategy['name']}: {results.total_count} results")
            
            if results.total_count >= min_results:
                return results
                
        return results  # Return last attempt
```

### Weight Configuration Guide

| Use Case | Vector Weight | BM25 Weight | Example |
|----------|---------------|-------------|---------|
| Technical Docs | 0.3-0.4 | 0.6-0.7 | API references, code docs |
| Natural Language | 0.6-0.8 | 0.2-0.4 | Tutorials, guides |
| Balanced | 0.5 | 0.5 | General search |
| Code Search | 0.2 | 0.8 | Function names, syntax |
| Conceptual | 0.8 | 0.2 | "How to" queries |

### Performance Considerations

The hybrid search performs both vector and BM25 searches in parallel, then combines scores. Consider:

1. **Pre-filtering**: Use metadata filters to reduce search space
2. **Weight caching**: Cache optimal weights for common query patterns
3. **Threshold tuning**: Set appropriate thresholds for each component

```python
# Efficient hybrid search with pre-filtering
results = api.search_hybrid(
    project_id=1,
    query_text="OAuth implementation",
    vector_weight=0.6,
    bm25_weight=0.4,
    similarity_threshold=0.5,  # Minimum vector similarity
    rank_threshold=0.1,        # Minimum BM25 rank
    metadata_filter={
        "date_created": {"$gte": "2024-01-01"},
        "status": "published"
    }
)
```

### 📚 Learn More

For detailed information about BM25 and hybrid search implementation, including:
- PostgreSQL FTS configuration
- Performance benchmarks
- Advanced tuning strategies
- Migration guides

See the comprehensive [BM25 and Hybrid Search Documentation](./bm25_hybrid_search.md).

## Advanced Metadata Filtering

### Filter Types and Examples

#### 1. Basic Filters
```python
# Single value
metadata_filter = {"status": "published"}

# Multiple values (OR condition)
metadata_filter = {"category": ["tech", "science", "engineering"]}

# Numeric values
metadata_filter = {"year": 2024, "priority": [1, 2, 3]}
```

#### 2. Nested Object Filters
```python
# Direct nesting
metadata_filter = {
    "author": {
        "name": "Jane Doe",
        "credentials": ["PhD", "MSc"]
    }
}

# Dot notation
metadata_filter = {
    "publication.journal": "Nature",
    "metrics.citations": {"$gte": 100}
}
```

#### 3. Advanced Operators (if supported)
```python
# Range queries
metadata_filter = {
    "metrics.accuracy": {"$gte": 0.9, "$lte": 0.99},
    "date_published": {"$gte": "2023-01-01", "$lte": "2024-12-31"}
}

# Existence checks
metadata_filter = {
    "peer_reviewed": {"$exists": True},
    "deprecated": {"$exists": False}
}

# Pattern matching
metadata_filter = {
    "filename": {"$regex": "^test_.*\\.py$"},
    "content_type": {"$in": ["text/plain", "text/markdown"]}
}
```

#### 4. Complex Real-World Filter
```python
# Academic paper search filter
complex_filter = {
    "document_type": "research_paper",
    "publication": {
        "year": [2023, 2024],
        "venue": ["NeurIPS", "ICML", "ICLR"],
        "peer_reviewed": True
    },
    "authors": {
        "$elemMatch": {
            "affiliation": ["MIT", "Stanford", "Berkeley"],
            "h_index": {"$gte": 20}
        }
    },
    "metrics": {
        "citations": {"$gte": 50},
        "impact_factor": {"$gte": 5.0}
    },
    "topics": {
        "$all": ["machine learning", "optimization"]
    }
}
```

### Metadata Filter Builder Pattern
```python
class MetadataFilterBuilder:
    def __init__(self):
        self.filter = {}
    
    def add_basic(self, key, value):
        self.filter[key] = value
        return self
    
    def add_range(self, key, min_val=None, max_val=None):
        range_filter = {}
        if min_val is not None:
            range_filter["$gte"] = min_val
        if max_val is not None:
            range_filter["$lte"] = max_val
        self.filter[key] = range_filter
        return self
    
    def add_nested(self, path, value):
        keys = path.split(".")
        current = self.filter
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        return self
    
    def build(self):
        return self.filter

# Usage
filter_builder = MetadataFilterBuilder()
metadata_filter = (filter_builder
    .add_basic("status", "published")
    .add_range("year", min_val=2023)
    .add_nested("author.institution", "MIT")
    .build())
```

## Performance Guide

### Benchmarks (Approximate)

| Operation | 1K chunks | 10K chunks | 100K chunks | 1M chunks |
|-----------|-----------|------------|-------------|-----------|
| `query()` | <10ms | <50ms | <200ms | <2s |
| `search_text()` | 50-100ms | 100-200ms | 200-500ms | 0.5-2s |
| `search_embedding()` | 20-50ms | 50-100ms | 100-300ms | 0.3-1s |
| `search_bm25()` | <20ms | <60ms | <250ms | <2.5s |
| `search_hybrid()` | 60-120ms | 120-250ms | 300-600ms | 0.6-2.5s |

### Optimization Strategies

#### 1. Query Optimization
```python
# ❌ Inefficient: Multiple separate searches
results1 = handler.query(project_id=1, query_text="error")
results2 = handler.query(project_id=1, query_text="warning")
results3 = handler.query(project_id=1, query_text="info")

# ✅ Efficient: Combined search with post-processing
all_results = handler.query(project_id=1, query_text="")
error_chunks = [c for c in all_results.chunks if "error" in c.content.lower()]
warning_chunks = [c for c in all_results.chunks if "warning" in c.content.lower()]
```

#### 2. Embedding Optimization
```python
# ❌ Inefficient: Generate embedding for each search
for i in range(100):
    results = handler.search_chunks_by_text(
        project_id=1,
        query_text="machine learning",
        page=i+1
    )

# ✅ Efficient: Generate once, reuse embedding
embedding = embedder.embed_texts([Chunk(content="machine learning", index=0)])[0]
for i in range(100):
    results = handler.search_chunks_by_embedding(
        project_id=1,
        embedding=embedding,
        page=i+1
    )
```

#### 3. Metadata Index Optimization
```python
# For frequently filtered fields, ensure they're indexed
# Common indexed fields should include:
indexed_fields = [
    "project_id",
    "file_id", 
    "document_type",
    "status",
    "created_date",
    "category"
]
```

## Common Patterns & Best Practices

### 1. Unified Search Pattern
```python
class UnifiedSearch:
    """Smart search that adapts to query type."""
    
    def search(self, api, project_id, query, max_results=20):
        # Analyze query characteristics
        is_technical = any(word in query.lower() for word in 
                         ['function', 'class', 'error', 'config', 'api'])
        is_question = query.strip().endswith('?')
        word_count = len(query.split())
        
        # Choose search strategy
        if is_technical and word_count < 5:
            # Use BM25 for short technical queries
            return api.search_bm25(
                project_id=project_id,
                query_text=query,
                page_size=max_results
            )
        elif is_question or word_count > 8:
            # Use vector search for questions/long queries
            return api.search_text(
                project_id=project_id,
                query_text=query,
                page_size=max_results,
                similarity_threshold=0.7
            )
        else:
            # Use hybrid for everything else
            return api.search_hybrid(
                project_id=project_id,
                query_text=query,
                page_size=max_results,
                vector_weight=0.5,
                bm25_weight=0.5
            )
```

### 2. Progressive Search Pattern
```python
def progressive_search(handler, project_id, query):
    """Start narrow, progressively broaden search."""
    
    strategies = [
        # Exact match in titles
        {"method": "query", "params": {"query_text": query, "metadata_filter": {"is_title": True}}},
        
        # Exact match anywhere
        {"method": "query", "params": {"query_text": query}},
        
        # High-threshold semantic
        {"method": "search_chunks_by_text", "params": {"query_text": query, "similarity_threshold": 0.85}},
        
        # Normal semantic
        {"method": "search_chunks_by_text", "params": {"query_text": query, "similarity_threshold": 0.7}},
        
        # Broad semantic
        {"method": "search_chunks_by_text", "params": {"query_text": query, "similarity_threshold": 0.6}}
    ]
    
    for strategy in strategies:
        method = getattr(handler, strategy["method"])
        results = method(project_id=project_id, **strategy["params"])
        
        if results.total_count > 0:
            print(f"Found {results.total_count} results using {strategy['method']}")
            return results
    
    return None
```

### 3. Search Analytics Pattern
```python
class SearchAnalytics:
    def __init__(self):
        self.queries = []
    
    def track_search(self, query, method, results_count, duration):
        self.queries.append({
            "timestamp": datetime.now(),
            "query": query,
            "method": method,
            "results_count": results_count,
            "duration": duration
        })
    
    def get_insights(self):
        if not self.queries:
            return {}
        
        df = pd.DataFrame(self.queries)
        
        return {
            "total_searches": len(df),
            "avg_results": df["results_count"].mean(),
            "avg_duration": df["duration"].mean(),
            "popular_queries": df["query"].value_counts().head(10),
            "method_distribution": df["method"].value_counts(),
            "zero_result_queries": df[df["results_count"] == 0]["query"].tolist()
        }

# Usage with timing
import time

analytics = SearchAnalytics()

start = time.time()
results = handler.search_chunks_by_text(project_id=1, query_text="machine learning")
duration = time.time() - start

analytics.track_search("machine learning", "semantic", results.total_count, duration)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. No Results Found
```python
def diagnose_no_results(handler, project_id, query):
    """Diagnose why a query returns no results."""
    
    print(f"Diagnosing query: '{query}'")
    
    # Check if any chunks exist
    all_chunks = handler.query(project_id=project_id, query_text="")
    print(f"Total chunks in project: {all_chunks.total_count}")
    
    if all_chunks.total_count == 0:
        print("❌ No chunks in project!")
        return
    
    # Try progressively looser searches
    tests = [
        ("Exact match", lambda: handler.query(project_id=project_id, query_text=query)),
        ("Semantic 0.9", lambda: handler.search_chunks_by_text(project_id=project_id, query_text=query, similarity_threshold=0.9)),
        ("Semantic 0.7", lambda: handler.search_chunks_by_text(project_id=project_id, query_text=query, similarity_threshold=0.7)),
        ("Semantic 0.5", lambda: handler.search_chunks_by_text(project_id=project_id, query_text=query, similarity_threshold=0.5)),
    ]
    
    for test_name, test_func in tests:
        results = test_func()
        print(f"{test_name}: {results.total_count} results")
        
        if results.total_count > 0:
            print(f"✅ Found results with {test_name}")
            print(f"   Best match score: {results.chunks[0].similarity_score if hasattr(results.chunks[0], 'similarity_score') else 'N/A'}")
            break
```

#### 2. Poor Semantic Search Results
```python
def analyze_semantic_quality(handler, project_id, query, expected_content):
    """Analyze why semantic search isn't finding expected content."""
    
    # Get the expected content's embedding
    expected_chunk = Chunk(content=expected_content, index=0)
    expected_embedding = embedder.embed_texts([expected_chunk])[0]
    
    # Get query embedding
    query_chunk = Chunk(content=query, index=0)
    query_embedding = embedder.embed_texts([query_chunk])[0]
    
    # Calculate similarity
    from numpy import dot
    from numpy.linalg import norm
    
    similarity = dot(query_embedding, expected_embedding) / (norm(query_embedding) * norm(expected_embedding))
    
    print(f"Query: '{query}'")
    print(f"Expected content: '{expected_content[:100]}...'")
    print(f"Similarity score: {similarity:.4f}")
    
    if similarity < 0.7:
        print("❌ Low similarity - consider:")
        print("   - Rephrasing the query")
        print("   - Using more specific terms")
        print("   - Checking if content was properly embedded")
```

#### 3. Performance Issues
```python
class PerformanceMonitor:
    def __init__(self, handler):
        self.handler = handler
        self.original_methods = {}
        self._wrap_methods()
    
    def _wrap_methods(self):
        methods = ['query', 'search_chunks_by_text', 'search_chunks_by_embedding']
        
        for method_name in methods:
            original = getattr(self.handler, method_name)
            
            def wrapped(*args, **kwargs):
                start = time.time()
                result = original(*args, **kwargs)
                duration = time.time() - start
                
                print(f"{method_name} took {duration:.3f}s, returned {result.total_count} results")
                
                if duration > 1.0:
                    print(f"⚠️  Slow query detected! Consider:")
                    print(f"   - Adding metadata filters to narrow scope")
                    print(f"   - Reducing page_size")
                    print(f"   - Using cached embeddings")
                
                return result
            
            setattr(self.handler, method_name, wrapped)

# Usage
monitor = PerformanceMonitor(handler)
# Now all searches will be monitored
results = handler.search_chunks_by_text(project_id=1, query_text="test")
```

### Error Handling Best Practices
```python
def safe_search(handler, project_id, query, method="semantic", **kwargs):
    """Robust search with comprehensive error handling."""
    
    try:
        if method == "exact":
            return handler.query(
                project_id=project_id,
                query_text=query,
                **kwargs
            )
        
        elif method == "semantic":
            return handler.search_chunks_by_text(
                project_id=project_id,
                query_text=query,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    except ValueError as e:
        print(f"Invalid parameters: {e}")
        # Return empty results
        return ChunkResults(chunks=[], total_count=0, page=1, page_size=10)
    
    except ConnectionError as e:
        print(f"Database connection error: {e}")
        # Implement retry logic
        time.sleep(1)
        return safe_search(handler, project_id, query, method, **kwargs)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Log for debugging
        import traceback
        traceback.print_exc()
        
        # Return empty results
        return ChunkResults(chunks=[], total_count=0, page=1, page_size=10)
```

## Summary

### Quick Reference Card

```python
from vector_rag.api import VectorRAGAPI
api = VectorRAGAPI()

# Exact substring search (no pagination)
api.query(project_id=1, query_text="exact phrase")

# Semantic search with auto-embedding (with pagination)
api.search_text(
    project_id=1, 
    query_text="conceptual search",
    page=1, 
    page_size=10,
    similarity_threshold=0.7
)

# BM25 keyword search (with pagination)
api.search_bm25(
    project_id=1,
    query_text="PostgreSQL tsvector",
    page=1,
    page_size=10,
    rank_threshold=0.0
)

# Hybrid search combining vector + BM25
api.search_hybrid(
    project_id=1,
    query_text="implement authentication",
    vector_weight=0.6,
    bm25_weight=0.4,
    page=1,
    page_size=10
)

# Pre-computed embedding search (with pagination)
api.search_embedding(
    project_id=1,
    embedding=my_vector,
    page=1,
    page_size=10,
    similarity_threshold=0.7
)

# All methods support metadata filtering
metadata_filter = {
    "simple": "value",
    "multiple": ["val1", "val2"],
    "nested.path": "value",
    "range": {"$gte": 10, "$lte": 100}
}
```

### Method Selection Guide
- **Exact text?** → `query()`
- **Technical terms/keywords?** → `search_bm25()`
- **Semantic meaning?** → `search_text()`
- **Best of both worlds?** → `search_hybrid()`
- **Have embeddings?** → `search_embedding()`
- **Need speed?** → Cache embeddings or use `search_bm25()`
- **Building search UI?** → Start with `search_hybrid()` for best results

### Remember
1. 🎯 Choose the right method for your use case
2. 📊 Use appropriate similarity thresholds (0.7 default)
3. 🏃‍♂️ Cache embeddings for repeated searches
4. 🔍 Leverage metadata filters to narrow results
5. 📈 Monitor performance and adjust strategies
6. 🛡️ Always implement error handling
7. 📝 Track search analytics to improve over time