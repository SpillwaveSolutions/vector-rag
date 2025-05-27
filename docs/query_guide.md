# Vector RAG Complete Query Guide: Text, Semantic Search, and Metadata Filtering

The Vector RAG system provides three powerful methods for querying your chunked documents. This guide covers all query methods with practical examples, performance tips, and real-world use cases.

## Table of Contents
- [Quick Start](#quick-start)
- [Query Methods Overview](#query-methods-overview)
- [Method 1: Direct Text Search](#method-1-direct-text-search-with-query)
- [Method 2: Semantic Text Search](#method-2-semantic-search-with-search_chunks_by_text)
- [Method 3: Direct Embedding Search](#method-3-direct-embedding-search-with-search_chunks_by_embedding)
- [Advanced Metadata Filtering](#advanced-metadata-filtering)
- [Performance Guide](#performance-guide)
- [Common Patterns & Best Practices](#common-patterns--best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

```python
from vector_rag.db import DBFileHandler

# Initialize handler
handler = DBFileHandler(config)

# Method 1: Exact text search (fast, no pagination)
exact_results = handler.query(
    project_id=1,
    query_text="machine learning"
)

# Method 2: Semantic search (finds related content)
semantic_results = handler.search_chunks_by_text(
    project_id=1,
    query_text="AI algorithms",
    page=1,
    page_size=10,
    similarity_threshold=0.7
)

# Method 3: Pre-computed embedding search (fastest semantic)
embedding_results = handler.search_chunks_by_embedding(
    project_id=1,
    embedding=my_embedding_vector,
    similarity_threshold=0.8
)
```

## Query Methods Overview

| Method | Type | Speed | Use Case | Pagination | Similarity Score |
|--------|------|-------|----------|------------|------------------|
| `query()` | SQL ILIKE | ⚡⚡⚡ Fastest | Exact/substring matches | ❌ No | ❌ No |
| `search_chunks_by_text()` | Vector similarity | ⚡ Slowest | Semantic understanding | ✅ Yes | ✅ Yes |
| `search_chunks_by_embedding()` | Vector similarity | ⚡⚡ Fast | Cached/batch queries | ✅ Yes | ✅ Yes |

### Decision Tree
```
Need exact phrase matching? → Use query()
    ↓ No
Need semantic understanding? → Use search_chunks_by_text()
    ↓ No
Have pre-computed embeddings? → Use search_chunks_by_embedding()
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
| `search_chunks_by_text()` | 50-100ms | 100-200ms | 200-500ms | 0.5-2s |
| `search_chunks_by_embedding()` | 20-50ms | 50-100ms | 100-300ms | 0.3-1s |

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

### 1. Hybrid Search Pattern
```python
class HybridSearch:
    """Combines exact and semantic search for best results."""
    
    def search(self, handler, project_id, query, max_results=20):
        # Step 1: Try exact match first
        exact_results = handler.query(
            project_id=project_id,
            query_text=query
        )
        
        if exact_results.total_count >= max_results:
            # Enough exact matches
            return exact_results.chunks[:max_results]
        
        # Step 2: Augment with semantic search
        semantic_results = handler.search_chunks_by_text(
            project_id=project_id,
            query_text=query,
            page_size=max_results - exact_results.total_count,
            similarity_threshold=0.7
        )
        
        # Combine and deduplicate
        all_chunks = exact_results.chunks + semantic_results.chunks
        seen_ids = set()
        unique_chunks = []
        
        for chunk in all_chunks:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)
        
        return unique_chunks[:max_results]
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
# Exact substring search (no pagination)
handler.query(project_id=1, query_text="exact phrase")

# Semantic search with auto-embedding (with pagination)
handler.search_chunks_by_text(
    project_id=1, 
    query_text="conceptual search",
    page=1, 
    page_size=10,
    similarity_threshold=0.7
)

# Pre-computed embedding search (with pagination)
handler.search_chunks_by_embedding(
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
- **Semantic meaning?** → `search_chunks_by_text()`
- **Have embeddings?** → `search_chunks_by_embedding()`
- **Need speed?** → Cache embeddings, use `search_chunks_by_embedding()`
- **Building search UI?** → Start with `search_chunks_by_text()`, add caching later

### Remember
1. 🎯 Choose the right method for your use case
2. 📊 Use appropriate similarity thresholds (0.7 default)
3. 🏃‍♂️ Cache embeddings for repeated searches
4. 🔍 Leverage metadata filters to narrow results
5. 📈 Monitor performance and adjust strategies
6. 🛡️ Always implement error handling
7. 📝 Track search analytics to improve over time