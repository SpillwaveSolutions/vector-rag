# BM25 and Hybrid Search Documentation

## Overview

This document describes the BM25 (lexical/keyword) search and hybrid search capabilities added to the vector-rag system. These features complement the existing vector similarity search by providing exact keyword matching and combined search strategies.

## Background

### Why BM25?

While vector embeddings excel at capturing semantic similarity, they can miss exact keyword matches that are crucial for:
- Technical documentation with specific terminology
- Code samples with exact function/variable names
- Regulatory text where specific words matter
- Searching for acronyms or identifiers

### What is BM25?

BM25 (Best Matching 25) is a probabilistic ranking function that scores documents based on:
- **Term Frequency (TF)**: How often query terms appear in a document
- **Inverse Document Frequency (IDF)**: How rare terms are across all documents
- **Length Normalization**: Prevents bias toward longer documents

PostgreSQL implements BM25-style ranking through its Full-Text Search (FTS) features:
- `tsvector`: Stores preprocessed searchable text
- `ts_rank_cd()`: Provides cover density ranking similar to BM25

## Implementation Details

### Database Schema Changes

Added a `tsvector` column to the `chunks` table:

```sql
ALTER TABLE chunks 
ADD COLUMN content_tsv tsvector 
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX idx_chunks_content_tsv 
ON chunks USING GIN (content_tsv);
```

The `tsvector` column:
- Automatically generated from the `content` column
- Lowercases text and removes stop words
- Applies stemming (e.g., "running" → "run")
- Indexed with GIN for fast searches

### API Methods

#### BM25 Search

```python
from vector_rag.api import VectorRAGAPI

api = VectorRAGAPI()
results = api.search_bm25(
    project_id=1,
    query_text="PostgreSQL tsvector",
    page=1,
    page_size=10,
    rank_threshold=0.0,  # Minimum BM25 score
    file_id=None,        # Optional: search specific file
    metadata_filter={"category": "technical"}  # Optional filters
)
```

#### Hybrid Search

```python
results = api.search_hybrid(
    project_id=1,
    query_text="implement full text search",
    page=1,
    page_size=10,
    vector_weight=0.5,   # Weight for semantic similarity (0-1)
    bm25_weight=0.5,     # Weight for BM25 score (0-1)
    similarity_threshold=0.0,  # Min vector similarity
    rank_threshold=0.0,        # Min BM25 rank
    file_id=None,
    metadata_filter=None
)

# Access individual scores
for result in results.results:
    scores = result.chunk.metadata.get('_scores', {})
    print(f"Vector: {scores.get('vector', 0)}")
    print(f"BM25: {scores.get('bm25', 0)}")
    print(f"Hybrid: {result.score}")
```

## Usage Examples

### 1. Finding Exact Technical Terms

```python
# BM25 excels at finding exact matches
results = api.search_bm25(
    project_id=project_id,
    query_text="ts_rank_cd PostgreSQL",
    page_size=5
)
```

### 2. Semantic Search with Keyword Backup

```python
# Hybrid search with vector-heavy weighting
results = api.search_hybrid(
    project_id=project_id,
    query_text="how to implement search functionality",
    vector_weight=0.7,  # 70% semantic
    bm25_weight=0.3,    # 30% keyword
)
```

### 3. Keyword-First Search

```python
# Hybrid search with BM25-heavy weighting
results = api.search_hybrid(
    project_id=project_id,
    query_text="BM25 k1 parameter tuning",
    vector_weight=0.3,  # 30% semantic
    bm25_weight=0.7,    # 70% keyword
)
```

## Weight Tuning Guidelines

### Vector-Heavy (0.7-0.9 vector weight)
Best for:
- Natural language queries
- Conceptual searches
- Finding related content
- Handling synonyms and paraphrases

### Balanced (0.5 vector, 0.5 BM25)
Best for:
- General-purpose search
- Mixed technical/natural language
- Default configuration

### BM25-Heavy (0.7-0.9 BM25 weight)
Best for:
- Technical documentation
- Code search
- Exact term matching
- Acronyms and identifiers

## Performance Considerations

1. **Index Maintenance**: The GIN index on `content_tsv` is automatically maintained but adds overhead during inserts/updates.

2. **Query Performance**: 
   - BM25 searches use the GIN index and are very fast
   - Hybrid searches perform both vector and text searches in parallel

3. **Storage**: The `tsvector` column adds ~30-50% storage overhead depending on content.

## Migration Guide

To enable BM25 search on existing databases:

1. Run the migration script:
   ```bash
   psql -d your_database -f db/sql/add_fts_support.sql
   ```

2. The tsvector column will be automatically populated for existing chunks.

3. New chunks will have their tsvector generated on insert.

## Troubleshooting

### Common Issues

1. **"column content_tsv does not exist"**
   - Run the migration script to add the tsvector column

2. **Poor BM25 results**
   - Check that content is in the expected language (default: English)
   - Consider adjusting the rank threshold
   - Verify the GIN index exists

3. **Hybrid search not finding expected results**
   - Adjust the weight balance
   - Check both similarity and rank thresholds
   - Ensure at least one weight is non-zero

## Future Enhancements

Potential improvements to consider:

1. **Multi-language support**: Use language-specific text search configurations
2. **Custom dictionaries**: Add domain-specific terms
3. **Phrase search**: Utilize tsvector position information
4. **Query expansion**: Automatically add synonyms to queries
5. **Learning to rank**: Use ML to optimize weight combinations

## References

- [PostgreSQL Full Text Search Documentation](https://www.postgresql.org/docs/current/textsearch.html)
- [BM25 Algorithm Overview](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Hybrid Search Best Practices](https://www.pinecone.io/learn/hybrid-search/)