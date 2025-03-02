# Vector RAG Search API

The Vector RAG system provides two main search methods for querying your indexed content:

## `search_text(project_id, query_text, ...)`

Use this method when you have raw text queries. It handles:
1. Converting the text to an embedding using the configured embedder
2. Performing the vector similarity search
3. Returning ranked results

### When to use:
- Building text-based search interfaces
- Simple integration with text input
- When you don't need to manage embeddings

### Example:
```python
results = api.search_text(
    project_id=1,
    query_text="machine learning concepts",
    page=1,
    page_size=10
)
```

## `search_embedding(project_id, embedding, ...)`

Use this method when you have pre-computed embeddings. It:
1. Takes a raw embedding vector as input
2. Performs the vector similarity search
3. Returns ranked results

### When to use:
- You already have pre-computed embeddings
- You want to cache embeddings for repeated searches
- Integrating with other systems that provide embeddings
- When you need more control over the embedding process

### Example:
```python
embedding = [0.12, 0.34, 0.56, ...]  # Your pre-computed vector
results = api.search_embedding(
    project_id=1,
    embedding=embedding,
    page=1,
    page_size=10
)
```

## Common Parameters

Both methods share these parameters:

- `project_id`: ID of the project to search within
- `page`: Page number for paginated results (1-based)
- `page_size`: Number of results per page
- `similarity_threshold`: Minimum similarity score (0.0 to 1.0)
- `metadata_filter`: Optional dictionary of metadata key-value pairs to filter results
    - Example: `{"source": "manual", "category": "technical"}`

## Metadata Filtering

The search API supports filtering results by metadata using Postgres' JSONB capabilities. The metadata filter is a dictionary where:
- Keys are metadata field names
- Values are the exact values to match

Example usage:
```python
# Search for technical documents from manual sources
results = api.search_text(
    project_id=1,
    query_text="machine learning",
    metadata_filter={
        "source": "manual",
        "category": "technical"
    }
)
```

## Performance Considerations

- `search_text` has slightly more overhead due to embedding generation
- `search_embedding` is more efficient when you can reuse embeddings
- For repeated searches of the same text, consider:
  1. Generate embedding once
  2. Cache it
  3. Use `search_embedding` for subsequent queries

## Choosing the Right Method

| Feature               | search_text | search_embedding |
|-----------------------|-------------|------------------|
| Input Type            | Text        | Embedding Vector |
| Embedding Generation  | Automatic   | Manual           |
| Performance           | Slightly slower | Faster        |
| Use Case              | Simple text search | Advanced/cached searches |
