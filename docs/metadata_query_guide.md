# Metadata Query Guide

This guide explains how to use the `query` method for searching chunks based on JSONB metadata without using semantic search or BM25 text search.

## Overview

The `query` method provides direct access to chunks using PostgreSQL's JSONB operators with GIN indexes. This is ideal when you need to retrieve chunks based on structured metadata rather than content similarity.

## Method Signature

```python
def query(
    self,
    project_id: int,
    file_id: int = None,
    query_text: str = None,
    metadata_filter: Optional[dict] = None,
    page: int = 1,
    page_size: int = 10
) -> ChunkResults
```

## Parameters

- **project_id** (required): The ID of the project to search within
- **file_id** (optional): Limit search to a specific file
- **query_text** (optional): Simple text search in content (uses ILIKE)
- **metadata_filter** (optional): Dictionary of metadata key-value pairs to filter by
- **page**: Page number for pagination (default: 1)
- **page_size**: Number of results per page (default: 10)

## Example Metadata Structure

Let's assume your chunks have metadata structured like this:

```json
{
    "type": "paragraph",
    "section": "4. Clinical Results",
    "subsection": "4.2 Efficacy Analysis",
    "document": "clinical_trial_2023.pdf",
    "page_number": 47,
    "language": "en",
    "tags": ["efficacy", "primary-endpoint", "statistics"],
    "author": {
        "name": "Dr. Sarah Chen",
        "department": "Biostatistics"
    },
    "review_status": "approved",
    "confidence_score": 0.95
}
```

## Usage Examples

### Basic Metadata Filtering

```python
from vector_rag.db.db_file_handler import DBFileHandler

# Initialize handler
db_handler = DBFileHandler(config, embedder, chunker)

# Find all paragraphs in a specific section
results = db_handler.query(
    project_id=1,
    metadata_filter={
        'type': 'paragraph',
        'section': '4. Clinical Results'
    }
)

# Process results
for result in results.results:
    print(f"Content: {result.chunk.content[:100]}...")
    print(f"Metadata: {result.chunk.metadata}")
```

### Filtering by Multiple Criteria

```python
# Find all approved paragraphs in efficacy analysis
results = db_handler.query(
    project_id=1,
    metadata_filter={
        'type': 'paragraph',
        'subsection': '4.2 Efficacy Analysis',
        'review_status': 'approved'
    },
    page_size=20
)
```

### Working with Nested Metadata

```python
# Filter by nested author information
results = db_handler.query(
    project_id=1,
    metadata_filter={
        'author': {
            'name': 'Dr. Sarah Chen',
            'department': 'Biostatistics'
        }
    }
)

# Alternative: Using dot notation
results = db_handler.query(
    project_id=1,
    metadata_filter={
        'author.name': 'Dr. Sarah Chen'
    }
)
```

### Filtering by Arrays

```python
# Find chunks with specific tags
# Note: This finds chunks where tags array contains "efficacy"
results = db_handler.query(
    project_id=1,
    metadata_filter={
        'tags': ['efficacy']  # Will match if 'efficacy' is in the tags array
    }
)
```

### Numeric Filtering

```python
# Find chunks from specific pages
results = db_handler.query(
    project_id=1,
    metadata_filter={
        'page_number': 47,
        'confidence_score': 0.95
    }
)
```

### Combining with Text Search

```python
# Find paragraphs containing "placebo" in efficacy section
results = db_handler.query(
    project_id=1,
    query_text="placebo",
    metadata_filter={
        'type': 'paragraph',
        'section': '4. Clinical Results'
    }
)
```

### Pagination

```python
# Get all results page by page
page = 1
page_size = 50
all_chunks = []

while True:
    results = db_handler.query(
        project_id=1,
        metadata_filter={'type': 'paragraph'},
        page=page,
        page_size=page_size
    )
    
    all_chunks.extend(results.results)
    
    # Check if we've retrieved all results
    if len(all_chunks) >= results.total_count:
        break
        
    page += 1

print(f"Retrieved {len(all_chunks)} total chunks")
```

### Advanced Use Cases

#### Load All Content from a Document Section

```python
# Get all chunks from a specific document section
def load_document_section(db_handler, project_id, document_name, section):
    results = db_handler.query(
        project_id=project_id,
        metadata_filter={
            'document': document_name,
            'section': section
        },
        page_size=100  # Adjust based on expected section size
    )
    
    # Sort by page number if available
    sorted_chunks = sorted(
        results.results,
        key=lambda x: x.chunk.metadata.get('page_number', 0)
    )
    
    # Concatenate content
    full_text = "\n\n".join([chunk.chunk.content for chunk in sorted_chunks])
    return full_text

# Usage
section_text = load_document_section(
    db_handler, 
    project_id=1,
    document_name="clinical_trial_2023.pdf",
    section="4. Clinical Results"
)
```

#### Filter by Review Status and Department

```python
# Get all approved content from specific department
results = db_handler.query(
    project_id=1,
    metadata_filter={
        'review_status': 'approved',
        'author.department': 'Biostatistics'
    },
    page_size=50
)

# Count by status
for status in ['draft', 'in_review', 'approved']:
    count_results = db_handler.query(
        project_id=1,
        metadata_filter={'review_status': status},
        page_size=1  # We only need the count
    )
    print(f"{status}: {count_results.total_count} chunks")
```

## Performance Tips

1. **Use GIN Indexes**: The system automatically creates GIN indexes on `chunk_metadata` for fast JSONB queries
2. **Be Specific**: More specific queries (more filter criteria) typically perform better
3. **Pagination**: Use appropriate page sizes - larger sizes for bulk operations, smaller for UI display
4. **Avoid Large Arrays**: When filtering by arrays, performance is better with smaller arrays

## Common Patterns

### Document Navigation
```python
# Get table of contents from metadata
toc_chunks = db_handler.query(
    project_id=1,
    metadata_filter={'type': 'heading'},
    page_size=100
)

# Build navigation structure
toc = []
for chunk in toc_chunks.results:
    toc.append({
        'title': chunk.chunk.content,
        'section': chunk.chunk.metadata.get('section'),
        'page': chunk.chunk.metadata.get('page_number')
    })
```

### Quality Control
```python
# Find low-confidence chunks for review
review_needed = db_handler.query(
    project_id=1,
    metadata_filter={
        'confidence_score': 0.7  # Finds exact match
        # Note: For range queries, you'd need custom SQL
    },
    page_size=50
)
```

### Multi-language Support
```python
# Get chunks in specific language
spanish_chunks = db_handler.query(
    project_id=1,
    metadata_filter={'language': 'es'},
    page_size=20
)
```

## Limitations

- The `query` method uses exact matches for metadata values
- For range queries (e.g., confidence_score > 0.8), you'll need to use custom SQL
- Array filtering checks for containment, not exact array matching

## See Also

- [BM25 Hybrid Search Guide](bm25_hybrid_search.md) - For text-based search
- [Query Guide](query_guide.md) - For semantic search operations