# RAG (Retrieval-Augmented Generation) System

A Python-based RAG system that processes text files, generates embeddings, and stores them in a Postgres database 
with pgvector for efficient similarity search.

For detailed information, see:
- [Design Document](DESIGN.md) - System architecture and requirements
- [Developer Guide](developer_guide.md) - Detailed setup and development instructions
- [Metadata Query Guide](docs/metadata_query_guide.md) - Using JSONB metadata queries without search

## Features

- File ingestion (source code, Markdown, plain text)
- Text chunking with configurable overlap
- Vector embeddings via OpenAI or Hugging Face
- Postgres + pgvector for vector storage and search
- **BM25 full-text search** using PostgreSQL's tsvector
- **Hybrid search** combining vector similarity and BM25 scores
- Project-based organization of documents
- Comprehensive Taskfile for development workflows
- Dockerized Postgres database with pgvector

## Getting Started

### Prerequisites

- Python 3.10+
- Docker
- Poetry (will be installed automatically by the setup script)

### Poetry Management

This project uses Poetry for dependency management. Key Poetry commands are wrapped in Taskfile tasks:

- Install dependencies:
```bash
task install
```

- Set up development environment:
```bash
task setup-dev
```

- Update dependencies:
```bash
task update-deps
```

- Export requirements files:
```bash
task export-reqs
```

- Check dependency status:
```bash
task verify-deps
```

The project includes both runtime and development dependencies specified in `pyproject.toml`.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SpillwaveSolutions/vector-rag
cd vector-rag
```

2. Set up the development environment:
```bash
task setup-dev
```

3. Configure environment variables:
```bash
cp environment/.env.example .env
# Edit .env with your settings:
# - Database credentials
# - OpenAI API key (if using OpenAI embeddings)
```

## Running the System

### Start the Database

The system uses a Dockerized Postgres database with pgvector:
```bash
task db:up
```

### Apply Database Migrations

For BM25 search support on existing databases, run migrations:
```bash
task db:migrate
```

Note: New installations will have BM25 support included automatically.

### Run Examples

- With mock embeddings (no API key required):
```bash
task demo:mock
```

- With OpenAI embeddings (requires API key in .env):
```bash
task demo:openai
```

### Interactive Database Access

To access the database directly:
```bash
task psql
```

## Testing

The project includes comprehensive tests:

- Run all tests:
```bash
task test:all
```

- Run integration tests:
```bash
task test:integration
```

- Run tests with coverage report:
```bash
task test:coverage
```

- Run a specific test:
```bash
task test:single -- tests/path/to/test_file.py::test_name
```

## Development Workflow

### Code Formatting and Linting
```bash
task format  # Runs black and isort
task typecheck  # Runs mypy
task lint  # Runs all code quality checks
```

### Dependency Management

- Update dependencies:
```bash
task update-deps
```

- Export requirements files:
```bash
task export-reqs
```

## Database Management

- Recreate the database from scratch:
```bash
task db:recreate
```

- Stop the database:
```bash
task db:down
```

## Configuration

The system is configured through environment variables in `.env`. Key settings include:

- `DB_*`: Database connection settings
- `OPENAI_API_KEY`: Required for OpenAI embeddings
- `LOCAL_EMBEDDING`: Set to `true` to use local SentenceTransformers
- `EMBEDDINGS_DIM`: Vector dimension (384 for local, 1536 for OpenAI)
- `CHUNK_SIZE`/`CHUNK_OVERLAP`: Text chunking parameters

## Search Capabilities

The system provides three types of search:

### 1. Vector Search (Semantic)
Traditional semantic search using vector embeddings:
```python
from vector_rag.api import VectorRAGAPI

api = VectorRAGAPI()
results = api.search_text(
    project_id=1,
    query_text="implement authentication",
    similarity_threshold=0.7
)
```

### 2. BM25 Search (Keyword/Lexical)
Full-text search using PostgreSQL's tsvector for exact keyword matching:
```python
results = api.search_bm25(
    project_id=1,
    query_text="PostgreSQL tsvector index",
    rank_threshold=0.0
)
```

### 3. Hybrid Search
Combines vector and BM25 scores with configurable weights:
```python
results = api.search_hybrid(
    project_id=1,
    query_text="implement OAuth authentication",
    vector_weight=0.6,  # 60% semantic
    bm25_weight=0.4,    # 40% keyword
)
```

### When to Use Each Search Type

- **Vector Search**: Best for natural language queries, finding conceptually similar content
- **BM25 Search**: Best for technical terms, code, specific keywords
- **Hybrid Search**: Best overall performance, especially for technical documentation

For more details, see [BM25 and Hybrid Search Documentation](docs/bm25_hybrid_search.md).

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Please ensure all tests pass and code is properly formatted before submitting PRs.
