#!/usr/bin/env python3
"""Database initialization script."""

import logging
from pathlib import Path

from sqlalchemy import create_engine, text

from vector_rag.config import Config

config = Config()
DB_URL = config.DB_URL
VECTOR_INDEX_LISTS = config.VECTOR_INDEX_LISTS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize database with all required schemas and indexes."""
    from vector_rag.db.db_model import DbBase, ChunkDB
    
    engine = create_engine(DB_URL)

    try:
        # Create vector extension first
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        # Set embedding dimension from config
        if config.EMBEDDINGS_DIM:
            ChunkDB.set_embedding_dimension(config.EMBEDDINGS_DIM)
            logger.info(f"Set embedding dimension to {config.EMBEDDINGS_DIM}")

        # Create tables
        DbBase.metadata.create_all(engine)
        logger.info("Created database tables")

        # Execute any additional SQL scripts
        sql_dir = Path(__file__).parent.parent.parent / "db" / "sql" / "ddl"
        if sql_dir.exists():
            for sql_file in sorted(sql_dir.glob("*.sql")):
                logger.info(f"Executing {sql_file.name}")
                sql = sql_file.read_text()
                sql = sql.replace(":vector_index_lists", str(VECTOR_INDEX_LISTS))
                
                with engine.connect() as conn:
                    statements = sql.split(";")
                    for statement in statements:
                        if statement.strip():
                            try:
                                conn.execute(text(statement))
                                conn.commit()
                            except Exception as e:
                                logger.warning(f"Error executing statement: {e}")
                                continue

        logger.info("Database initialization completed successfully")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    init_database()
