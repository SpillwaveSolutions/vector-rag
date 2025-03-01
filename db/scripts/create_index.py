import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters from .env
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "vectordb")

"""
Takes
"""
def create_index():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    # Create indexes after data import
    try:
        print("Creating indexes...")
        index_commands = """
        SET maintenance_work_mem = '256MB';

        -- Drop existing vector index if it exists
        DROP INDEX IF EXISTS idx_chunks_embedding;

        -- Create optimized IVFFlat index for vector similarity search with increased lists
        CREATE INDEX idx_chunks_embedding ON chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 500);

        -- Maintain supporting index for efficient file and chunk lookups
        CREATE INDEX IF NOT EXISTS idx_chunks_file_metadata ON chunks
        USING btree (file_id, chunk_index);

        -- Update table statistics for query optimization
        ANALYZE chunks;
        """
        cur.execute(index_commands)
        conn.commit()
        print("Successfully created indexes")
    except psycopg2.Error as e:
        print(f"Error creating indexes: {str(e)}")
        conn.rollback()

    cur.close()
    conn.close()


if __name__ == "__main__":
    create_index()

