-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the vector column with the correct dimension
-- This will be used as a template for all vector columns
CREATE TABLE IF NOT EXISTS vector_dimension_template (
    embedding vector(384)
);
