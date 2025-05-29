-- Add full-text search support to chunks table

-- Add tsvector column with generated value
ALTER TABLE chunks 
ADD COLUMN IF NOT EXISTS content_tsv tsvector 
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- Create GIN index for efficient full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv 
ON chunks USING GIN (content_tsv);

-- Analyze table to update statistics for query planner
ANALYZE chunks;