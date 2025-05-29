-- Add full-text search support to chunks table
-- This migration is idempotent and can be run multiple times safely

-- Use a transaction to ensure atomicity
BEGIN;

-- Check if the content_tsv column exists, add it if not
DO $$ 
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables 
               WHERE table_schema = 'public' 
               AND table_name = 'chunks') THEN
        
        IF NOT EXISTS (SELECT FROM information_schema.columns 
                       WHERE table_schema = 'public' 
                       AND table_name = 'chunks' 
                       AND column_name = 'content_tsv') THEN
            
            -- Add tsvector column
            ALTER TABLE chunks 
            ADD COLUMN content_tsv tsvector 
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
            
            RAISE NOTICE 'Added content_tsv column to chunks table';
        ELSE
            RAISE NOTICE 'content_tsv column already exists';
        END IF;
        
        -- Create or replace the GIN index
        CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv 
        ON chunks USING GIN (content_tsv);
        
        -- Update statistics for query planner
        ANALYZE chunks;
        
        RAISE NOTICE 'Full-text search support is ready';
    ELSE
        RAISE WARNING 'chunks table does not exist - skipping FTS setup';
    END IF;
END $$;

COMMIT;