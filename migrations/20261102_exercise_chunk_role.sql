-- Review only: apply before deploying exercise-aware ingestion.
-- No rows are changed; existing roles and NULL remain valid.
BEGIN;
ALTER TABLE public.chunks DROP CONSTRAINT IF EXISTS chunks_chunk_role_check;
ALTER TABLE public.chunks ADD CONSTRAINT chunks_chunk_role_check CHECK (
    chunk_role IS NULL OR chunk_role IN (
        'teaching', 'intro', 'outline', 'bibliography',
        'administrative', 'cover', 'exercise'
    )
);
COMMIT;
-- Rollback requires confirming no exercise rows exist. Never relabel them teaching.
