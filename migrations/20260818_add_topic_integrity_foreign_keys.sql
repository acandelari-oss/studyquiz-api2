-- Add database-level integrity protection for the topic/chunk layer.
--
-- Preconditions:
-- - public.topic_chunks contains no rows whose chunk_id is missing from chunks.
-- - public.topics contains no rows whose project_id is missing from projects.
--
-- This migration intentionally does not make nullable columns NOT NULL.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.topic_chunks tc
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.chunks c
            WHERE c.id = tc.chunk_id
        )
    ) THEN
        RAISE EXCEPTION
            'Cannot add topic_chunks.chunk_id FK: orphan topic_chunks rows still exist';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.topics t
        WHERE t.project_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1
            FROM public.projects p
            WHERE p.id = t.project_id
        )
    ) THEN
        RAISE EXCEPTION
            'Cannot add topics.project_id FK: topics referencing missing projects still exist';
    END IF;
END $$;

ALTER TABLE public.topic_chunks
ADD CONSTRAINT topic_chunks_chunk_id_fkey
FOREIGN KEY (chunk_id)
REFERENCES public.chunks(id)
ON DELETE CASCADE;

ALTER TABLE public.topics
ADD CONSTRAINT topics_project_id_fkey
FOREIGN KEY (project_id)
REFERENCES public.projects(id)
ON DELETE CASCADE;

