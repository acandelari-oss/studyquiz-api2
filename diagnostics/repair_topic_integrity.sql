-- DOUNO topic/chunk integrity repair script.
--
-- IMPORTANT:
-- - Review the read-only diagnostics before running this script.
-- - This script deletes only impossible topic_chunks rows whose chunk_id
--   no longer exists and historical topics whose project no longer exists.
-- - It intentionally does NOT delete zero-evidence topics that still belong
--   to existing projects.

BEGIN;

CREATE TEMP TABLE orphan_topic_chunks_to_delete AS
SELECT tc.id
FROM public.topic_chunks tc
WHERE NOT EXISTS (
    SELECT 1
    FROM public.chunks c
    WHERE c.id = tc.chunk_id
);

CREATE TEMP TABLE ghost_topics_to_delete AS
SELECT t.id
FROM public.topics t
WHERE t.project_id IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM public.projects p
    WHERE p.id = t.project_id
);

SELECT
    count(*) AS orphan_topic_chunks_to_delete
FROM orphan_topic_chunks_to_delete;

SELECT
    count(*) AS ghost_topics_to_delete
FROM ghost_topics_to_delete;

DELETE FROM public.topic_chunks tc
USING orphan_topic_chunks_to_delete doomed
WHERE tc.id = doomed.id;

DELETE FROM public.topics t
USING ghost_topics_to_delete doomed
WHERE t.id = doomed.id;

COMMIT;

