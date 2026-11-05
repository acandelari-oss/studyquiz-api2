-- Review before execution. No topic/chunk contents or ownership are changed.
-- Verified live identifiers: chunks.project_id UUID, chunks.document_id TEXT.
-- B-tree uses native column types; no casts, backfill, or new columns.
-- Equality filters cover project/document/role; id supplies snapshot ordering.
-- Do not INCLUDE chunk_text, embedding, or provenance: these large values
-- would substantially increase index size and write cost.
--
-- Execute CREATE INDEX CONCURRENTLY as a standalone statement, outside any
-- transaction block. Do not run this file through a transactional migration
-- runner. Concurrent creation permits ordinary writes but consumes resources;
-- schedule during a quiet period, with no upload/retry running.
-- If interrupted, inspect pg_index.indisvalid before retrying: an invalid
-- index may remain. Deliberately omit IF NOT EXISTS so it cannot hide that.
CREATE INDEX CONCURRENTLY chunks_project_document_role_id_idx
    ON public.chunks (project_id, document_id, chunk_role, id);

-- Run separately after successful index creation. Refresh planner statistics
-- without modifying application rows. Do not change statistics targets here.
ANALYZE public.chunks;

-- Verification (read-only): expect indisvalid=true and indisready=true.
-- SELECT i.indisvalid, i.indisready, pg_get_indexdef(i.indexrelid)
-- FROM pg_index i
-- WHERE i.indexrelid = to_regclass('public.chunks_project_document_role_id_idx');
-- Then repeat the bounded source-snapshot timing before any AI retry.
--
-- Rollback: execute separately, outside a transaction block.
-- DROP INDEX CONCURRENTLY public.chunks_project_document_role_id_idx;
-- ANALYZE has no data changes to roll back; refreshed statistics may remain.
