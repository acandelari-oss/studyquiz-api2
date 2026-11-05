-- Review only; do not apply automatically. No backfill or source/topic changes.
BEGIN;
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS source_review jsonb;
ALTER TABLE public.documents ADD CONSTRAINT documents_source_review_object
    CHECK (source_review IS NULL OR jsonb_typeof(source_review) = 'object');
COMMIT;
-- Rollback: disable DOUNO_SOURCE_REVIEW first. Keep the column and review API
-- until outstanding reviews are resolved/exported. Dropping it loses review
-- decisions, not source text; do not drop it as part of a routine rollback.
