-- Review only: no backfill or taxonomy changes. Existing TEXT/UUID IDs unchanged.
BEGIN;
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS chapter_structure jsonb;
ALTER TABLE public.chunks ADD COLUMN IF NOT EXISTS structural_provenance jsonb;
ALTER TABLE public.documents ADD CONSTRAINT documents_chapter_structure_object
    CHECK (chapter_structure IS NULL OR jsonb_typeof(chapter_structure) = 'object');
ALTER TABLE public.chunks ADD CONSTRAINT chunks_structural_provenance_object
    CHECK (structural_provenance IS NULL OR jsonb_typeof(structural_provenance) = 'object');
COMMIT;
-- Rollback ONLY after disabling the chapter-aware application version:
-- ALTER TABLE public.chunks DROP COLUMN structural_provenance;
-- ALTER TABLE public.documents DROP COLUMN chapter_structure;
-- Dropping columns discards new provenance, but does not alter topics/study data.
