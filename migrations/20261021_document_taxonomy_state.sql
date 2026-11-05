-- NULL preserves unknown legacy state; no existing taxonomy is migrated.
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS taxonomy_status text;
ALTER TABLE public.documents ADD CONSTRAINT documents_taxonomy_status_check
    CHECK (taxonomy_status IS NULL OR taxonomy_status IN ('queued', 'building', 'ready', 'failed'));
CREATE INDEX IF NOT EXISTS topics_project_document_idx ON public.topics(project_id, document_id);
