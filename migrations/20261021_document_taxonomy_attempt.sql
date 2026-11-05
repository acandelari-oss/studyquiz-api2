-- Required for explicit recovery: fences superseded background workers.
-- No backfill or regeneration; NULL remains valid until a new attempt starts.
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS taxonomy_run_id uuid;
