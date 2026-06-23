ALTER TABLE public.projects
ADD COLUMN IF NOT EXISTS taxonomy_language text;
