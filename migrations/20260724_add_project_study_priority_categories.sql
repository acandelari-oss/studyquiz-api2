ALTER TABLE public.projects
ADD COLUMN IF NOT EXISTS study_priority_categories jsonb NOT NULL DEFAULT '[]'::jsonb;
