ALTER TABLE public.projects
ADD COLUMN IF NOT EXISTS study_mode text NOT NULL DEFAULT 'building';

ALTER TABLE public.projects
DROP CONSTRAINT IF EXISTS projects_study_mode_check;

ALTER TABLE public.projects
ADD CONSTRAINT projects_study_mode_check
CHECK (study_mode IN ('building', 'learning'));
