-- Study Module organization metadata foundation.
--
-- This migration stores the student's intended organization for a module
-- without connecting it to taxonomy generation yet.
-- Existing modules remain valid and default to inferred organization.

ALTER TABLE public.study_modules
ADD COLUMN IF NOT EXISTS organization_mode text DEFAULT 'infer'::text NOT NULL;

ALTER TABLE public.study_modules
ADD COLUMN IF NOT EXISTS organization_blueprint jsonb;

ALTER TABLE public.study_modules
ADD COLUMN IF NOT EXISTS organization_source_title text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'study_modules_organization_mode_check'
    ) THEN
        ALTER TABLE public.study_modules
        ADD CONSTRAINT study_modules_organization_mode_check
        CHECK (
            organization_mode = ANY (
                ARRAY[
                    'infer'::text,
                    'manual'::text,
                    'syllabus'::text
                ]
            )
        );
    END IF;
END $$;
