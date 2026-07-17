ALTER TABLE public.quiz_attempts
ADD COLUMN IF NOT EXISTS target_duration_seconds integer;

ALTER TABLE public.quiz_attempts
ADD COLUMN IF NOT EXISTS actual_duration_seconds integer;
