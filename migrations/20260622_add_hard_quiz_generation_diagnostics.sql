CREATE TABLE IF NOT EXISTS public.hard_quiz_generation_runs (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    project_id uuid NOT NULL
        REFERENCES public.projects(id) ON DELETE CASCADE,
    project_name text NOT NULL,
    quiz_id uuid NOT NULL,
    difficulty text NOT NULL,
    question_style text NOT NULL,
    requested_questions integer NOT NULL,
    generated_questions integer NOT NULL,
    accepted_questions integer NOT NULL,
    rejected_questions integer NOT NULL,
    acceptance_rate double precision NOT NULL,
    rejection_reasons_breakdown jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamp with time zone NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS hard_quiz_generation_runs_project_created_idx
ON public.hard_quiz_generation_runs (project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS hard_quiz_generation_runs_quiz_id_idx
ON public.hard_quiz_generation_runs (quiz_id);

CREATE INDEX IF NOT EXISTS hard_quiz_generation_runs_created_idx
ON public.hard_quiz_generation_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS public.hard_quiz_generation_samples (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    run_id uuid NOT NULL
        REFERENCES public.hard_quiz_generation_runs(id)
        ON DELETE CASCADE,
    outcome text NOT NULL
        CHECK (outcome IN ('accepted', 'rejected')),
    question_stem text NOT NULL,
    rejection_reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
    question_type text,
    topic text,
    category text,
    created_at timestamp with time zone NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS hard_quiz_generation_samples_run_outcome_idx
ON public.hard_quiz_generation_samples (run_id, outcome);
