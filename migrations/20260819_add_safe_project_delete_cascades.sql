-- Add safe project-delete cascade constraints for project-owned tables.
--
-- These constraints are limited to relationships that were verified to have
-- zero invalid rows during the project-deletion lifecycle audit.
--
-- Do not apply this as a substitute for the separate topic integrity cleanup:
-- topics.project_id and topic_chunks.chunk_id are handled by
-- 20260818_add_topic_integrity_foreign_keys.sql after historical cleanup.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.quizzes q
        WHERE NOT EXISTS (
            SELECT 1 FROM public.projects p WHERE p.id = q.project_id
        )
    ) THEN
        RAISE EXCEPTION 'Cannot add quizzes.project_id FK: invalid rows exist';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.flashcards f
        WHERE NOT EXISTS (
            SELECT 1 FROM public.projects p WHERE p.id = f.project_id
        )
    ) THEN
        RAISE EXCEPTION 'Cannot add flashcards.project_id FK: invalid rows exist';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.recall_answers r
        WHERE r.project_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM public.projects p WHERE p.id = r.project_id
        )
    ) THEN
        RAISE EXCEPTION 'Cannot add recall_answers.project_id FK: invalid rows exist';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.learning_events le
        WHERE le.project_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM public.projects p WHERE p.id = le.project_id
        )
    ) THEN
        RAISE EXCEPTION 'Cannot add learning_events.project_id FK: invalid rows exist';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.chat_messages cm
        WHERE cm.project_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM public.projects p WHERE p.id = cm.project_id
        )
    ) THEN
        RAISE EXCEPTION 'Cannot add chat_messages.project_id FK: invalid rows exist';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.chat_messages cm
        WHERE cm.quiz_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM public.quizzes q WHERE q.id = cm.quiz_id
        )
    ) THEN
        RAISE EXCEPTION 'Cannot add chat_messages.quiz_id FK: invalid rows exist';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.hard_quiz_generation_runs h
        WHERE NOT EXISTS (
            SELECT 1 FROM public.quizzes q WHERE q.id = h.quiz_id
        )
    ) THEN
        RAISE EXCEPTION 'Cannot add hard_quiz_generation_runs.quiz_id FK: invalid rows exist';
    END IF;
END $$;

ALTER TABLE public.quizzes
ADD CONSTRAINT quizzes_project_id_fkey
FOREIGN KEY (project_id)
REFERENCES public.projects(id)
ON DELETE CASCADE;

ALTER TABLE public.flashcards
ADD CONSTRAINT flashcards_project_id_fkey
FOREIGN KEY (project_id)
REFERENCES public.projects(id)
ON DELETE CASCADE;

ALTER TABLE public.recall_answers
ADD CONSTRAINT recall_answers_project_id_fkey
FOREIGN KEY (project_id)
REFERENCES public.projects(id)
ON DELETE CASCADE;

ALTER TABLE public.learning_events
ADD CONSTRAINT learning_events_project_id_fkey
FOREIGN KEY (project_id)
REFERENCES public.projects(id)
ON DELETE CASCADE;

ALTER TABLE public.chat_messages
ADD CONSTRAINT chat_messages_project_id_fkey
FOREIGN KEY (project_id)
REFERENCES public.projects(id)
ON DELETE CASCADE;

ALTER TABLE public.chat_messages
ADD CONSTRAINT chat_messages_quiz_id_fkey
FOREIGN KEY (quiz_id)
REFERENCES public.quizzes(id)
ON DELETE CASCADE;

ALTER TABLE public.hard_quiz_generation_runs
ADD CONSTRAINT hard_quiz_generation_runs_quiz_id_fkey
FOREIGN KEY (quiz_id)
REFERENCES public.quizzes(id)
ON DELETE CASCADE;

