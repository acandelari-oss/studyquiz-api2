ALTER TABLE public.quiz_attempts ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS quiz_attempts_authenticated_select_own_project_quiz
ON public.quiz_attempts;

CREATE POLICY quiz_attempts_authenticated_select_own_project_quiz
ON public.quiz_attempts
FOR SELECT
TO authenticated
USING (
    user_id = auth.uid()
    AND EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = quiz_attempts.project_id
          AND p.user_id = auth.uid()
    )
    AND EXISTS (
        SELECT 1
        FROM public.quizzes q
        WHERE q.id = quiz_attempts.quiz_id
          AND q.project_id = quiz_attempts.project_id
          AND q.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS quiz_attempts_authenticated_insert_own_project_quiz
ON public.quiz_attempts;

CREATE POLICY quiz_attempts_authenticated_insert_own_project_quiz
ON public.quiz_attempts
FOR INSERT
TO authenticated
WITH CHECK (
    user_id = auth.uid()
    AND EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = quiz_attempts.project_id
          AND p.user_id = auth.uid()
    )
    AND EXISTS (
        SELECT 1
        FROM public.quizzes q
        WHERE q.id = quiz_attempts.quiz_id
          AND q.project_id = quiz_attempts.project_id
          AND q.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS quiz_attempts_authenticated_update_own_project_quiz
ON public.quiz_attempts;

CREATE POLICY quiz_attempts_authenticated_update_own_project_quiz
ON public.quiz_attempts
FOR UPDATE
TO authenticated
USING (
    user_id = auth.uid()
    AND EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = quiz_attempts.project_id
          AND p.user_id = auth.uid()
    )
    AND EXISTS (
        SELECT 1
        FROM public.quizzes q
        WHERE q.id = quiz_attempts.quiz_id
          AND q.project_id = quiz_attempts.project_id
          AND q.user_id = auth.uid()
    )
)
WITH CHECK (
    user_id = auth.uid()
    AND EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = quiz_attempts.project_id
          AND p.user_id = auth.uid()
    )
    AND EXISTS (
        SELECT 1
        FROM public.quizzes q
        WHERE q.id = quiz_attempts.quiz_id
          AND q.project_id = quiz_attempts.project_id
          AND q.user_id = auth.uid()
    )
);
