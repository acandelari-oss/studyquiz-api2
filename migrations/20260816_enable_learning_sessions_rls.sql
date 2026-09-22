ALTER TABLE public.learning_sessions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS learning_sessions_authenticated_select_own_project
ON public.learning_sessions;

CREATE POLICY learning_sessions_authenticated_select_own_project
ON public.learning_sessions
FOR SELECT
TO authenticated
USING (
    user_id = auth.uid()
    AND EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = learning_sessions.project_id
          AND p.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS learning_sessions_authenticated_insert_own_project
ON public.learning_sessions;

CREATE POLICY learning_sessions_authenticated_insert_own_project
ON public.learning_sessions
FOR INSERT
TO authenticated
WITH CHECK (
    user_id = auth.uid()
    AND EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = learning_sessions.project_id
          AND p.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS learning_sessions_authenticated_update_own_project
ON public.learning_sessions;

CREATE POLICY learning_sessions_authenticated_update_own_project
ON public.learning_sessions
FOR UPDATE
TO authenticated
USING (
    user_id = auth.uid()
    AND EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = learning_sessions.project_id
          AND p.user_id = auth.uid()
    )
)
WITH CHECK (
    user_id = auth.uid()
    AND EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = learning_sessions.project_id
          AND p.user_id = auth.uid()
    )
);
