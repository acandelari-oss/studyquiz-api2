ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS projects_authenticated_select_own
ON public.projects;

CREATE POLICY projects_authenticated_select_own
ON public.projects
FOR SELECT
TO authenticated
USING (
    user_id = auth.uid()
);

DROP POLICY IF EXISTS projects_authenticated_insert_own
ON public.projects;

CREATE POLICY projects_authenticated_insert_own
ON public.projects
FOR INSERT
TO authenticated
WITH CHECK (
    user_id = auth.uid()
);

DROP POLICY IF EXISTS projects_authenticated_update_own
ON public.projects;

CREATE POLICY projects_authenticated_update_own
ON public.projects
FOR UPDATE
TO authenticated
USING (
    user_id = auth.uid()
)
WITH CHECK (
    user_id = auth.uid()
);

DROP POLICY IF EXISTS projects_authenticated_delete_own
ON public.projects;

CREATE POLICY projects_authenticated_delete_own
ON public.projects
FOR DELETE
TO authenticated
USING (
    user_id = auth.uid()
);
