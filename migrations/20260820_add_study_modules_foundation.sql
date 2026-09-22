-- Study Module foundation.
--
-- This migration introduces the minimum data model needed for module-owned
-- taxonomies without changing upload, extraction, topic generation or planner
-- behavior. Legacy documents/topics remain valid with module_id = NULL.

CREATE TABLE IF NOT EXISTS public.study_modules (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    project_id uuid NOT NULL,
    name text NOT NULL,
    order_index integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    status text DEFAULT 'building'::text NOT NULL,
    accepted_for_study boolean DEFAULT false NOT NULL,
    accepted_at timestamp with time zone,
    taxonomy_status text DEFAULT 'not_started'::text NOT NULL,
    CONSTRAINT study_modules_pkey PRIMARY KEY (id),
    CONSTRAINT study_modules_status_check CHECK (
        status = ANY (
            ARRAY[
                'building'::text,
                'taxonomy_ready'::text,
                'pending_study'::text,
                'accepted_for_study'::text
            ]
        )
    ),
    CONSTRAINT study_modules_taxonomy_status_check CHECK (
        taxonomy_status = ANY (
            ARRAY[
                'not_started'::text,
                'building'::text,
                'ready'::text,
                'failed'::text
            ]
        )
    )
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'study_modules_project_id_fkey'
    ) THEN
        ALTER TABLE public.study_modules
        ADD CONSTRAINT study_modules_project_id_fkey
        FOREIGN KEY (project_id)
        REFERENCES public.projects(id)
        ON DELETE CASCADE;
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS study_modules_project_order_idx
ON public.study_modules(project_id, order_index);

CREATE INDEX IF NOT EXISTS study_modules_project_created_idx
ON public.study_modules(project_id, created_at);

ALTER TABLE public.documents
ADD COLUMN IF NOT EXISTS module_id uuid;

ALTER TABLE public.documents
ADD COLUMN IF NOT EXISTS module_order_index integer;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'documents_module_id_fkey'
    ) THEN
        ALTER TABLE public.documents
        ADD CONSTRAINT documents_module_id_fkey
        FOREIGN KEY (module_id)
        REFERENCES public.study_modules(id)
        ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS documents_module_id_idx
ON public.documents(module_id);

CREATE INDEX IF NOT EXISTS documents_project_module_order_idx
ON public.documents(project_id, module_id, module_order_index);

ALTER TABLE public.topics
ADD COLUMN IF NOT EXISTS module_id uuid;

ALTER TABLE public.topics
ADD COLUMN IF NOT EXISTS category_order_index integer;

ALTER TABLE public.topics
ADD COLUMN IF NOT EXISTS topic_order_index integer;

ALTER TABLE public.topics
ADD COLUMN IF NOT EXISTS first_source_page integer;

ALTER TABLE public.topics
ADD COLUMN IF NOT EXISTS first_source_block_index integer;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'topics_module_id_fkey'
    ) THEN
        ALTER TABLE public.topics
        ADD CONSTRAINT topics_module_id_fkey
        FOREIGN KEY (module_id)
        REFERENCES public.study_modules(id)
        ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS topics_module_id_idx
ON public.topics(module_id);

CREATE INDEX IF NOT EXISTS topics_project_module_order_idx
ON public.topics(
    project_id,
    module_id,
    category_order_index,
    topic_order_index
);

ALTER TABLE public.study_modules ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS study_modules_authenticated_select_own_project
ON public.study_modules;

CREATE POLICY study_modules_authenticated_select_own_project
ON public.study_modules
FOR SELECT
TO authenticated
USING (
    EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = study_modules.project_id
          AND p.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS study_modules_authenticated_insert_own_project
ON public.study_modules;

CREATE POLICY study_modules_authenticated_insert_own_project
ON public.study_modules
FOR INSERT
TO authenticated
WITH CHECK (
    EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = study_modules.project_id
          AND p.user_id = auth.uid()
    )
);

DROP POLICY IF EXISTS study_modules_authenticated_update_own_project
ON public.study_modules;

CREATE POLICY study_modules_authenticated_update_own_project
ON public.study_modules
FOR UPDATE
TO authenticated
USING (
    EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = study_modules.project_id
          AND p.user_id = auth.uid()
    )
)
WITH CHECK (
    EXISTS (
        SELECT 1
        FROM public.projects p
        WHERE p.id = study_modules.project_id
          AND p.user_id = auth.uid()
    )
);
