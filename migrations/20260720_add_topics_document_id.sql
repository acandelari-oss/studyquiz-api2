ALTER TABLE public.topics
ADD COLUMN IF NOT EXISTS document_id text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'topics_document_id_fkey'
    ) THEN
        ALTER TABLE public.topics
        ADD CONSTRAINT topics_document_id_fkey
        FOREIGN KEY (document_id)
        REFERENCES public.documents(id)
        ON DELETE SET NULL;
    END IF;
END $$;
