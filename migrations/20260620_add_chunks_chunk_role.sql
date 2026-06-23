ALTER TABLE public.chunks
ADD COLUMN IF NOT EXISTS chunk_role text;

ALTER TABLE public.chunks
DROP CONSTRAINT IF EXISTS chunks_chunk_role_check;

ALTER TABLE public.chunks
ADD CONSTRAINT chunks_chunk_role_check
CHECK (
    chunk_role IS NULL
    OR chunk_role IN (
        'teaching',
        'intro',
        'outline',
        'bibliography',
        'administrative',
        'cover'
    )
);
