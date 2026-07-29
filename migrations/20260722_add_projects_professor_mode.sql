ALTER TABLE projects
ADD COLUMN IF NOT EXISTS professor_mode text NOT NULL DEFAULT 'coverage';

UPDATE projects
SET professor_mode = 'coverage'
WHERE professor_mode IS NULL;

ALTER TABLE projects
DROP CONSTRAINT IF EXISTS projects_professor_mode_check;

ALTER TABLE projects
ADD CONSTRAINT projects_professor_mode_check
CHECK (professor_mode IN ('coverage', 'adaptive'));
