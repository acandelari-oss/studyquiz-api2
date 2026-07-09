CREATE TABLE IF NOT EXISTS planner_weeks (
    id text PRIMARY KEY,
    project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    start_date date NOT NULL,
    end_date date NOT NULL,
    status text NOT NULL,
    planning_parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    weekly_briefing text,
    weekly_statistics jsonb NOT NULL DEFAULT '{}'::jsonb,
    weekly_review text,
    next_week_options jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS planner_weeks_one_active_per_project_idx
ON planner_weeks(project_id)
WHERE status = 'ACTIVE';

CREATE INDEX IF NOT EXISTS planner_weeks_project_status_idx
ON planner_weeks(project_id, status);

CREATE TABLE IF NOT EXISTS planner_daily_plans (
    id text PRIMARY KEY,
    week_id text NOT NULL REFERENCES planner_weeks(id) ON DELETE CASCADE,
    session_index integer NOT NULL,
    plan_date date NOT NULL,
    day_name text NOT NULL,
    status text NOT NULL,
    objective text,
    briefing text,
    planned_allocations jsonb NOT NULL DEFAULT '[]'::jsonb,
    summary jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (week_id, session_index)
);

CREATE INDEX IF NOT EXISTS planner_daily_plans_week_idx
ON planner_daily_plans(week_id, session_index);

CREATE TABLE IF NOT EXISTS planner_activities (
    id text PRIMARY KEY,
    daily_plan_id text NOT NULL REFERENCES planner_daily_plans(id) ON DELETE CASCADE,
    activity_index integer NOT NULL,
    activity_type text NOT NULL,
    configuration jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (daily_plan_id, activity_index)
);

CREATE INDEX IF NOT EXISTS planner_activities_daily_plan_idx
ON planner_activities(daily_plan_id, activity_index);
