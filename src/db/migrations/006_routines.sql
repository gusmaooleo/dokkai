CREATE TABLE routine_runs (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('review', 'bughunt')),
    project TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary TEXT,
    stats JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_routine_runs_project_created ON routine_runs(project, created_at DESC);

CREATE TABLE findings (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES routine_runs(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    start_line INT,
    end_line INT,
    severity TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info')),
    category TEXT NOT NULL CHECK (category IN ('bug', 'security', 'performance', 'maintainability', 'style', 'other')),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    suggestion TEXT,
    evidence JSONB,
    anchored BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_findings_run ON findings(run_id);
