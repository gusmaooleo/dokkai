CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    project TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    data JSONB NOT NULL
);

CREATE INDEX idx_jobs_created ON jobs(created_at DESC);
