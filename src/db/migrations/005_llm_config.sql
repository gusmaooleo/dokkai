CREATE TABLE llm_config (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    is_local BOOLEAN NOT NULL,
    provider_name TEXT NOT NULL,
    model TEXT NOT NULL,
    key TEXT NOT NULL DEFAULT '',
    base_url TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
