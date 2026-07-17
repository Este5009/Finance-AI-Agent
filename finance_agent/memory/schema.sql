PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    period TEXT NOT NULL,
    period_type TEXT NOT NULL,
    started_at_utc TEXT,
    completed_at_utc TEXT NOT NULL,
    report_hash TEXT NOT NULL,
    goals_hash TEXT NOT NULL,
    report_path TEXT NOT NULL,
    goals_path TEXT NOT NULL,
    language TEXT NOT NULL,
    model TEXT NOT NULL,
    confidence REAL,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    cache_key TEXT,
    status TEXT NOT NULL,
    artifact_directory TEXT NOT NULL,
    configuration_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    checksum TEXT,
    created_at_utc TEXT NOT NULL,
    UNIQUE(run_id, artifact_type, path),
    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kpis (
    kpi_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    period TEXT,
    department TEXT,
    metric TEXT NOT NULL,
    value REAL,
    unit TEXT,
    status TEXT,
    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS anomalies (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    anomaly_id TEXT NOT NULL,
    period TEXT,
    department TEXT,
    type TEXT,
    severity TEXT,
    metric TEXT,
    values_json TEXT,
    description TEXT,
    UNIQUE(run_id, anomaly_id),
    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    priority TEXT,
    department TEXT,
    action TEXT NOT NULL,
    expected_impact TEXT,
    status TEXT NOT NULL DEFAULT 'unknown',
    follow_up_required INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(run_id, recommendation_id),
    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS goals (
    goal_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    target REAL,
    actual REAL,
    unit TEXT,
    progress_status TEXT,
    PRIMARY KEY(run_id, goal_id),
    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_facts (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    category TEXT NOT NULL,
    subject TEXT NOT NULL,
    fact TEXT NOT NULL,
    confidence REAL,
    UNIQUE(run_id, category, subject, fact),
    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_period ON pipeline_runs(period);
CREATE INDEX IF NOT EXISTS idx_artifacts_run_type ON artifacts(run_id, artifact_type);
CREATE INDEX IF NOT EXISTS idx_kpis_metric ON kpis(metric);
CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies(severity);
CREATE INDEX IF NOT EXISTS idx_memory_facts_category ON memory_facts(category);
