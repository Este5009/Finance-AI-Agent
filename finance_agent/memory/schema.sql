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

CREATE TABLE IF NOT EXISTS source_documents (
    document_id TEXT PRIMARY KEY,
    content_sha256 TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    document_type TEXT NOT NULL CHECK(document_type IN ('integrated_workbook', 'financial_report', 'goals_document')),
    size_bytes INTEGER NOT NULL,
    detected_period TEXT,
    effective_period TEXT NOT NULL,
    upload_time_utc TEXT NOT NULL,
    processing_status TEXT NOT NULL,
    source_metadata_json TEXT NOT NULL DEFAULT '{}',
    version_number INTEGER NOT NULL DEFAULT 1,
    supersedes_document_id TEXT,
    is_current INTEGER NOT NULL DEFAULT 1,
    UNIQUE(document_type, content_sha256),
    FOREIGN KEY(supersedes_document_id) REFERENCES source_documents(document_id)
);

CREATE TABLE IF NOT EXISTS pipeline_run_documents (
    run_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_role TEXT NOT NULL CHECK(document_role IN ('integrated_workbook', 'financial_report', 'goals_document')),
    PRIMARY KEY(run_id, document_role),
    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY(document_id) REFERENCES source_documents(document_id)
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
    gap REAL,
    score REAL,
    direction TEXT,
    source_provenance_json TEXT,
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
CREATE INDEX IF NOT EXISTS idx_source_documents_hash ON source_documents(document_type, content_sha256);
CREATE INDEX IF NOT EXISTS idx_source_documents_period ON source_documents(document_type, effective_period, is_current);
CREATE UNIQUE INDEX IF NOT EXISTS idx_source_documents_current_period
ON source_documents(document_type, effective_period)
WHERE is_current = 1 AND processing_status = 'accepted';
CREATE INDEX IF NOT EXISTS idx_artifacts_run_type ON artifacts(run_id, artifact_type);
CREATE INDEX IF NOT EXISTS idx_kpis_metric ON kpis(metric);
CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies(severity);
CREATE INDEX IF NOT EXISTS idx_memory_facts_category ON memory_facts(category);
