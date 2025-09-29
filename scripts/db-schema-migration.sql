
CREATE TABLE IF NOT EXISTS performance_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_name TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    branch_name TEXT DEFAULT 'master',
    created_at TEXT NOT NULL,
    description TEXT,
    is_active INTEGER DEFAULT 1,
    UNIQUE(baseline_name, commit_sha)
);

CREATE TABLE IF NOT EXISTS performance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_time TEXT NOT NULL,
    build_commit TEXT NOT NULL,
    model_type TEXT,
    backends TEXT,
    n_gpu_layers INTEGER,
    avg_ts REAL,
    avg_ns INTEGER,
    stddev_ts REAL,
    stddev_ns INTEGER,
    cpu_info TEXT,
    gpu_info TEXT,
    n_threads INTEGER,
    n_prompt INTEGER,
    n_gen INTEGER,
    memory_usage_kb INTEGER,
    memory_status TEXT,
    FOREIGN KEY (build_commit) REFERENCES performance_baselines(commit_sha)
);

CREATE TABLE IF NOT EXISTS regression_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_time TEXT NOT NULL,
    baseline_commit TEXT NOT NULL,
    current_commit TEXT NOT NULL,
    benchmark_key TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    baseline_value REAL NOT NULL,
    current_value REAL NOT NULL,
    change_percentage REAL NOT NULL,
    threshold_percentage REAL NOT NULL,
    severity TEXT CHECK(severity IN ('warning', 'critical')) DEFAULT 'warning',
    status TEXT CHECK(status IN ('open', 'investigating', 'resolved', 'false_positive')) DEFAULT 'open',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS memory_leak_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_time TEXT NOT NULL,
    build_commit TEXT NOT NULL,
    test_name TEXT NOT NULL,
    memory_status TEXT NOT NULL,
    initial_memory_kb INTEGER,
    final_memory_kb INTEGER,
    peak_memory_kb INTEGER,
    leaked_memory_kb INTEGER,
    status_code INTEGER,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_performance_history_commit ON performance_history(build_commit);
CREATE INDEX IF NOT EXISTS idx_performance_history_time ON performance_history(test_time);
CREATE INDEX IF NOT EXISTS idx_performance_history_model ON performance_history(model_type);
CREATE INDEX IF NOT EXISTS idx_regression_alerts_time ON regression_alerts(alert_time);
CREATE INDEX IF NOT EXISTS idx_regression_alerts_status ON regression_alerts(status);
CREATE INDEX IF NOT EXISTS idx_memory_leak_logs_commit ON memory_leak_logs(build_commit);
CREATE INDEX IF NOT EXISTS idx_memory_leak_logs_time ON memory_leak_logs(test_time);

CREATE VIEW IF NOT EXISTS latest_baselines AS
SELECT
    b.baseline_name,
    b.commit_sha,
    b.branch_name,
    b.created_at,
    COUNT(h.id) as benchmark_count
FROM performance_baselines b
LEFT JOIN performance_history h ON b.commit_sha = h.build_commit
WHERE b.is_active = 1
GROUP BY b.id
ORDER BY b.created_at DESC;

CREATE VIEW IF NOT EXISTS regression_summary AS
SELECT
    current_commit,
    COUNT(*) as total_regressions,
    SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_count,
    SUM(CASE WHEN severity = 'warning' THEN 1 ELSE 0 END) as warning_count,
    AVG(ABS(change_percentage)) as avg_degradation
FROM regression_alerts
WHERE status = 'open'
GROUP BY current_commit
ORDER BY total_regressions DESC;

CREATE VIEW IF NOT EXISTS memory_leak_summary AS
SELECT
    build_commit,
    COUNT(*) as total_tests,
    SUM(CASE WHEN memory_status = 'LLAMA_MEMORY_STATUS_SUCCESS' THEN 1 ELSE 0 END) as passed_tests,
    SUM(CASE WHEN leaked_memory_kb > 0 THEN 1 ELSE 0 END) as leak_detected,
    SUM(leaked_memory_kb) as total_leaked_kb
FROM memory_leak_logs
GROUP BY build_commit
ORDER BY test_time DESC;

CREATE TRIGGER IF NOT EXISTS update_regression_alert_timestamp
AFTER UPDATE ON regression_alerts
FOR EACH ROW
WHEN OLD.status != NEW.status
BEGIN
    UPDATE regression_alerts
    SET alert_time = datetime('now')
    WHERE id = NEW.id;
END;
