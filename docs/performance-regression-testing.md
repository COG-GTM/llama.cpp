# Performance Regression Testing

This document describes the automated performance regression testing system for llama.cpp, implemented as part of JIRA ticket AT-105.

## Overview

The performance regression testing system automatically detects performance degradations in llama.cpp by comparing benchmark results against established baselines. It integrates with GitHub Actions CI/CD pipelines and provides automated alerts when performance regressions exceed a configurable threshold (default: 5%).

## Components

### 1. GitHub Actions Workflow

**File:** `.github/workflows/performance-regression.yml`

The workflow runs performance benchmarks on different hardware backends (CPU, CUDA, Metal) for every pull request and push to master. It:

- Builds the `llama-bench` target
- Downloads a test model (TinyLlama 1.1B)
- Runs benchmarks with consistent parameters
- Compares results against cached baselines
- Posts results as PR comments
- Fails the build if regressions are detected

**Jobs:**
- `performance-cpu`: Runs on Ubuntu with CPU backend
- `performance-cuda`: Runs on GPU runners (disabled by default)
- `performance-metal`: Runs on macOS with Apple Silicon

**Triggers:**
- Pull requests to any branch
- Pushes to master branch
- Manual workflow dispatch

### 2. Performance Regression Detector

**File:** `scripts/performance-regression-detector.py`

Python script that analyzes benchmark results and detects performance regressions.

**Usage:**
```bash
python3 scripts/performance-regression-detector.py \
  --baseline baseline.sqlite \
  --current current.sqlite \
  --threshold 5.0 \
  --output regression-report.md
```

**Features:**
- Compares multiple performance metrics (tokens/second, latency)
- Configurable regression threshold
- Generates markdown and JSON reports
- Creates flag file when regressions detected
- Integrates with existing llama-bench SQLite schema

**Key Metrics:**
- `avg_ts`: Average tokens per second (higher is better)
- `avg_ns`: Average latency in nanoseconds (lower is better)
- `model_size`: Model memory footprint (lower is better)

### 3. Enhanced Comparison Script

**File:** `scripts/compare-llama-bench.py` (enhanced)

The existing comparison script has been extended with CI automation support.

**New Features:**
- `--ci-mode`: Enable CI-specific formatting and behavior
- `--baseline-db`: Path to baseline database for tracking
- `--save-baseline`: Save current results as new baseline
- `--json-output`: Export comparison results to JSON

**Example:**
```bash
python3 scripts/compare-llama-bench.py \
  -i results.sqlite \
  --ci-mode \
  --json-output comparison.json
```

### 4. Database Schema Extensions

**Files:**
- `scripts/db-schema-migration.sql`: SQL migration script
- `scripts/apply-db-migration.py`: Migration application tool

The database schema has been extended to support:

**New Tables:**
- `performance_baselines`: Stores baseline snapshots
- `performance_history`: Historical performance data
- `regression_alerts`: Logged regression detections
- `memory_leak_logs`: Memory leak monitoring results

**Views:**
- `latest_baselines`: Active baseline information
- `regression_summary`: Aggregated regression statistics
- `memory_leak_summary`: Memory leak detection summary

**Applying Migrations:**
```bash
python3 scripts/apply-db-migration.py -d llama-bench.sqlite
```

### 5. Memory Leak Monitoring

**File:** `scripts/memory-leak-monitor.py`

Integrates with the existing `llama-memory.h` interfaces to detect memory leaks and excessive memory consumption.

**Usage:**
```bash
python3 scripts/memory-leak-monitor.py \
  --benchmark-output benchmark.log \
  --test-log test.log \
  --database results.sqlite \
  --commit abc123 \
  --report memory-report.md
```

**Features:**
- Parses benchmark output for memory usage patterns
- Detects memory leaks (threshold: 1 MB)
- Monitors excessive memory usage (threshold: 16 GB)
- Logs results to database
- Generates markdown reports

**Memory Status Codes** (from `llama-memory.h`):
- `0`: `LLAMA_MEMORY_STATUS_SUCCESS`
- `1`: `LLAMA_MEMORY_STATUS_NO_UPDATE`
- `2`: `LLAMA_MEMORY_STATUS_FAILED_PREPARE`
- `3`: `LLAMA_MEMORY_STATUS_FAILED_COMPUTE`

### 6. CMake Test Integration

**File:** `tests/CMakeLists.txt` (extended)

A new performance test target has been added:

```cmake
llama_test_cmd(
    ${CMAKE_BINARY_DIR}/bin/llama-bench
    NAME test-performance-regression-cpu
    LABEL "performance"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ARGS -p 512 -n 128 -r 3 -o sql
)
```

**Running Performance Tests:**
```bash
cd build
ctest -L performance --verbose
```

## Workflow

### For Pull Requests

1. Developer opens a PR with code changes
2. GitHub Actions triggers the performance regression workflow
3. The workflow:
   - Builds llama-bench with the PR code
   - Restores the baseline database from cache
   - If no baseline exists, creates one from the base commit
   - Runs benchmarks with the current code
   - Compares results using the regression detector
4. Results are posted as a PR comment
5. Build fails if regressions exceed 5% threshold

### For Master Branch Commits

1. Code is merged to master
2. GitHub Actions runs the workflow
3. Benchmark results are cached as the new baseline
4. Historical data is stored in the database
5. Future PRs compare against this baseline

### Manual Baseline Management

**Creating a Baseline:**
```bash
# Run benchmarks
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -r 3 -o sql | sqlite3 baseline.sqlite

# Save as baseline
python3 scripts/apply-db-migration.py -d baseline.sqlite
sqlite3 baseline.sqlite "INSERT INTO performance_baselines (baseline_name, commit_sha, created_at) VALUES ('v1.0', '$(git rev-parse HEAD)', '$(date -Iseconds)')"
```

**Comparing Against Baseline:**
```bash
# Run current benchmarks
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -r 3 -o sql | sqlite3 current.sqlite

# Detect regressions
python3 scripts/performance-regression-detector.py \
  --baseline baseline.sqlite \
  --current current.sqlite \
  --threshold 5.0
```

## Configuration

### Environment Variables

- `REGRESSION_THRESHOLD`: Regression detection threshold (default: 5.0)
- `BASELINE_DB`: Baseline database filename (default: performance-baseline.sqlite)
- `RESULTS_DB`: Results database filename (default: performance-results.sqlite)

### Workflow Customization

Edit `.github/workflows/performance-regression.yml` to:

- Change benchmark parameters (prompt length, generation tokens, repetitions)
- Add/remove backend configurations
- Modify caching strategy
- Adjust model selection

### Threshold Configuration

The default 5% threshold can be adjusted per-backend or per-metric:

```python
# In performance-regression-detector.py
PERFORMANCE_METRICS = {
    "avg_ts": {
        "threshold": 5.0,  # Custom threshold for this metric
        ...
    }
}
```

## Reports

### Regression Report Format

```markdown
# Performance Regression Analysis Report

**Generated:** 2025-09-29 12:34:56
**Threshold:** 5.0%

## Summary
- Total Benchmarks Compared: 10
- Regressions Found: 2
- Improvements Found: 3
- Stable Benchmarks: 5

## ⚠️ Performance Regressions Detected

### TinyLlama-1.1B | backend:CPU | p:512 | g:128

⚠️ **Average Tokens/Second**:
- Baseline: 45.23 tokens/s
- Current: 42.15 tokens/s
- Change: ↓ 6.81%

...
```

### Memory Leak Report Format

```markdown
# Memory Leak Monitoring Report

**Generated:** 2025-09-29 12:34:56

## ⚠️ Memory Leaks Detected

### benchmark
- Initial Memory: 1234.56 MB
- Final Memory: 1250.78 MB
- Leaked: 16.22 MB
```

## Troubleshooting

### No Baseline Available

If the baseline cache is empty or expired:

1. The workflow will attempt to build the baseline from the base commit
2. If that fails, it will create a baseline from the current code
3. Subsequent runs will use this baseline

### False Positives

Regressions can be marked as false positives in the database:

```sql
UPDATE regression_alerts 
SET status = 'false_positive', notes = 'Expected due to architectural change'
WHERE id = <alert_id>;
```

### Excessive Memory Usage Warnings

If memory usage exceeds thresholds:

1. Review the memory leak report
2. Check for memory leaks using valgrind or similar tools
3. Adjust the threshold if legitimate increased usage

## Integration with CI/CD

### GitHub Actions Artifacts

The workflow uploads artifacts containing:
- Regression reports (markdown)
- SQLite databases (baseline and current)
- Memory leak reports

**Downloading Artifacts:**
```bash
gh run download <run-id> -n performance-report-cpu
```

### PR Comments

The workflow automatically comments on PRs with:
- Summary of regression detection
- Links to detailed reports
- Pass/fail status

### Build Status

The workflow sets the build status to:
- ✅ **Success**: No regressions detected
- ❌ **Failure**: Regressions exceed threshold
- ⚠️ **Warning**: Issues detected but below threshold

## Best Practices

1. **Run locally before PR**: Test performance changes locally
2. **Review memory reports**: Check for memory leaks regularly
3. **Update baselines**: Refresh baselines after major changes
4. **Monitor trends**: Use historical data to identify gradual degradation
5. **Document exceptions**: Note expected performance changes in PR descriptions

## Future Enhancements

Potential improvements to the system:

- [ ] Add GPU-specific benchmarks when runners available
- [ ] Implement trend analysis over multiple commits
- [ ] Add visualization dashboard for historical performance
- [ ] Support for custom benchmark configurations per PR
- [ ] Integration with performance profiling tools
- [ ] Automatic bisection for regression identification
- [ ] Multi-model benchmark comparisons

## References

- [llama-bench documentation](../tools/llama-bench/README.md)
- [compare-llama-bench.py usage](../scripts/compare-llama-bench.py)
- [llama-memory.h interface](../src/llama-memory.h)
- [GitHub Actions workflows](../.github/workflows/)

## Support

For issues or questions:
- Check existing GitHub issues
- Review workflow run logs
- Examine generated reports
- Contact the performance testing team
