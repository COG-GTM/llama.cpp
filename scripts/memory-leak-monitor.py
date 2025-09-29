#!/usr/bin/env python3
"""
Memory Leak Monitoring Integration for llama.cpp

This script integrates with the CI pipeline to monitor memory consumption
patterns using the existing llama_memory_status interfaces from llama-memory.h.

It parses benchmark results and test logs to detect memory leaks and excessive
memory consumption that could indicate performance issues.
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("memory-leak-monitor")

MEMORY_STATUS_CODES = {
    0: "LLAMA_MEMORY_STATUS_SUCCESS",
    1: "LLAMA_MEMORY_STATUS_NO_UPDATE",
    2: "LLAMA_MEMORY_STATUS_FAILED_PREPARE",
    3: "LLAMA_MEMORY_STATUS_FAILED_COMPUTE",
}

LEAK_THRESHOLD_KB = 1024  # 1 MB leak threshold
EXCESSIVE_MEMORY_THRESHOLD_GB = 16  # 16 GB excessive usage threshold


class MemoryLeakMonitor:
    """Monitor memory usage and detect potential leaks."""

    def __init__(self, db_path: str = None):
        """
        Initialize memory leak monitor.

        Args:
            db_path: Optional path to SQLite database for storing results
        """
        self.db_path = db_path
        self.leaks_detected: List[Dict] = []
        self.memory_issues: List[Dict] = []

    def parse_benchmark_output(self, output_file: str) -> List[Dict]:
        """
        Parse benchmark output for memory usage information.

        Args:
            output_file: Path to benchmark output file

        Returns:
            List of memory usage records
        """
        memory_records = []

        if not os.path.exists(output_file):
            logger.warning(f"Output file not found: {output_file}")
            return memory_records

        with open(output_file, 'r') as f:
            content = f.read()


        size_pattern = r'model size:\s+(\d+\.?\d*)\s+(GiB|MiB|GB|MB)'
        usage_pattern = r'memory usage:\s+(\d+)\s+(MB|KB|GB)'
        peak_pattern = r'peak memory:\s+(\d+\.?\d*)\s+(GB|MB)'

        for pattern_name, pattern in [
            ("model_size", size_pattern),
            ("memory_usage", usage_pattern),
            ("peak_memory", peak_pattern)
        ]:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                unit = match.group(2).upper()

                if unit in ["GIB", "GB"]:
                    value_kb = value * 1024 * 1024
                elif unit in ["MIB", "MB"]:
                    value_kb = value * 1024
                else:
                    value_kb = value

                memory_records.append({
                    "type": pattern_name,
                    "value_kb": value_kb,
                    "original_value": match.group(1),
                    "unit": match.group(2)
                })

        logger.info(f"Parsed {len(memory_records)} memory records from {output_file}")
        return memory_records

    def parse_test_logs(self, log_file: str) -> List[Dict]:
        """
        Parse test logs for memory status codes.

        Args:
            log_file: Path to test log file

        Returns:
            List of memory status records
        """
        status_records = []

        if not os.path.exists(log_file):
            logger.warning(f"Log file not found: {log_file}")
            return status_records

        with open(log_file, 'r') as f:
            lines = f.readlines()

        status_pattern = r'memory.*status[:\s]+(\d+)'
        failure_pattern = r'memory.*(?:leak|fail|error)'

        for i, line in enumerate(lines):
            status_match = re.search(status_pattern, line, re.IGNORECASE)
            if status_match:
                status_code = int(status_match.group(1))
                status_name = MEMORY_STATUS_CODES.get(status_code, "UNKNOWN")

                status_records.append({
                    "line_number": i + 1,
                    "status_code": status_code,
                    "status_name": status_name,
                    "line": line.strip(),
                    "is_failure": status_code >= 2
                })

            failure_match = re.search(failure_pattern, line, re.IGNORECASE)
            if failure_match:
                status_records.append({
                    "line_number": i + 1,
                    "status_code": -1,
                    "status_name": "MEMORY_ISSUE_DETECTED",
                    "line": line.strip(),
                    "is_failure": True
                })

        logger.info(f"Parsed {len(status_records)} memory status records from {log_file}")
        return status_records

    def detect_leaks(
        self,
        initial_memory_kb: float,
        final_memory_kb: float,
        test_name: str = "unknown"
    ) -> Optional[Dict]:
        """
        Detect memory leaks by comparing initial and final memory usage.

        Args:
            initial_memory_kb: Initial memory usage in KB
            final_memory_kb: Final memory usage in KB
            test_name: Name of the test

        Returns:
            Leak information if detected, None otherwise
        """
        leaked_kb = final_memory_kb - initial_memory_kb

        if leaked_kb > LEAK_THRESHOLD_KB:
            leak_info = {
                "test_name": test_name,
                "initial_memory_kb": initial_memory_kb,
                "final_memory_kb": final_memory_kb,
                "leaked_memory_kb": leaked_kb,
                "leaked_memory_mb": leaked_kb / 1024,
                "timestamp": datetime.now().isoformat()
            }
            self.leaks_detected.append(leak_info)
            logger.warning(f"Memory leak detected in {test_name}: {leak_info['leaked_memory_mb']:.2f} MB")
            return leak_info

        return None

    def check_excessive_usage(self, memory_kb: float, test_name: str = "unknown") -> bool:
        """
        Check if memory usage exceeds acceptable thresholds.

        Args:
            memory_kb: Memory usage in KB
            test_name: Name of the test

        Returns:
            True if excessive usage detected
        """
        memory_gb = memory_kb / (1024 * 1024)

        if memory_gb > EXCESSIVE_MEMORY_THRESHOLD_GB:
            issue = {
                "test_name": test_name,
                "memory_kb": memory_kb,
                "memory_gb": memory_gb,
                "threshold_gb": EXCESSIVE_MEMORY_THRESHOLD_GB,
                "timestamp": datetime.now().isoformat()
            }
            self.memory_issues.append(issue)
            logger.warning(
                f"Excessive memory usage in {test_name}: "
                f"{memory_gb:.2f} GB (threshold: {EXCESSIVE_MEMORY_THRESHOLD_GB} GB)"
            )
            return True

        return False

    def store_results(self, build_commit: str):
        """
        Store memory monitoring results in database.

        Args:
            build_commit: Git commit SHA
        """
        if not self.db_path:
            logger.warning("No database path configured, skipping storage")
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
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
                )
            """)

            for leak in self.leaks_detected:
                cursor.execute("""
                    INSERT INTO memory_leak_logs (
                        test_time, build_commit, test_name, memory_status,
                        initial_memory_kb, final_memory_kb, leaked_memory_kb,
                        status_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    build_commit,
                    leak["test_name"],
                    "LEAK_DETECTED",
                    int(leak["initial_memory_kb"]),
                    int(leak["final_memory_kb"]),
                    int(leak["leaked_memory_kb"]),
                    -1
                ))

            for issue in self.memory_issues:
                cursor.execute("""
                    INSERT INTO memory_leak_logs (
                        test_time, build_commit, test_name, memory_status,
                        peak_memory_kb, status_code
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    build_commit,
                    issue["test_name"],
                    "EXCESSIVE_USAGE",
                    int(issue["memory_kb"]),
                    -2
                ))

            conn.commit()
            conn.close()
            logger.info(f"Stored {len(self.leaks_detected)} leak records and "
                       f"{len(self.memory_issues)} excessive usage records")

        except sqlite3.Error as e:
            logger.error(f"Error storing results: {e}")

    def generate_report(self, output_file: str):
        """
        Generate a markdown report of memory monitoring results.

        Args:
            output_file: Path to output markdown file
        """
        with open(output_file, 'w') as f:
            f.write("# Memory Leak Monitoring Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if self.leaks_detected:
                f.write("## ⚠️ Memory Leaks Detected\n\n")
                f.write(f"**Total Leaks:** {len(self.leaks_detected)}\n\n")

                for leak in self.leaks_detected:
                    f.write(f"### {leak['test_name']}\n\n")
                    f.write(f"- **Initial Memory:** {leak['initial_memory_kb'] / 1024:.2f} MB\n")
                    f.write(f"- **Final Memory:** {leak['final_memory_kb'] / 1024:.2f} MB\n")
                    f.write(f"- **Leaked:** {leak['leaked_memory_mb']:.2f} MB\n\n")
            else:
                f.write("## ✅ No Memory Leaks Detected\n\n")

            if self.memory_issues:
                f.write("## ⚠️ Excessive Memory Usage\n\n")
                f.write(f"**Total Issues:** {len(self.memory_issues)}\n\n")

                for issue in self.memory_issues:
                    f.write(f"### {issue['test_name']}\n\n")
                    f.write(f"- **Memory Used:** {issue['memory_gb']:.2f} GB\n")
                    f.write(f"- **Threshold:** {issue['threshold_gb']} GB\n\n")

        logger.info(f"Report written to {output_file}")


def main():
    """Main entry point for memory leak monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor memory usage and detect leaks in llama.cpp benchmarks"
    )
    parser.add_argument(
        "--benchmark-output",
        help="Path to benchmark output file to analyze"
    )
    parser.add_argument(
        "--test-log",
        help="Path to test log file to analyze"
    )
    parser.add_argument(
        "--database",
        help="Path to SQLite database for storing results"
    )
    parser.add_argument(
        "--commit",
        default="unknown",
        help="Git commit SHA for this run"
    )
    parser.add_argument(
        "--report",
        default="memory-report.md",
        help="Output path for memory report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    monitor = MemoryLeakMonitor(db_path=args.database)

    if args.benchmark_output:
        memory_records = monitor.parse_benchmark_output(args.benchmark_output)
        if len(memory_records) >= 2:
            initial = memory_records[0]["value_kb"]
            final = memory_records[-1]["value_kb"]
            monitor.detect_leaks(initial, final, "benchmark")

    if args.test_log:
        status_records = monitor.parse_test_logs(args.test_log)
        for record in status_records:
            if record.get("is_failure"):
                logger.error(f"Memory failure at line {record['line_number']}: {record['line']}")

    if args.database:
        monitor.store_results(args.commit)

    monitor.generate_report(args.report)

    has_issues = bool(monitor.leaks_detected or monitor.memory_issues)
    sys.exit(1 if has_issues else 0)


if __name__ == "__main__":
    main()
