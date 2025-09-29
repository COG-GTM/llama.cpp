#!/usr/bin/env python3
"""
Performance Regression Detector for llama.cpp

This script compares benchmark results between baseline and current runs,
detecting performance regressions above a configurable threshold.

It integrates with the existing llama-bench SQLite database schema and
provides automated alerts for CI/CD pipelines.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LLAMA_BENCH_DB_FIELDS = [
    "build_commit", "build_number", "cpu_info",       "gpu_info",   "backends",     "model_filename",
    "model_type",   "model_size",   "model_n_params", "n_batch",    "n_ubatch",     "n_threads",
    "cpu_mask",     "cpu_strict",   "poll",           "type_k",     "type_v",       "n_gpu_layers",
    "split_mode",   "main_gpu",     "no_kv_offload",  "flash_attn", "tensor_split", "tensor_buft_overrides",
    "use_mmap",     "embeddings",   "no_op_offload",  "n_prompt",   "n_gen",        "n_depth",
    "test_time",    "avg_ns",       "stddev_ns",      "avg_ts",     "stddev_ts",
]

BENCHMARK_KEY_PROPERTIES = [
    "model_type", "n_batch", "n_ubatch", "n_threads", "n_gpu_layers",
    "backends", "n_prompt", "n_gen", "flash_attn"
]

PERFORMANCE_METRICS = {
    "avg_ts": {
        "name": "Average Tokens/Second",
        "unit": "tokens/s",
        "direction": "higher_is_better",
        "format": "{:.2f}"
    },
    "avg_ns": {
        "name": "Average Latency",
        "unit": "ns",
        "direction": "lower_is_better",
        "format": "{:.0f}"
    },
    "model_size": {
        "name": "Model Size",
        "unit": "bytes",
        "direction": "lower_is_better",
        "format": "{:.0f}"
    }
}

logger = logging.getLogger("performance-regression-detector")


class RegressionDetector:
    """Detects performance regressions by comparing benchmark results."""

    def __init__(self, baseline_db: str, current_db: str, threshold: float = 5.0):
        """
        Initialize the regression detector.

        Args:
            baseline_db: Path to baseline SQLite database
            current_db: Path to current run SQLite database
            threshold: Regression threshold percentage (default: 5.0)
        """
        self.baseline_db = baseline_db
        self.current_db = current_db
        self.threshold = threshold
        self.regressions: List[Dict[str, Any]] = []
        self.improvements: List[Dict[str, Any]] = []
        self.stable: List[Dict[str, Any]] = []

    def load_results(self, db_path: str) -> List[Dict[str, Any]]:
        """Load benchmark results from SQLite database."""
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM test")
            results = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Loaded {len(results)} results from {db_path}")
            return results
        except sqlite3.OperationalError as e:
            logger.error(f"Error reading database {db_path}: {e}")
            return []
        finally:
            conn.close()

    def match_benchmark(
        self, baseline: Dict[str, Any], current_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find matching benchmark in current results based on key properties.

        Args:
            baseline: Baseline benchmark result
            current_results: List of current benchmark results

        Returns:
            Matching benchmark or None if no match found
        """
        for current in current_results:
            match = True
            for key in BENCHMARK_KEY_PROPERTIES:
                if key not in baseline or key not in current:
                    continue
                if baseline[key] != current[key]:
                    match = False
                    break
            if match:
                return current
        return None

    def calculate_regression(
        self, metric_name: str, baseline_value: float, current_value: float
    ) -> Tuple[float, bool]:
        """
        Calculate regression percentage and determine if it exceeds threshold.

        Args:
            metric_name: Name of the metric being compared
            baseline_value: Baseline metric value
            current_value: Current metric value

        Returns:
            Tuple of (change_percentage, is_regression)
        """
        if baseline_value == 0:
            return 0.0, False

        metric_info = PERFORMANCE_METRICS.get(metric_name, {})
        direction = metric_info.get("direction", "higher_is_better")

        change_pct = ((current_value - baseline_value) / baseline_value) * 100

        if direction == "higher_is_better":
            is_regression = change_pct < -self.threshold
        else:
            is_regression = change_pct > self.threshold

        return change_pct, is_regression

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze benchmark results and detect regressions.

        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting regression analysis...")

        baseline_results = self.load_results(self.baseline_db)
        current_results = self.load_results(self.current_db)

        if not baseline_results:
            logger.warning("No baseline results found - skipping comparison")
            return {
                "status": "no_baseline",
                "message": "No baseline results available for comparison",
                "regressions": [],
                "improvements": [],
                "stable": [],
                "summary": {
                    "total_benchmarks": 0,
                    "regressions_found": 0,
                    "improvements_found": 0,
                    "stable_benchmarks": 0
                }
            }

        if not current_results:
            logger.error("No current results found")
            return {
                "status": "error",
                "message": "No current results found",
                "regressions": [],
                "improvements": [],
                "stable": [],
                "summary": {
                    "total_benchmarks": len(baseline_results),
                    "regressions_found": 0,
                    "improvements_found": 0,
                    "stable_benchmarks": 0
                }
            }

        for baseline in baseline_results:
            current = self.match_benchmark(baseline, current_results)
            if not current:
                logger.debug(f"No matching current result for baseline: {baseline.get('model_type')}")
                continue

            benchmark_key = self._generate_benchmark_key(baseline)
            has_regression = False
            has_improvement = False
            changes = {}

            for metric_name in ["avg_ts", "avg_ns"]:
                if metric_name not in baseline or metric_name not in current:
                    continue

                baseline_value = baseline[metric_name]
                current_value = current[metric_name]

                if baseline_value is None or current_value is None:
                    continue

                change_pct, is_regression = self.calculate_regression(
                    metric_name, baseline_value, current_value
                )

                metric_info = PERFORMANCE_METRICS[metric_name]
                changes[metric_name] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "change_pct": change_pct,
                    "is_regression": is_regression,
                    "unit": metric_info["unit"],
                    "name": metric_info["name"]
                }

                if is_regression:
                    has_regression = True
                elif abs(change_pct) > self.threshold:
                    has_improvement = True

            result = {
                "benchmark_key": benchmark_key,
                "baseline": baseline,
                "current": current,
                "changes": changes
            }

            if has_regression:
                self.regressions.append(result)
            elif has_improvement:
                self.improvements.append(result)
            else:
                self.stable.append(result)

        status = "regression" if self.regressions else "pass"

        return {
            "status": status,
            "threshold": self.threshold,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "stable": self.stable,
            "summary": {
                "total_benchmarks": len(baseline_results),
                "regressions_found": len(self.regressions),
                "improvements_found": len(self.improvements),
                "stable_benchmarks": len(self.stable)
            }
        }

    def _generate_benchmark_key(self, benchmark: Dict[str, Any]) -> str:
        """Generate a human-readable key for a benchmark."""
        parts = []
        if "model_type" in benchmark:
            parts.append(benchmark["model_type"])
        if "backends" in benchmark:
            parts.append(f"backend:{benchmark['backends']}")
        if "n_gpu_layers" in benchmark and benchmark["n_gpu_layers"]:
            parts.append(f"ngl:{benchmark['n_gpu_layers']}")
        if "n_prompt" in benchmark:
            parts.append(f"p:{benchmark['n_prompt']}")
        if "n_gen" in benchmark:
            parts.append(f"g:{benchmark['n_gen']}")
        return " | ".join(parts) if parts else "unknown"

    def generate_report(self, output_path: str, analysis: Dict[str, Any]):
        """Generate a markdown report of the regression analysis."""
        with open(output_path, "w") as f:
            f.write("# Performance Regression Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Threshold:** {self.threshold}%\n\n")

            summary = analysis["summary"]
            f.write("## Summary\n\n")
            f.write(f"- **Total Benchmarks Compared:** {summary['total_benchmarks']}\n")
            f.write(f"- **Regressions Found:** {summary['regressions_found']}\n")
            f.write(f"- **Improvements Found:** {summary['improvements_found']}\n")
            f.write(f"- **Stable Benchmarks:** {summary['stable_benchmarks']}\n\n")

            if analysis["status"] == "regression":
                f.write("## ⚠️ Performance Regressions Detected\n\n")
                for reg in analysis["regressions"]:
                    self._write_benchmark_section(f, reg, "Regression")
            elif analysis["status"] == "no_baseline":
                f.write("## ℹ️ No Baseline Available\n\n")
                f.write(analysis["message"] + "\n\n")
            else:
                f.write("## ✅ No Performance Regressions Detected\n\n")

            if analysis["improvements"]:
                f.write("## 📈 Performance Improvements\n\n")
                for imp in analysis["improvements"]:
                    self._write_benchmark_section(f, imp, "Improvement")

            if analysis.get("stable"):
                f.write("## 📊 Stable Performance\n\n")
                f.write(f"**{len(analysis['stable'])} benchmarks** showed stable performance ")
                f.write(f"(within ±{self.threshold}% threshold).\n\n")

        logger.info(f"Report written to {output_path}")

    def _write_benchmark_section(self, f, result: Dict[str, Any], section_type: str):
        """Write a benchmark comparison section to the report."""
        f.write(f"### {result['benchmark_key']}\n\n")

        for metric_name, change in result["changes"].items():
            if not change.get("is_regression") and section_type == "Regression":
                continue
            if change.get("is_regression") and section_type == "Improvement":
                continue

            baseline_val = change["baseline"]
            current_val = change["current"]
            change_pct = change["change_pct"]
            unit = change["unit"]
            name = change["name"]

            icon = "⚠️" if change.get("is_regression") else "✅"
            direction = "↓" if change_pct < 0 else "↑"

            f.write(f"{icon} **{name}**:\n")
            f.write(f"- Baseline: {baseline_val:.2f} {unit}\n")
            f.write(f"- Current: {current_val:.2f} {unit}\n")
            f.write(f"- Change: {direction} {abs(change_pct):.2f}%\n\n")


def main():
    """Main entry point for the regression detector."""
    parser = argparse.ArgumentParser(
        description="Detect performance regressions in llama.cpp benchmarks"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline SQLite database"
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current run SQLite database"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Regression threshold percentage (default: 5.0)"
    )
    parser.add_argument(
        "--output",
        default="regression-report.md",
        help="Output path for regression report (default: regression-report.md)"
    )
    parser.add_argument(
        "--json-output",
        help="Optional JSON output path for machine-readable results"
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

    detector = RegressionDetector(args.baseline, args.current, args.threshold)
    analysis = detector.analyze()

    detector.generate_report(args.output, analysis)

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"JSON report written to {args.json_output}")

    if analysis["status"] == "regression":
        Path("regression-detected.flag").touch()
        logger.error(f"Performance regression detected: {len(analysis['regressions'])} benchmarks affected")
        sys.exit(1)
    else:
        logger.info("No performance regressions detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
