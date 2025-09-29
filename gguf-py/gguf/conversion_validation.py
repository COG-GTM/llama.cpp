#!/usr/bin/env python3
"""
Quantization conversion accuracy validation utilities for GGUF format conversions.
Provides functions to validate accuracy after converting from HuggingFace to GGUF format.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any
from pathlib import Path

logger = logging.getLogger("gguf-validation")


def calculate_rmse(original: np.ndarray, converted: np.ndarray) -> float:
    """Calculate Root Mean Square Error between original and converted tensors."""
    if original.shape != converted.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {converted.shape}")
    
    diff = original.astype(np.float64) - converted.astype(np.float64)
    mse = np.mean(diff ** 2)
    return np.sqrt(mse)


def calculate_max_error(original: np.ndarray, converted: np.ndarray) -> float:
    """Calculate maximum absolute error between original and converted tensors."""
    if original.shape != converted.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {converted.shape}")
    
    diff = np.abs(original.astype(np.float64) - converted.astype(np.float64))
    return np.max(diff)


def validate_tensor_conversion(
    tensor_name: str,
    original_data: np.ndarray,
    converted_data: np.ndarray,
    max_rmse_threshold: float = 0.01,
    max_error_threshold: float = 0.1,
    verbose: bool = False
) -> tuple[bool, dict[str, float]]:
    """
    Validate accuracy of a single tensor conversion.
    
    Args:
        tensor_name: Name of the tensor being validated
        original_data: Original tensor data
        converted_data: Converted tensor data (after GGUF conversion)
        max_rmse_threshold: Maximum allowed RMSE
        max_error_threshold: Maximum allowed absolute error
        verbose: Whether to print detailed validation results
    
    Returns:
        Tuple of (passed: bool, metrics: dict)
    """
    try:
        rmse = calculate_rmse(original_data, converted_data)
        max_err = calculate_max_error(original_data, converted_data)
        
        passed = rmse <= max_rmse_threshold and max_err <= max_error_threshold
        
        metrics = {
            "rmse": float(rmse),
            "max_error": float(max_err),
            "rmse_threshold": max_rmse_threshold,
            "max_error_threshold": max_error_threshold,
            "passed": passed
        }
        
        if verbose or not passed:
            status = "✓" if passed else "✗"
            logger.info(
                f"{status} {tensor_name}: RMSE={rmse:.6f} (threshold={max_rmse_threshold}), "
                f"MaxErr={max_err:.6f} (threshold={max_error_threshold})"
            )
        
        return passed, metrics
    
    except Exception as e:
        logger.error(f"Error validating {tensor_name}: {e}")
        return False, {"error": str(e)}


def validate_model_conversion(
    original_tensors: dict[str, np.ndarray],
    converted_tensors: dict[str, np.ndarray],
    quantization_type: str = "f16",
    verbose: bool = False
) -> dict[str, Any]:
    """
    Validate accuracy of entire model conversion.
    
    Args:
        original_tensors: Dictionary of original tensor names to data
        converted_tensors: Dictionary of converted tensor names to data
        quantization_type: Type of quantization used (affects thresholds)
        verbose: Whether to print detailed validation results
    
    Returns:
        Dictionary with validation results and statistics
    """
    thresholds = get_quantization_thresholds(quantization_type)
    
    results = {
        "total_tensors": 0,
        "passed_tensors": 0,
        "failed_tensors": [],
        "metrics": {},
        "overall_passed": True
    }
    
    common_tensors = set(original_tensors.keys()) & set(converted_tensors.keys())
    
    if not common_tensors:
        logger.warning("No common tensors found between original and converted models")
        results["overall_passed"] = False
        return results
    
    results["total_tensors"] = len(common_tensors)
    
    for tensor_name in sorted(common_tensors):
        passed, metrics = validate_tensor_conversion(
            tensor_name,
            original_tensors[tensor_name],
            converted_tensors[tensor_name],
            max_rmse_threshold=thresholds["rmse"],
            max_error_threshold=thresholds["max_error"],
            verbose=verbose
        )
        
        results["metrics"][tensor_name] = metrics
        
        if passed:
            results["passed_tensors"] += 1
        else:
            results["failed_tensors"].append(tensor_name)
            results["overall_passed"] = False
    
    if verbose:
        logger.info(
            f"\nValidation Summary: {results['passed_tensors']}/{results['total_tensors']} tensors passed"
        )
        if results["failed_tensors"]:
            logger.warning(f"Failed tensors: {', '.join(results['failed_tensors'])}")
    
    return results


def get_quantization_thresholds(quantization_type: str) -> dict[str, float]:
    """
    Get appropriate error thresholds for different quantization types.
    
    Args:
        quantization_type: Type of quantization (f32, f16, q4_0, q8_0, etc.)
    
    Returns:
        Dictionary with "rmse" and "max_error" thresholds
    """
    thresholds_map = {
        "f32": {"rmse": 1e-6, "max_error": 1e-5},
        "f16": {"rmse": 1e-3, "max_error": 1e-2},
        "bf16": {"rmse": 1e-2, "max_error": 1e-1},
        "q8_0": {"rmse": 2e-3, "max_error": 2e-2},
        "q4_0": {"rmse": 1e-2, "max_error": 1e-1},
        "q4_1": {"rmse": 1e-2, "max_error": 1e-1},
        "q5_0": {"rmse": 8e-3, "max_error": 8e-2},
        "q5_1": {"rmse": 8e-3, "max_error": 8e-2},
        "q2_k": {"rmse": 2e-2, "max_error": 2e-1},
        "q3_k": {"rmse": 1.5e-2, "max_error": 1.5e-1},
        "q4_k": {"rmse": 1e-2, "max_error": 1e-1},
        "q5_k": {"rmse": 8e-3, "max_error": 8e-2},
        "q6_k": {"rmse": 5e-3, "max_error": 5e-2},
    }
    
    default = {"rmse": 1e-2, "max_error": 1e-1}
    
    return thresholds_map.get(quantization_type.lower(), default)


def save_validation_report(results: dict[str, Any], output_path: Path) -> None:
    """
    Save validation results to a JSON file.
    
    Args:
        results: Validation results dictionary
        output_path: Path to save the report
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Validation report saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("GGUF Conversion Validation Utilities")
    logger.info("This module provides functions for validating HuggingFace to GGUF conversions")
    logger.info("Import this module in convert_hf_to_gguf.py to enable validation")
