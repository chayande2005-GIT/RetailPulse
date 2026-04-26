"""
Drift Detection Module using Evidently AI
Monitors data distribution shifts and model performance degradation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

# Try importing Evidently, fallback to None if not available
try:
    from evidently.test_suite import TestSuite  # type: ignore
    from evidently.tests import (  # type: ignore
        TestShareOfOutliersInColumn,
        TestNumberOfMissingValues,
        TestMeanInNSigmas,
        TestColumnShareOfDriftedValues
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

class DriftDetector:
    """Monitor data drift and model performance drift"""
    
    def __init__(self, reference_data, current_data):
        """
        Initialize drift detector
        
        Args:
            reference_data: Historical baseline data (pandas DataFrame)
            current_data: Recent production data (pandas DataFrame)
        """
        self.reference_data = reference_data
        self.current_data = current_data
        self.drift_report = None
        self.timestamp = datetime.now().isoformat()
    
    def detect_data_drift(self, columns_to_monitor=None):
        """
        Detect data drift using Evidently AI (with fallback to statistical method)
        
        Args:
            columns_to_monitor: List of column names to check (optional)
        
        Returns:
            dict with drift detection results
        """
        if not EVIDENTLY_AVAILABLE:
            return self.detect_statistical_drift()
        
        if columns_to_monitor is None:
            columns_to_monitor = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        try:
            # Create test suite
            test_suite = TestSuite(tests=[
                TestShareOfOutliersInColumn(column) for column in columns_to_monitor
            ] + [
                TestNumberOfMissingValues(),
            ] + [
                TestMeanInNSigmas(column) for column in columns_to_monitor[:3]  # Limit for performance
            ])
            
            # Run tests
            test_suite.run(reference_data=self.reference_data, current_data=self.current_data)
            
            # Compile results
            drift_results = {
                "timestamp": self.timestamp,
                "tests_passed": test_suite.summary()["passed"],
                "tests_failed": test_suite.summary()["failed"],
                "total_tests": test_suite.summary()["total"],
                "drift_detected": test_suite.summary()["failed"] > 0
            }
            
            return drift_results
        except Exception as e:
            # Fallback to statistical drift detection if Evidently fails
            return self.detect_statistical_drift()
    
    def detect_statistical_drift(self):
        """
        Detect drift using statistical tests (Kolmogorov-Smirnov)
        
        Returns:
            dict with statistical drift information
        """
        from scipy import stats
        
        drift_info = {
            "timestamp": self.timestamp,
            "drift_details": []
        }
        
        # Check numerical columns
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.current_data.columns:
                ref_vals = self.reference_data[col].dropna()
                curr_vals = self.current_data[col].dropna()
                
                if len(ref_vals) > 0 and len(curr_vals) > 0:
                    # KS test
                    ks_stat, p_value = stats.ks_2samp(ref_vals, curr_vals)
                    
                    # Mean difference
                    mean_diff = (curr_vals.mean() - ref_vals.mean()) / (ref_vals.std() + 1e-6) * 100
                    
                    drift_info["drift_details"].append({
                        "column": col,
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_value),
                        "mean_shift_percent": float(mean_diff),
                        "is_drift": p_value < 0.05
                    })
        
        return drift_info
    
    def save_drift_report(self, filepath="drift_report.json"):
        """
        Save drift detection report to file
        
        Args:
            filepath: Path to save JSON report
        """
        report = {
            "timestamp": self.timestamp,
            "reference_samples": len(self.reference_data),
            "current_samples": len(self.current_data),
            "statistical_drift": self.detect_statistical_drift()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath


def monitor_churn_model_drift(y_true_hist, y_pred_hist, y_true_current, y_pred_current):
    """
    Monitor churn prediction model performance drift
    
    Args:
        y_true_hist: Historical true labels
        y_pred_hist: Historical predictions
        y_true_current: Current true labels
        y_pred_current: Current predictions
    
    Returns:
        dict with drift metrics
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    
    try:
        # Historical metrics
        hist_auc = roc_auc_score(y_true_hist, y_pred_hist) if len(np.unique(y_true_hist)) > 1 else 0
        hist_acc = accuracy_score(y_true_hist, (y_pred_hist > 0.5).astype(int))
        
        # Current metrics
        curr_auc = roc_auc_score(y_true_current, y_pred_current) if len(np.unique(y_true_current)) > 1 else 0
        curr_acc = accuracy_score(y_true_current, (y_pred_current > 0.5).astype(int))
        
        # Drift detection
        auc_drift = abs(curr_auc - hist_auc)
        acc_drift = abs(curr_acc - hist_acc)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "historical_auc": float(hist_auc),
            "current_auc": float(curr_auc),
            "auc_drift": float(auc_drift),
            "historical_accuracy": float(hist_acc),
            "current_accuracy": float(curr_acc),
            "accuracy_drift": float(acc_drift),
            "drift_alert": auc_drift > 0.05 or acc_drift > 0.05  # Alert threshold
        }
    except Exception as e:
        return {"error": str(e), "drift_alert": False}


if __name__ == "__main__":
    """Example usage"""
    print("Drift Detection Module - Use in Streamlit or Airflow DAG")
