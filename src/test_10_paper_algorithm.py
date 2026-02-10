#!/usr/bin/env python
"""
Test RAG System on 10 Sample Scenarios using Paper Algorithm
Uses trained Gradient Boosting model + Threshold validation
No API calls required - fully local inference
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Tuple


class PaperAlgorithmPredictor:
    """
    Deployment predictor using paper's ML + Threshold algorithm.
    Uses trained Gradient Boosting model with threshold validation.
    """
    
    def __init__(self, model_dir: str = "network_faiss_store"):
        self.model_dir = Path(__file__).parent.parent / model_dir
        self.ml_model = None
        self.label_encoder = None
        self.feature_cols = ['datarate_mbps', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
        
        # Load trained model
        self._load_model()
    
    def _load_model(self):
        """Load trained model from disk."""
        
        model_path = self.model_dir / "gradient_boosting_model.pkl"
        encoder_path = self.model_dir / "label_encoder.pkl"
        
        if not model_path.exists() or not encoder_path.exists():
            raise FileNotFoundError(f"Trained model not found in {self.model_dir}. Please run train_paper_algorithm.py first.")
        
        with open(model_path, 'rb') as f:
            self.ml_model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"✓ Loaded trained model from {self.model_dir}")
        print(f"  Model Type: GradientBoostingClassifier")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
    
    def predict_deployment(self, metrics: Dict) -> Tuple[str, float, str]:
        """
        Predict deployment using ML + Threshold strategy (Paper Algorithm).
        
        Returns: (layer, confidence, reasoning)
        """
        
        # Prepare features as DataFrame to avoid sklearn warning
        import warnings
        warnings.filterwarnings('ignore')
        
        features = pd.DataFrame([[
            metrics.get('datarate_mbps', 0),
            metrics.get('sinr_db', 0),
            metrics.get('latency_ms', 0),
            metrics.get('rsrp_dbm', -100),
            metrics.get('cpu_demand', 50),
            metrics.get('memory_mb', 500)
        ]], columns=self.feature_cols)
        
        # ML Prediction
        proba = self.ml_model.predict_proba(features)[0]
        ml_pred_idx = np.argmax(proba)
        ml_pred = self.label_encoder.classes_[ml_pred_idx]
        ml_confidence = proba[ml_pred_idx]
        
        # Extract metrics for threshold rules
        latency = metrics.get('latency_ms', 0)
        datarate = metrics.get('datarate_mbps', 0)
        cpu = metrics.get('cpu_demand', 50)
        rsrp = metrics.get('rsrp_dbm', -100)
        sinr = metrics.get('sinr_db', 10)
        
        # Initialize
        final_pred = ml_pred
        reasoning = f"ML predicted {ml_pred} ({ml_confidence:.0%} confidence). "
        
        # === THRESHOLD OVERRIDE RULES (From Paper) ===
        
        # Rule 1: High latency + High CPU -> Cloud
        if latency > 150 and cpu > 70:
            final_pred = "Cloud"
            reasoning += f"Override→Cloud: High latency ({latency:.0f}ms) + high CPU ({cpu}%)."
        
        # Rule 2: Ultra-low latency + High bandwidth + Low CPU -> Edge
        elif latency < 30 and datarate > 40 and cpu < 30:
            final_pred = "Edge"
            reasoning += f"Override→Edge: Ultra-low latency ({latency:.0f}ms), high bandwidth ({datarate:.0f}Mbps)."
        
        # Rule 3: Poor signal -> Cloud for reliability
        elif rsrp < -120:
            final_pred = "Cloud"
            reasoning += f"Override→Cloud: Poor signal (RSRP={rsrp:.0f}dBm)."
        
        # Rule 4: High interference -> Prefer Fog/Cloud
        elif sinr < 5:
            if final_pred == "Edge":
                final_pred = "Fog"
                reasoning += f"Override→Fog: High interference (SINR={sinr:.1f}dB)."
        
        # Rule 5: Low ML confidence -> Use pure threshold rules
        elif ml_confidence < 0.5:
            if latency < 50 and datarate > 30:
                final_pred = "Edge"
            elif latency < 200 and datarate > 15:
                final_pred = "Fog"
            else:
                final_pred = "Cloud"
            reasoning += f"Low confidence→Threshold rules: {final_pred}."
        else:
            reasoning += "Validated by threshold rules."
        
        return final_pred, ml_confidence, reasoning


def load_real_data():
    """Load real ai4mobile dataset scenarios."""
    
    print("\n" + "="*80)
    print("LOADING REAL NETWORK DATA")
    print("="*80)
    
    workspace_root = Path(__file__).parent.parent.parent.parent
    csv_path = workspace_root / "data" / "edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Normalize datarate
        df['datarate_mbps'] = df['datarate'] / 1e6
        print(f"\n📂 Loaded: {csv_path.name}")
        print(f"   Total Scenarios: {len(df)}")
        return df
    
    return None


def extract_metrics(row) -> Dict:
    """Extract metrics from dataframe row."""
    return {
        "datarate_mbps": float(row["datarate_mbps"]) if pd.notna(row.get("datarate_mbps", 0)) else float(row["datarate"]) / 1e6,
        "latency_ms": float(row["latency_ms"]) if pd.notna(row["latency_ms"]) else 0,
        "sinr_db": float(row["sinr"]) if pd.notna(row["sinr"]) else 0,
        "rsrp_dbm": float(row["rsrp_dbm"]) if pd.notna(row["rsrp_dbm"]) else -120,
        "cpu_demand": float(row["cpu_demand"]) if pd.notna(row["cpu_demand"]) else 50,
        "memory_mb": float(row["memory_demand"]) if pd.notna(row["memory_demand"]) else 500,
    }


def test_10_scenarios():
    """Test RAG system on 10 sample scenarios using paper algorithm."""
    
    print("\n" + "*"*80)
    print("* RAG SYSTEM - 10 SAMPLE SCENARIOS WITH PAPER ALGORITHM")
    print("* Model: Gradient Boosting + Threshold Validation")
    print("*"*80)
    
    # Load data
    df = load_real_data()
    
    if df is None or len(df) == 0:
        print("\n❌ No real data found!")
        return
    
    # Use only 10 scenarios
    num_scenarios = min(10, len(df))
    df = df.iloc[:num_scenarios]
    
    print(f"\n📊 Using {num_scenarios} scenarios")
    
    # Initialize predictor
    print("\n" + "="*80)
    print("INITIALIZING PREDICTOR")
    print("="*80)
    predictor = PaperAlgorithmPredictor()
    print("  Decision Method: ML + Threshold (Paper Algorithm)")
    print("  Scenarios to Test: 10")
    
    # Test scenarios
    print("\n" + "="*80)
    print("TESTING ON 10 SAMPLE SCENARIOS")
    print("="*80)
    
    test_results = []
    layer_distribution = defaultdict(int)
    latency_stats = []
    datarate_stats = []
    correct_predictions = 0
    
    print(f"\nProcessing {num_scenarios} scenarios...\n")
    
    for idx, row in df.iterrows():
        # Extract metrics
        metrics = extract_metrics(row)
        
        # Get prediction
        recommendation, confidence, reasoning = predictor.predict_deployment(metrics)
        
        # Check if matches original assignment
        actual_layer = row['assigned_layer']
        is_correct = recommendation == actual_layer
        if is_correct:
            correct_predictions += 1
        
        # Store statistics
        latency_stats.append(metrics['latency_ms'])
        datarate_stats.append(metrics['datarate_mbps'])
        layer_distribution[recommendation] += 1
        
        # Store result
        result = {
            "scenario_id": idx,
            "datarate_mbps": metrics['datarate_mbps'],
            "latency_ms": metrics['latency_ms'],
            "sinr_db": metrics['sinr_db'],
            "rsrp_dbm": metrics['rsrp_dbm'],
            "cpu_demand": metrics['cpu_demand'],
            "memory_mb": metrics['memory_mb'],
            "ml_recommendation": recommendation,
            "confidence": confidence,
            "reasoning": reasoning,
            "actual_layer": actual_layer,
            "is_correct": is_correct
        }
        test_results.append(result)
        
        # Print result
        status = "✓" if is_correct else "✗"
        print(f"  [{idx + 1}/10] {status} Latency={metrics['latency_ms']:.1f}ms, DataRate={metrics['datarate_mbps']:.1f}Mbps")
        print(f"         Predicted: {recommendation}, Actual: {actual_layer}, Confidence: {confidence:.0%}")
        print(f"         {reasoning}")
        print()
    
    # Statistics
    accuracy = (correct_predictions / num_scenarios) * 100
    
    print("\n" + "="*80)
    print("TEST STATISTICS")
    print("="*80)
    
    stats = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_tests": num_scenarios,
        "correct_predictions": correct_predictions,
        "accuracy": f"{accuracy:.1f}%",
        "decision_method": "ML + Threshold (Paper Algorithm)",
        "model": "GradientBoostingClassifier",
        "deployment_distribution": dict(layer_distribution),
        "latency_stats": {
            "min": float(np.min(latency_stats)),
            "max": float(np.max(latency_stats)),
            "mean": float(np.mean(latency_stats)),
            "median": float(np.median(latency_stats)),
        },
        "datarate_stats": {
            "min": float(np.min(datarate_stats)),
            "max": float(np.max(datarate_stats)),
            "mean": float(np.mean(datarate_stats)),
            "median": float(np.median(datarate_stats)),
        }
    }
    
    print(f"\n✅ Tested {num_scenarios} scenarios!")
    print(f"   Correct Predictions: {correct_predictions}/{num_scenarios}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    print(f"\nDeployment Distribution:")
    for layer, count in layer_distribution.items():
        pct = (count / num_scenarios) * 100
        print(f"  {layer:6s}: {count:4d} scenarios ({pct:5.1f}%)")
    
    print(f"\nLatency Statistics (ms):")
    print(f"  Min:    {stats['latency_stats']['min']:.2f} ms")
    print(f"  Max:    {stats['latency_stats']['max']:.2f} ms")
    print(f"  Mean:   {stats['latency_stats']['mean']:.2f} ms")
    print(f"  Median: {stats['latency_stats']['median']:.2f} ms")
    
    print(f"\nDatarate Statistics (Mbps):")
    print(f"  Min:    {stats['datarate_stats']['min']:.2f} Mbps")
    print(f"  Max:    {stats['datarate_stats']['max']:.2f} Mbps")
    print(f"  Mean:   {stats['datarate_stats']['mean']:.2f} Mbps")
    print(f"  Median: {stats['datarate_stats']['median']:.2f} Mbps")
    
    # Save results
    save_results(test_results, stats)
    
    print("\n" + "*"*80)
    print(f"* TEST COMPLETED: {accuracy:.1f}% accuracy on {num_scenarios} scenarios")
    print("*"*80 + "\n")


def save_results(test_results, stats):
    """Save test results."""
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = stats["timestamp"]
    
    # 1. Full JSON results
    results_file = results_dir / f"paper_algorithm_10_scenarios_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "metadata": stats,
            "results": test_results
        }, f, indent=2)
    print(f"\n✓ Full results: {results_file.name}")
    
    # 2. Statistics summary
    stats_file = results_dir / f"paper_algorithm_10_statistics_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics: {stats_file.name}")
    
    # 3. CSV with all results
    csv_file = results_dir / f"paper_algorithm_10_results_{timestamp}.csv"
    df_results = pd.DataFrame(test_results)
    df_results.to_csv(csv_file, index=False)
    print(f"✓ CSV Export: {csv_file.name}")
    
    # 4. Text Report
    report_file = results_dir / f"paper_algorithm_10_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAG SYSTEM - 10 SAMPLE SCENARIOS - PAPER ALGORITHM REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("ALGORITHM OVERVIEW\n")
        f.write("-"*40 + "\n")
        f.write("Paper: Adaptive Task Scheduling in Edge-Fog-Cloud\n")
        f.write("       with Network Failure Resilience\n\n")
        f.write("Method: ML + Threshold (Gradient Boosting)\n")
        f.write("  1. Gradient Boosting predicts deployment layer\n")
        f.write("  2. Threshold rules validate/override prediction\n")
        f.write("  3. Combines ML learning with domain expertise\n\n")
        
        f.write("TEST OVERVIEW\n")
        f.write("-"*40 + "\n")
        f.write(f"Timestamp: {stats['timestamp']}\n")
        f.write(f"Total Scenarios: {stats['total_tests']}\n")
        f.write(f"Correct Predictions: {stats['correct_predictions']}\n")
        f.write(f"Accuracy: {stats['accuracy']}\n")
        f.write(f"Decision Method: {stats['decision_method']}\n\n")
        
        f.write("DEPLOYMENT DISTRIBUTION\n")
        f.write("-"*40 + "\n")
        for layer, count in stats['deployment_distribution'].items():
            pct = (count / stats['total_tests']) * 100 if stats['total_tests'] > 0 else 0
            f.write(f"{layer:10s}: {count:5d} scenarios ({pct:5.1f}%)\n")
        
        f.write("\nNETWORK METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Latency (ms): {stats['latency_stats']['min']:.2f} - {stats['latency_stats']['max']:.2f}\n")
        f.write(f"Datarate (Mbps): {stats['datarate_stats']['min']:.2f} - {stats['datarate_stats']['max']:.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for i, result in enumerate(test_results, 1):
            status = "CORRECT" if result['is_correct'] else "INCORRECT"
            f.write(f"Scenario {i}: [{status}]\n")
            f.write(f"  Metrics: Latency={result['latency_ms']:.2f}ms, DataRate={result['datarate_mbps']:.2f}Mbps\n")
            f.write(f"  Signal: SINR={result['sinr_db']:.2f}dB, RSRP={result['rsrp_dbm']:.2f}dBm\n")
            f.write(f"  Resources: CPU={result['cpu_demand']:.0f}%, Memory={result['memory_mb']:.0f}MB\n")
            f.write(f"  Predicted: {result['ml_recommendation']} | Actual: {result['actual_layer']}\n")
            f.write(f"  Confidence: {result['confidence']:.0%}\n")
            f.write(f"  Reasoning: {result['reasoning']}\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Text Report: {report_file.name}")
    print(f"\n✓ All results saved to: {results_dir.name}/")


if __name__ == "__main__":
    test_10_scenarios()
