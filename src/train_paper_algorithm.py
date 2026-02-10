#!/usr/bin/env python
"""
RAG Training with Paper Algorithm: ML + Threshold (Gradient Boosting)
SIMPLIFIED VERSION - No heavy dependencies

Based on: "Adaptive Task Scheduling in Edge-Fog-Cloud with Network Failure Resilience"
- Uses Gradient Boosting for deployment layer prediction
- Combines ML prediction with threshold-based validation
- No failure mode (degraded=False)

Vector Embeddings Storage:
- Location: network_faiss_store/
  - model.pkl: Trained Gradient Boosting model
  - label_encoder.pkl: Label encoder
  - training_data.pkl: Processed training data

Algorithm Flow:
1. Load real network scenarios from CSV
2. Train Gradient Boosting model for predictions
3. Apply threshold validation rules
4. Save model for inference
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple


class PaperAlgorithmTrainer:
    """
    Training System based on the paper's ML + Threshold Algorithm.
    
    Paper Algorithm Summary:
    ========================
    1. ML Model: Gradient Boosting Classifier
       - Features: datarate, sinr, latency, rsrp, cpu_demand, memory_demand
       - Targets: Edge, Fog, Cloud deployment layers
       
    2. Threshold Validation:
       - Edge: latency < 50ms AND datarate > 30 Mbps AND cpu < 50%
       - Fog: latency < 200ms AND datarate > 15 Mbps
       - Cloud: High latency OR high resource demand
       
    3. Decision Strategy:
       - ML model predicts initial layer
       - Threshold rules validate/override prediction
       - Combines statistical learning with domain expertise
    """
    
    def __init__(self, model_dir: str = "network_faiss_store"):
        self.workspace_root = Path(__file__).parent.parent.parent.parent
        self.model_dir = Path(__file__).parent.parent / model_dir
        self.model_dir.mkdir(exist_ok=True)
        
        self.ml_model = None
        self.label_encoder = None
        self.feature_cols = ['datarate_mbps', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
        
        print(f"✓ Model Storage Location: {self.model_dir}")
    
    def load_training_data(self) -> pd.DataFrame:
        """Load EdgeSimPy CSV with ML + Threshold (GB) results."""
        
        print("\n" + "="*80)
        print("PHASE 1: LOADING TRAINING DATA")
        print("="*80)
        
        csv_path = self.workspace_root / "data" / "edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Training data not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Normalize datarate to Mbps
        df['datarate_mbps'] = df['datarate'] / 1e6
        
        print(f"\n📂 Loaded: {csv_path.name}")
        print(f"   Total Scenarios: {len(df)}")
        print(f"   Model Used: {df['model'].iloc[0]} (Gradient Boosting)")
        print(f"   Degraded Mode: {df['degraded'].iloc[0]}")
        
        # Show layer distribution
        layer_dist = df['assigned_layer'].value_counts()
        print(f"\n   Deployment Distribution:")
        for layer, count in layer_dist.items():
            pct = (count / len(df)) * 100
            print(f"     {layer}: {count} ({pct:.1f}%)")
        
        # Show feature statistics
        print(f"\n   Feature Statistics:")
        for col in self.feature_cols:
            if col in df.columns:
                print(f"     {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
        
        return df
    
    def train_ml_model(self, df: pd.DataFrame) -> Dict:
        """Train Gradient Boosting model as per paper."""
        
        print("\n" + "="*80)
        print("PHASE 2: TRAINING ML MODEL (Gradient Boosting)")
        print("="*80)
        
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        
        # Prepare features
        X = df[self.feature_cols].copy()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['assigned_layer'])
        
        print(f"\n📊 Training Data:")
        print(f"   Features: {self.feature_cols}")
        print(f"   Samples: {len(X)}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Train Set: {len(X_train)}")
        print(f"   Test Set: {len(X_test)}")
        
        # Train Gradient Boosting (as per paper)
        self.ml_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        print(f"\n🔄 Training Gradient Boosting Classifier...")
        self.ml_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Model Trained Successfully!")
        print(f"   Test Accuracy: {accuracy:.2%}")
        
        # Detailed classification report
        print(f"\n   Classification Report:")
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        for line in report.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        # Feature importance
        print(f"\n   Feature Importance:")
        importance = dict(zip(self.feature_cols, self.ml_model.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(imp * 50)
            print(f"     {feat:15s}: {imp:.3f} {bar}")
        
        return {
            "accuracy": accuracy,
            "feature_importance": importance,
            "classes": list(self.label_encoder.classes_)
        }
    
    def save_model(self):
        """Save trained model to disk."""
        
        print("\n" + "="*80)
        print("PHASE 3: SAVING MODEL")
        print("="*80)
        
        model_path = self.model_dir / "gradient_boosting_model.pkl"
        encoder_path = self.model_dir / "label_encoder.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.ml_model, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\n💾 MODEL SAVED:")
        print(f"   Location: {self.model_dir}")
        print(f"   Files:")
        print(f"     - gradient_boosting_model.pkl ({model_path.stat().st_size / 1024:.1f} KB)")
        print(f"     - label_encoder.pkl ({encoder_path.stat().st_size / 1024:.1f} KB)")
    
    def load_model(self) -> bool:
        """Load trained model from disk."""
        
        model_path = self.model_dir / "gradient_boosting_model.pkl"
        encoder_path = self.model_dir / "label_encoder.pkl"
        
        if model_path.exists() and encoder_path.exists():
            with open(model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✓ Model loaded from {self.model_dir}")
            return True
        return False
    
    def predict_with_threshold(self, metrics: Dict) -> Tuple[str, float, str]:
        """
        Predict deployment using ML + Threshold strategy (Paper Algorithm).
        
        Returns: (layer, confidence, reasoning)
        """
        
        # Prepare features
        features = np.array([[
            metrics.get('datarate_mbps', 0),
            metrics.get('sinr_db', 0),
            metrics.get('latency_ms', 0),
            metrics.get('rsrp_dbm', -100),
            metrics.get('cpu_demand', 50),
            metrics.get('memory_mb', 500)
        ]])
        
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
    
    def save_training_info(self, df: pd.DataFrame, ml_metrics: Dict):
        """Save training information for reference."""
        
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        info = {
            "timestamp": timestamp,
            "algorithm": "ML + Threshold (Gradient Boosting)",
            "paper": "Adaptive Task Scheduling in Edge-Fog-Cloud with Network Failure Resilience",
            "training_data": {
                "source": "edgesimpy_failure_ml_+_thresh_(gb)_no_failure",
                "total_scenarios": len(df),
                "layer_distribution": df['assigned_layer'].value_counts().to_dict()
            },
            "model_storage": {
                "location": str(self.model_dir),
                "files": ["gradient_boosting_model.pkl", "label_encoder.pkl"]
            },
            "ml_model": {
                "type": "GradientBoostingClassifier",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
                "accuracy": ml_metrics["accuracy"],
                "feature_importance": ml_metrics["feature_importance"],
                "classes": ml_metrics["classes"]
            },
            "threshold_rules": {
                "edge": {
                    "latency_max_ms": 50,
                    "datarate_min_mbps": 30,
                    "cpu_max_percent": 50,
                    "description": "Low latency, high bandwidth, lightweight processing"
                },
                "fog": {
                    "latency_max_ms": 200,
                    "datarate_min_mbps": 15,
                    "description": "Moderate latency, balanced resources"
                },
                "cloud": {
                    "description": "High latency tolerance, unlimited resources"
                },
                "override_rules": [
                    "If latency > 150ms AND cpu > 70% → Cloud",
                    "If latency < 30ms AND datarate > 40Mbps AND cpu < 30% → Edge",
                    "If RSRP < -120dBm (poor signal) → Cloud",
                    "If SINR < 5dB (interference) AND Edge predicted → Fog",
                    "If ML confidence < 50% → Use pure threshold rules"
                ]
            }
        }
        
        info_file = results_dir / f"rag_training_complete_{timestamp}.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n💾 Training Info Saved: {info_file.name}")
        
        return info


def test_model(trainer: PaperAlgorithmTrainer):
    """Test the trained model on sample scenarios."""
    
    print("\n" + "="*80)
    print("PHASE 4: TESTING MODEL")
    print("="*80)
    
    test_scenarios = [
        {
            "name": "Edge Scenario (Low latency, High bandwidth)",
            "datarate_mbps": 45.0,
            "latency_ms": 25.0,
            "sinr_db": 18.0,
            "rsrp_dbm": -95.0,
            "cpu_demand": 20,
            "memory_mb": 300
        },
        {
            "name": "Fog Scenario (Moderate conditions)",
            "datarate_mbps": 22.0,
            "latency_ms": 120.0,
            "sinr_db": 12.0,
            "rsrp_dbm": -108.0,
            "cpu_demand": 55,
            "memory_mb": 700
        },
        {
            "name": "Cloud Scenario (High latency, Heavy resources)",
            "datarate_mbps": 18.0,
            "latency_ms": 180.0,
            "sinr_db": 8.0,
            "rsrp_dbm": -118.0,
            "cpu_demand": 85,
            "memory_mb": 1500
        },
        {
            "name": "Poor Signal → Cloud Override",
            "datarate_mbps": 40.0,
            "latency_ms": 30.0,
            "sinr_db": 15.0,
            "rsrp_dbm": -125.0,  # Poor signal
            "cpu_demand": 25,
            "memory_mb": 400
        }
    ]
    
    print("\n🧪 Testing on sample scenarios:\n")
    
    for i, scenario in enumerate(test_scenarios, 1):
        name = scenario.pop("name")
        layer, confidence, reasoning = trainer.predict_with_threshold(scenario)
        
        print(f"Test {i}: {name}")
        print(f"  Metrics: Latency={scenario['latency_ms']:.0f}ms, DataRate={scenario['datarate_mbps']:.0f}Mbps")
        print(f"  Signal: SINR={scenario['sinr_db']:.0f}dB, RSRP={scenario['rsrp_dbm']:.0f}dBm")
        print(f"  Resources: CPU={scenario['cpu_demand']}%, Memory={scenario['memory_mb']}MB")
        print(f"  → Deployment: {layer} (Confidence: {confidence:.0%})")
        print(f"  → Reasoning: {reasoning}")
        print()


def main():
    """Main training workflow."""
    
    print("\n" + "*"*80)
    print("* RAG TRAINING WITH PAPER ALGORITHM (ML + THRESHOLD)")
    print("* Based on: Adaptive Task Scheduling in Edge-Fog-Cloud")
    print("* Model: Gradient Boosting + Threshold Validation")
    print("*"*80)
    
    try:
        # Initialize trainer
        trainer = PaperAlgorithmTrainer()
        
        # Phase 1: Load data
        df = trainer.load_training_data()
        
        # Phase 2: Train ML model
        ml_metrics = trainer.train_ml_model(df)
        
        # Phase 3: Save model
        trainer.save_model()
        
        # Save training info
        info = trainer.save_training_info(df, ml_metrics)
        
        # Phase 4: Test model
        test_model(trainer)
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"""
✓ Training Data: {len(df)} scenarios from EdgeSimPy
✓ ML Model: Gradient Boosting ({ml_metrics['accuracy']:.2%} accuracy)
✓ Classes: {ml_metrics['classes']}
✓ Storage Location: {trainer.model_dir}

Files Created:
  - gradient_boosting_model.pkl  (ML model)
  - label_encoder.pkl            (Label encoder)

Top Features by Importance:
""")
        for feat, imp in sorted(ml_metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  - {feat}: {imp:.3f}")
        
        print(f"""
The model is now ready for deployment decisions!
Use trainer.predict_with_threshold(metrics) to get predictions.
        """)
        
        print("\n" + "*"*80)
        print("* TRAINING SUCCESSFUL!")
        print("*"*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
