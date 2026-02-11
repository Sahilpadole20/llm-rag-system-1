"""
Retrain all models using the exact Paper Algorithm from shedular_final_paper.ipynb
ML + Threshold (Gradient Boosting) with no failure condition
"""
import pandas as pd
import numpy as np
import pickle
import faiss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

print("=" * 60)
print("🧠 Paper Algorithm: ML + Threshold (Gradient Boosting)")
print("=" * 60)

# Load data - exactly as used in the paper results
data_path = Path(r'c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv')
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} records from simulation results")

# The dataset already has assigned_layer from the simulation
# Features used in the paper (mapped from dataset columns)
feature_cols = ['datarate', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
X = df[feature_cols].values
y = df['assigned_layer'].values

print(f"\n📊 Dataset Distribution:")
print(df['assigned_layer'].value_counts())

# Paper Threshold Parameters
LOW_PING_THRESHOLD = 20  # ms - latency-critical threshold
DATARATE_33RD = np.quantile(df['datarate'], 0.33)
DATARATE_66TH = np.quantile(df['datarate'], 0.66)
MIN_DATARATE = 5e6  # 5 Mbps minimum

print(f"\n📋 Paper Algorithm Thresholds:")
print(f"  - Low Ping Threshold: {LOW_PING_THRESHOLD} ms")
print(f"  - Datarate 33rd percentile: {DATARATE_33RD/1e6:.2f} Mbps")
print(f"  - Datarate 66th percentile: {DATARATE_66TH/1e6:.2f} Mbps")
print(f"  - Min Datarate: {MIN_DATARATE/1e6:.1f} Mbps")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n📌 Classes: {le.classes_}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting (exact model from paper)
print("\n🔧 Training Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 Test Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature Importances
print("\n📊 Feature Importances:")
for feat, imp in zip(feature_cols, model.feature_importances_):
    print(f"  {feat}: {imp:.4f}")

# Create documents for vector search
print("\n📝 Creating document embeddings for RAG...")
documents = []
for _, row in df.iterrows():
    doc = f"Network: datarate={row['datarate']/1e6:.1f}Mbps sinr={row['sinr']:.1f}dB latency={row['latency_ms']:.1f}ms rsrp={row['rsrp_dbm']:.1f}dBm cpu={row['cpu_demand']} mem={row['memory_demand']} → {row['assigned_layer']}"
    documents.append(doc)

# TF-IDF + SVD embeddings
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

n_components = min(100, tfidf_matrix.shape[1] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
embeddings = svd.fit_transform(tfidf_matrix).astype('float32')
print(f"✓ Embeddings shape: {embeddings.shape}")

# Create FAISS index
print("\n🗂️ Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"✓ FAISS index: {index.ntotal} vectors")

# Save all models
store_dir = Path('network_faiss_store')
store_dir.mkdir(exist_ok=True)

# Save ML model
with open(store_dir / 'gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save label encoder
with open(store_dir / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Save scaler
with open(store_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save TF-IDF embedder
with open(store_dir / 'tfidf_embedder.pkl', 'wb') as f:
    pickle.dump({'vectorizer': vectorizer, 'svd': svd}, f)

# Save metadata
with open(store_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump({
        'documents': documents, 
        'df': df.to_dict(),
        'feature_cols': feature_cols,
        'thresholds': {
            'low_ping_threshold': LOW_PING_THRESHOLD,
            'datarate_33rd': DATARATE_33RD,
            'datarate_66th': DATARATE_66TH,
            'min_datarate': MIN_DATARATE
        }
    }, f)

# Save FAISS index
faiss.write_index(index, str(store_dir / 'faiss.index'))

print("\n" + "=" * 60)
print("✅ All models saved successfully!")
print("=" * 60)
print(f"  - gradient_boosting_model.pkl")
print(f"  - label_encoder.pkl")
print(f"  - scaler.pkl")
print(f"  - tfidf_embedder.pkl")
print(f"  - metadata.pkl")
print(f"  - faiss.index")
print("\n🎯 Ready for Streamlit deployment!")
