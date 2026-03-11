"""
Train RAG System with Train/Test Split
========================================
- TRAINING (800 rows): Build FAISS index for RAG retrieval
- TEST (200 rows): Reserved for simulation verification

The test set is NOT in the RAG knowledge base, so we can fairly 
evaluate how well the LLM generalizes to unseen network conditions.
"""
import pandas as pd
import numpy as np
import pickle
import faiss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

print("=" * 60)
print("RAG System: Train/Test Split (800/200)")
print("=" * 60)

# Load data
data_path = Path(r'c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv')
df = pd.read_csv(data_path)
print(f"Total records: {len(df)}")

# Fixed train/test split (first 800 for training, last 200 for testing)
TRAIN_SIZE = 800
TEST_SIZE = 200

# Shuffle with fixed seed for reproducibility
np.random.seed(42)
indices = np.random.permutation(len(df))
train_indices = indices[:TRAIN_SIZE]
test_indices = indices[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

df_train = df.iloc[train_indices].reset_index(drop=True)
df_test = df.iloc[test_indices].reset_index(drop=True)

print(f"\nTRAIN set: {len(df_train)} records (for RAG knowledge base)")
print(f"TEST set: {len(df_test)} records (for simulation verification)")

print(f"\nTRAIN distribution:")
print(df_train['assigned_layer'].value_counts())

print(f"\nTEST distribution:")
print(df_test['assigned_layer'].value_counts())

# Thresholds (from paper algorithm)
DATARATE_33RD = np.quantile(df['datarate'], 0.33)
DATARATE_66TH = np.quantile(df['datarate'], 0.66)
print(f"\nThresholds (computed from full dataset):")
print(f"  33rd percentile: {DATARATE_33RD/1e6:.2f} Mbps")
print(f"  66th percentile: {DATARATE_66TH/1e6:.2f} Mbps")

# Create documents from TRAINING set only
print("\n" + "=" * 60)
print("Building RAG Knowledge Base from TRAINING set...")
print("=" * 60)

documents_train = []
for _, row in df_train.iterrows():
    doc = f"Network: datarate={row['datarate']/1e6:.1f}Mbps sinr={row['sinr']:.1f}dB latency={row['latency_ms']:.1f}ms rsrp={row['rsrp_dbm']:.1f}dBm cpu={row['cpu_demand']} mem={row['memory_demand']} -> {row['assigned_layer']}"
    documents_train.append(doc)

# TF-IDF + SVD embeddings (TRAINING only)
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents_train)

n_components = min(100, tfidf_matrix.shape[1] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=42)
embeddings = svd.fit_transform(tfidf_matrix).astype('float32')
print(f"FAISS embeddings shape: {embeddings.shape}")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index: {index.ntotal} vectors (from {TRAIN_SIZE} training rows)")

# Save models
store_dir = Path('network_faiss_store')
store_dir.mkdir(exist_ok=True)

# Save TF-IDF embedder
with open(store_dir / 'tfidf_embedder.pkl', 'wb') as f:
    pickle.dump({'vectorizer': vectorizer, 'svd': svd}, f)

# Save metadata (TRAINING set only)
with open(store_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump({
        'documents': documents_train, 
        'df': df_train.to_dict(),
        'thresholds': {
            'datarate_33rd': DATARATE_33RD,
            'datarate_66th': DATARATE_66TH
        },
        'train_size': TRAIN_SIZE,
        'test_size': TEST_SIZE
    }, f)

# Save FAISS index
faiss.write_index(index, str(store_dir / 'faiss.index'))

# Save TEST set separately for simulation
with open(store_dir / 'test_data.pkl', 'wb') as f:
    pickle.dump({
        'df': df_test,
        'indices': test_indices.tolist(),
        'size': TEST_SIZE
    }, f)

# Save label encoder (for consistency)
le = LabelEncoder()
le.fit(df['assigned_layer'])
with open(store_dir / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\n" + "=" * 60)
print("SUCCESS! Files saved:")
print("=" * 60)
print(f"  - faiss.index ({TRAIN_SIZE} vectors)")
print(f"  - tfidf_embedder.pkl")
print(f"  - metadata.pkl (training data)")
print(f"  - test_data.pkl ({TEST_SIZE} test samples)")
print(f"  - label_encoder.pkl")
print(f"\nRun 'streamlit run final_rag_simulator.py' to test!")
print("The simulator will use the 200 TEST rows to verify RAG decisions.")
