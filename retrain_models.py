"""Retrain all models for the RAG system."""
import pandas as pd
import numpy as np
import pickle
import faiss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

print('Loading data...')
data_path = Path(r'c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv')
df = pd.read_csv(data_path)
print(f'Loaded {len(df)} records')

# Features
feature_cols = ['datarate', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
X = df[feature_cols].values
y = df['assigned_layer'].values

# 1. Train ML Model
print('Training ML model...')
le = LabelEncoder()
y_encoded = le.fit_transform(y)
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)
print(f'ML Accuracy: {model.score(X, y_encoded):.2%}')

# 2. Create documents and TF-IDF embeddings
print('Creating TF-IDF embeddings...')
documents = []
for _, row in df.iterrows():
    doc = f"Network scenario: datarate={row['datarate']:.0f}, sinr={row['sinr']:.2f}, latency={row['latency_ms']:.2f}ms, rsrp={row['rsrp_dbm']:.2f}dBm, cpu={row['cpu_demand']}, memory={row['memory_demand']}. Assigned to {row['assigned_layer']} layer."
    documents.append(doc)

vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

svd = TruncatedSVD(n_components=min(100, tfidf_matrix.shape[1]-1), random_state=42)
embeddings = svd.fit_transform(tfidf_matrix)
embeddings = embeddings.astype('float32')
print(f'Embeddings shape: {embeddings.shape}')

# 3. Create FAISS index
print('Creating FAISS index...')
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f'FAISS index: {index.ntotal} vectors')

# 4. Save all models
store_dir = Path('network_faiss_store')
store_dir.mkdir(exist_ok=True)

with open(store_dir / 'gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open(store_dir / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open(store_dir / 'tfidf_embedder.pkl', 'wb') as f:
    pickle.dump({'vectorizer': vectorizer, 'svd': svd}, f)
with open(store_dir / 'metadata.pkl', 'wb') as f:
    pickle.dump({'documents': documents, 'df': df.to_dict()}, f)
faiss.write_index(index, str(store_dir / 'faiss.index'))

print('All models saved!')
print('Classes:', le.classes_)
print('Done!')
