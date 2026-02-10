#!/usr/bin/env python
"""
COMPLETE RAG PIPELINE WITH ALL STEPS (No Heavy Dependencies)
=============================================================

This script uses TF-IDF for embeddings instead of sentence-transformers
to avoid dependency issues. Works with sklearn + faiss + groq.

PIPELINE STEPS:
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA LOADING    → Load CSV dataset                            │
│  STEP 2: DOCUMENTS       → Create knowledge documents                   │
│  STEP 3: CHUNKING        → Split into smaller text chunks              │
│  STEP 4: EMBEDDINGS      → TF-IDF vectorization (sparse → dense)       │
│  STEP 5: VECTOR STORE    → FAISS index for similarity search           │
│  STEP 6: ML MODEL        → Gradient Boosting for predictions           │
│  STEP 7: GROQ LLM        → Natural language response generation        │
└─────────────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Groq API key should be set in environment or .streamlit/secrets.toml
# export GROQ_API_KEY="your_key_here"


# ============================================================================
# DOCUMENT CLASS
# ============================================================================
class Document:
    """Document container for RAG system."""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# ============================================================================
# STEP 1: DATA LOADER
# ============================================================================
class DataLoader:
    """Load training data from CSV."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.feature_cols = ['datarate_mbps', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
        
    def load(self) -> pd.DataFrame:
        """Load and preprocess CSV."""
        
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        self.df['datarate_mbps'] = self.df['datarate'] / 1e6
        
        print(f"\n📂 Dataset: {self.data_path.name}")
        print(f"   Records: {len(self.df)}")
        
        print(f"\n📊 Feature Statistics:")
        for col in self.feature_cols:
            if col in self.df.columns:
                print(f"   {col:15s}: min={self.df[col].min():.2f}, max={self.df[col].max():.2f}")
        
        print(f"\n🏷️  Class Distribution:")
        for layer, count in self.df['assigned_layer'].value_counts().items():
            pct = (count / len(self.df)) * 100
            print(f"   {layer:6s}: {count:4d} ({pct:.1f}%)")
        
        return self.df


# ============================================================================
# STEP 2: DOCUMENT CREATOR
# ============================================================================
class DocumentCreator:
    """Create knowledge documents from data."""
    
    def create_documents(self, df: pd.DataFrame) -> List[Document]:
        """Create algorithm + scenario documents."""
        
        print("\n" + "="*80)
        print("STEP 2: DOCUMENT CREATION")
        print("="*80)
        
        documents = []
        
        # Algorithm knowledge documents
        algo_docs = [
            Document(
                page_content="""
EDGE-FOG-CLOUD DEPLOYMENT DECISION ALGORITHM using ML + Threshold (Gradient Boosting)

EDGE LAYER - Ultra-low latency:
- Latency threshold: < 50 ms
- Data rate: > 30 Mbps, CPU demand: < 50%
- Use cases: Real-time gaming, AR/VR, autonomous vehicles

FOG LAYER - Balanced performance:
- Latency: 50-200 ms, Data rate: > 15 Mbps
- Use cases: Smart city, healthcare monitoring, retail analytics

CLOUD LAYER - Heavy processing:
- Latency tolerance: > 200 ms
- Use cases: Big data analytics, ML training, archival storage
""",
                metadata={"type": "algorithm", "topic": "deployment_rules"}
            ),
            Document(
                page_content="""
NETWORK QUALITY METRICS:

SINR (Signal-to-Interference-Noise Ratio):
- Excellent: > 20 dB → Edge
- Good: 10-20 dB → Edge/Fog
- Fair: 5-10 dB → Fog
- Poor: < 5 dB → Cloud

RSRP (Reference Signal Received Power):
- Excellent: > -100 dBm → Edge viable
- Good: -100 to -110 dBm → Fog
- Poor: < -120 dBm → Cloud required
""",
                metadata={"type": "algorithm", "topic": "signal_quality"}
            )
        ]
        documents.extend(algo_docs)
        
        # Scenario documents (sample every 10th to reduce size)
        for idx, row in df.iloc[::10].iterrows():
            content = f"""
Scenario #{idx}: Datarate={row['datarate_mbps']:.1f}Mbps, SINR={row['sinr']:.1f}dB, 
Latency={row['latency_ms']:.1f}ms, RSRP={row['rsrp_dbm']:.1f}dBm, 
CPU={row['cpu_demand']}%, Memory={row['memory_demand']}MB
→ Deployed to: {row['assigned_layer']}
"""
            documents.append(Document(
                page_content=content,
                metadata={
                    "type": "scenario",
                    "scenario_id": int(idx),
                    "assigned_layer": row['assigned_layer'],
                    "latency_ms": float(row['latency_ms'])
                }
            ))
        
        print(f"\n📚 Created {len(algo_docs)} algorithm documents")
        print(f"📄 Created {len(documents) - len(algo_docs)} scenario documents")
        print(f"📦 Total: {len(documents)} documents")
        
        return documents


# ============================================================================
# STEP 3: TEXT CHUNKER
# ============================================================================
class TextChunker:
    """Split documents into chunks for better retrieval."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with overlap."""
        
        print("\n" + "="*80)
        print("STEP 3: TEXT CHUNKING")
        print("="*80)
        print(f"\n⚙️  Config: chunk_size={self.chunk_size}, overlap={self.overlap}")
        
        chunks = []
        for doc in documents:
            text = doc.page_content
            start = 0
            chunk_idx = 0
            
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                
                if chunk_text:
                    metadata = doc.metadata.copy()
                    metadata['chunk_index'] = chunk_idx
                    chunks.append(Document(page_content=chunk_text, metadata=metadata))
                    chunk_idx += 1
                
                start = end - self.overlap if end < len(text) else len(text)
        
        print(f"\n📊 Input: {len(documents)} docs → Output: {len(chunks)} chunks")
        
        return chunks


# ============================================================================
# STEP 4: EMBEDDING GENERATOR (TF-IDF)
# ============================================================================
class TFIDFEmbedder:
    """Generate embeddings using TF-IDF + SVD."""
    
    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self.vectorizer = None
        self.svd = None
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and transform texts to dense vectors."""
        
        print("\n" + "="*80)
        print("STEP 4: EMBEDDING GENERATION (TF-IDF + SVD)")
        print("="*80)
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        # TF-IDF vectorization
        print(f"\n🔄 Fitting TF-IDF vectorizer on {len(texts)} texts...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        print(f"   TF-IDF matrix: {tfidf_matrix.shape}")
        
        # Reduce dimensionality with SVD
        n_comp = min(self.n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        print(f"\n🔄 Reducing to {n_comp} dimensions with SVD...")
        
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        embeddings = self.svd.fit_transform(tfidf_matrix)
        
        print(f"✅ Embeddings shape: {embeddings.shape}")
        print(f"   Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")
        
        return embeddings.astype('float32')
    
    def transform(self, text: str) -> np.ndarray:
        """Transform new text to embedding."""
        tfidf = self.vectorizer.transform([text])
        return self.svd.transform(tfidf).astype('float32')


# ============================================================================
# STEP 5: FAISS VECTOR STORE
# ============================================================================
class FAISSVectorStore:
    """FAISS index for similarity search."""
    
    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.documents = []
        
    def build_index(self, embeddings: np.ndarray, documents: List[Document]):
        """Build FAISS index."""
        
        print("\n" + "="*80)
        print("STEP 5: FAISS VECTOR STORE")
        print("="*80)
        
        import faiss
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.documents = documents
        
        print(f"\n🏗️  Index: IndexFlatL2, dim={dim}, vectors={self.index.ntotal}")
    
    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
        
        distances, indices = self.index.search(query_emb.astype('float32'), top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))
        return results
    
    def save(self):
        """Save index and documents."""
        import faiss
        
        faiss.write_index(self.index, str(self.store_dir / "faiss.index"))
        
        with open(self.store_dir / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': [(d.page_content, d.metadata) for d in self.documents]
            }, f)
        
        print(f"\n💾 Saved: faiss.index, metadata.pkl → {self.store_dir}")


# ============================================================================
# STEP 6: ML MODEL
# ============================================================================
class MLModelTrainer:
    """Train Gradient Boosting classifier."""
    
    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.model = None
        self.label_encoder = None
        self.feature_cols = ['datarate_mbps', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train model."""
        
        print("\n" + "="*80)
        print("STEP 6: ML MODEL TRAINING")
        print("="*80)
        
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X = df[self.feature_cols]
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['assigned_layer'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"\n📊 Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
        
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"\n✅ Accuracy: {accuracy:.2%}")
        
        # Feature importance
        print(f"\n📈 Feature Importance:")
        for feat, imp in sorted(zip(self.feature_cols, self.model.feature_importances_), key=lambda x: x[1], reverse=True):
            print(f"   {feat:15s}: {imp:.4f}")
        
        return {"accuracy": accuracy, "classes": list(self.label_encoder.classes_)}
    
    def save(self):
        """Save model."""
        with open(self.store_dir / "gradient_boosting_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.store_dir / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"\n💾 Saved: gradient_boosting_model.pkl, label_encoder.pkl")
    
    def predict(self, features: Dict) -> Tuple[str, float]:
        """Predict layer."""
        X = np.array([[features.get(c, 0) for c in self.feature_cols]])
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0].max()
        return self.label_encoder.inverse_transform([pred])[0], prob


# ============================================================================
# STEP 7: GROQ LLM
# ============================================================================
class GroqLLM:
    """Groq LLM for response generation."""
    
    def __init__(self):
        self.client = None
        self.model = "llama-3.3-70b-versatile"
    
    def initialize(self):
        """Initialize Groq client."""
        
        print("\n" + "="*80)
        print("STEP 7: GROQ LLM INTEGRATION")
        print("="*80)
        
        from groq import Groq
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        print(f"\n✅ Groq initialized: {self.model}")
    
    def generate(self, query: str, context: str, ml_prediction: str = None) -> str:
        """Generate response."""
        
        prompt = f"""You are a network deployment assistant. Based on the context below, answer the query.

CONTEXT:
{context}

{f"ML MODEL PREDICTION: {ml_prediction}" if ml_prediction else ""}

QUERY: {query}

Give a concise recommendation with reasoning."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {e}"


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================
class CompleteRAGPipeline:
    """Orchestrate all 7 steps."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_path = self.base_dir.parent.parent / "data" / "edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv"
        self.store_dir = self.base_dir / "network_faiss_store"
        self.results_dir = self.base_dir / "results"
        
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.df = None
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.embedder = None
        self.vector_store = None
        self.ml_trainer = None
        self.llm = None
    
    def run(self):
        """Execute pipeline."""
        
        print("\n" + "*"*80)
        print("*  COMPLETE RAG TRAINING PIPELINE")
        print("*  Using TF-IDF embeddings + FAISS + Groq LLM")
        print("*"*80)
        
        # Step 1: Load data
        loader = DataLoader(str(self.data_path))
        self.df = loader.load()
        
        # Step 2: Create documents
        creator = DocumentCreator()
        self.documents = creator.create_documents(self.df)
        
        # Step 3: Chunk documents
        chunker = TextChunker(chunk_size=500, overlap=100)
        self.chunks = chunker.split_documents(self.documents)
        
        # Step 4: Generate embeddings
        self.embedder = TFIDFEmbedder(n_components=256)
        texts = [c.page_content for c in self.chunks]
        self.embeddings = self.embedder.fit_transform(texts)
        
        # Step 5: Build vector store
        self.vector_store = FAISSVectorStore(str(self.store_dir))
        self.vector_store.build_index(self.embeddings, self.chunks)
        self.vector_store.save()
        
        # Save embedder
        with open(self.store_dir / "tfidf_embedder.pkl", 'wb') as f:
            pickle.dump({'vectorizer': self.embedder.vectorizer, 'svd': self.embedder.svd}, f)
        
        # Step 6: Train ML model
        self.ml_trainer = MLModelTrainer(str(self.store_dir))
        ml_metrics = self.ml_trainer.train(self.df)
        self.ml_trainer.save()
        
        # Step 7: Initialize LLM
        self.llm = GroqLLM()
        self.llm.initialize()
        
        # Save training info
        self._save_info(ml_metrics)
        
        return True
    
    def _save_info(self, ml_metrics: Dict):
        """Save training metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        info = {
            "timestamp": timestamp,
            "pipeline_steps": {
                "step1_data": {"records": len(self.df)},
                "step2_documents": {"count": len(self.documents)},
                "step3_chunks": {"count": len(self.chunks)},
                "step4_embeddings": {"shape": list(self.embeddings.shape), "method": "TF-IDF + SVD"},
                "step5_vector_store": {"type": "FAISS IndexFlatL2", "vectors": int(self.vector_store.index.ntotal)},
                "step6_ml_model": ml_metrics,
                "step7_llm": {"provider": "Groq", "model": "llama-3.3-70b-versatile"}
            },
            "storage": str(self.store_dir)
        }
        
        with open(self.results_dir / f"rag_training_{timestamp}.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n💾 Training info saved")
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """Query the RAG system."""
        
        # Get query embedding
        query_emb = self.embedder.transform(question)
        
        # Search vector store
        results = self.vector_store.search(query_emb, top_k=top_k)
        
        # Build context
        context = "\n\n".join([f"[{dist:.4f}] {doc.page_content}" for doc, dist in results])
        
        # Generate LLM response
        response = self.llm.generate(question, context)
        
        return {"question": question, "response": response, "docs_retrieved": len(results)}


# ============================================================================
# MAIN
# ============================================================================
def main():
    pipeline = CompleteRAGPipeline()
    
    if not pipeline.run():
        print("❌ Pipeline failed")
        return 1
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*80)
    print(f"""
✅ STEP 1: Loaded {len(pipeline.df)} records
✅ STEP 2: Created {len(pipeline.documents)} documents
✅ STEP 3: Chunked into {len(pipeline.chunks)} pieces
✅ STEP 4: Generated {pipeline.embeddings.shape} embeddings (TF-IDF)
✅ STEP 5: Built FAISS index ({pipeline.vector_store.index.ntotal} vectors)
✅ STEP 6: Trained ML model
✅ STEP 7: Initialized Groq LLM

📁 Storage: {pipeline.store_dir}
   ├── faiss.index
   ├── metadata.pkl
   ├── tfidf_embedder.pkl
   ├── gradient_boosting_model.pkl
   └── label_encoder.pkl
""")
    
    # Test query
    print("\n" + "="*80)
    print("TEST: RAG QUERY WITH GROQ LLM")
    print("="*80)
    
    result = pipeline.query("What deployment layer for low latency and good signal?")
    print(f"\n📤 QUERY: {result['question']}")
    print(f"\n📥 RESPONSE:\n{result['response']}")
    
    print("\n" + "*"*80)
    print("* RAG SYSTEM READY!")
    print("*"*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
