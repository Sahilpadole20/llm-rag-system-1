"""
RAG Agent Streamlit Application
================================
Interactive web interface for Edge-Fog-Cloud deployment decisions.

Run with: streamlit run app.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import streamlit as st

# Set page config
st.set_page_config(
    page_title="RAG Deployment Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = Path(__file__).parent
STORE_DIR = BASE_DIR / "network_faiss_store"
DATA_DIR = BASE_DIR.parent.parent / "data"

# Set Groq API key from environment or secrets
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# ============================================================================
# LOAD MODELS (Cached)
# ============================================================================
@st.cache_resource
def load_models():
    """Load all trained models."""
    models = {}
    
    # Load ML model
    model_path = STORE_DIR / "gradient_boosting_model.pkl"
    encoder_path = STORE_DIR / "label_encoder.pkl"
    
    if model_path.exists() and encoder_path.exists():
        with open(model_path, 'rb') as f:
            models['ml_model'] = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            models['label_encoder'] = pickle.load(f)
    
    # Load TF-IDF embedder
    embedder_path = STORE_DIR / "tfidf_embedder.pkl"
    if embedder_path.exists():
        with open(embedder_path, 'rb') as f:
            data = pickle.load(f)
            models['vectorizer'] = data['vectorizer']
            models['svd'] = data['svd']
    
    # Load FAISS index
    faiss_path = STORE_DIR / "faiss.index"
    meta_path = STORE_DIR / "metadata.pkl"
    
    if faiss_path.exists() and meta_path.exists():
        import faiss
        models['faiss_index'] = faiss.read_index(str(faiss_path))
        with open(meta_path, 'rb') as f:
            models['metadata'] = pickle.load(f)
    
    return models


# ============================================================================
# RAG AGENT CLASS
# ============================================================================
class RAGAgent:
    """RAG Agent for deployment decisions."""
    
    def __init__(self, models: dict, api_key: str = None):
        self.models = models
        self.api_key = api_key or GROQ_API_KEY
        self.feature_cols = ['datarate_mbps', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
        
    def predict_layer(self, features: dict) -> tuple:
        """Predict deployment layer using ML model."""
        if 'ml_model' not in self.models:
            return None, 0.0, "ML model not loaded"
        
        X = np.array([[features.get(c, 0) for c in self.feature_cols]])
        
        pred = self.models['ml_model'].predict(X)[0]
        prob = self.models['ml_model'].predict_proba(X)[0].max()
        layer = self.models['label_encoder'].inverse_transform([pred])[0]
        
        # Apply threshold rules
        layer, reasoning = self._apply_thresholds(features, layer, prob)
        
        return layer, prob, reasoning
    
    def _apply_thresholds(self, features: dict, ml_layer: str, confidence: float) -> tuple:
        """Apply paper algorithm threshold rules."""
        latency = features.get('latency_ms', 100)
        datarate = features.get('datarate_mbps', 20)
        rsrp = features.get('rsrp_dbm', -110)
        sinr = features.get('sinr', 10)
        cpu = features.get('cpu_demand', 30)
        
        reasoning = []
        final_layer = ml_layer
        
        # Override rules
        if latency > 150 and cpu > 70:
            final_layer = "Cloud"
            reasoning.append(f"High latency ({latency:.1f}ms) + high CPU ({cpu}%) → Cloud")
        elif latency < 30 and datarate > 40 and cpu < 30:
            final_layer = "Edge"
            reasoning.append(f"Ultra-low latency ({latency:.1f}ms) + high datarate → Edge")
        elif rsrp < -120:
            final_layer = "Cloud"
            reasoning.append(f"Poor signal (RSRP={rsrp:.1f}dBm) → Cloud fallback")
        elif sinr < 5 and ml_layer == "Edge":
            final_layer = "Fog"
            reasoning.append(f"Interference (SINR={sinr:.1f}dB) → Fog instead of Edge")
        
        if not reasoning:
            reasoning.append(f"ML prediction: {ml_layer} (confidence: {confidence:.1%})")
        
        return final_layer, " | ".join(reasoning)
    
    def search_similar(self, query: str, top_k: int = 3) -> list:
        """Search vector store for similar documents."""
        if 'vectorizer' not in self.models or 'faiss_index' not in self.models:
            return []
        
        # Generate query embedding
        tfidf = self.models['vectorizer'].transform([query])
        query_emb = self.models['svd'].transform(tfidf).astype('float32')
        
        # Search
        distances, indices = self.models['faiss_index'].search(query_emb, top_k)
        
        results = []
        docs = self.models['metadata'].get('documents', [])
        
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(docs):
                content, meta = docs[idx]
                results.append({
                    'content': content,
                    'metadata': meta,
                    'distance': float(dist)
                })
        
        return results
    
    def generate_llm_response(self, query: str, context: str, ml_prediction: str = None) -> str:
        """Generate response using Groq LLM."""
        if not self.api_key:
            return "❌ Groq API key not configured. Set GROQ_API_KEY environment variable."
        
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            prompt = f"""You are a network deployment assistant. Based on the context, answer the query.

CONTEXT:
{context}

{f"ML MODEL PREDICTION: {ml_prediction}" if ml_prediction else ""}

QUERY: {query}

Give a concise recommendation with reasoning."""
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ LLM Error: {str(e)}"


# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    # Header
    st.title("🤖 RAG Deployment Agent")
    st.markdown("*Intelligent Edge-Fog-Cloud deployment decisions using ML + Threshold algorithm*")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # Check if models loaded
    if not models:
        st.error("⚠️ No trained models found. Please run training first.")
        st.code("cd src && python rag_pipeline_simple.py")
        return
    
    # Initialize agent
    agent = RAGAgent(models, api_key=st.session_state.get('api_key', GROQ_API_KEY))
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key_input = st.text_input(
            "Groq API Key",
            value=GROQ_API_KEY,
            type="password",
            help="Get your API key from console.groq.com"
        )
        if api_key_input:
            st.session_state['api_key'] = api_key_input
            agent.api_key = api_key_input
        
        st.divider()
        
        # Model info
        st.subheader("📊 Model Info")
        if 'ml_model' in models:
            st.success("✅ ML Model loaded")
        if 'faiss_index' in models:
            st.success(f"✅ FAISS: {models['faiss_index'].ntotal} vectors")
        if 'vectorizer' in models:
            st.success("✅ TF-IDF Embedder loaded")
        
        st.divider()
        
        # Algorithm info
        st.subheader("📖 Algorithm")
        st.markdown("""
        **Threshold Rules:**
        - Edge: Latency < 50ms, Datarate > 30Mbps
        - Fog: Latency 50-200ms
        - Cloud: High latency or poor signal
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predict Deployment", "🔍 Search Knowledge", "💬 Ask Agent"])
    
    # =========================================================================
    # TAB 1: PREDICT DEPLOYMENT
    # =========================================================================
    with tab1:
        st.header("Network Metrics Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Network Conditions")
            datarate = st.slider("Data Rate (Mbps)", 0.0, 50.0, 25.0, 0.5)
            latency = st.slider("Latency (ms)", 0.0, 250.0, 100.0, 1.0)
            sinr = st.slider("SINR (dB)", 0.0, 25.0, 10.0, 0.5)
            rsrp = st.slider("RSRP (dBm)", -135.0, -90.0, -110.0, 1.0)
        
        with col2:
            st.subheader("Resource Requirements")
            cpu_demand = st.slider("CPU Demand (%)", 0, 100, 30, 5)
            memory_demand = st.slider("Memory Demand (MB)", 100, 1500, 500, 50)
        
        # Predict button
        if st.button("🚀 Predict Deployment Layer", type="primary", use_container_width=True):
            features = {
                'datarate_mbps': datarate,
                'latency_ms': latency,
                'sinr': sinr,
                'rsrp_dbm': rsrp,
                'cpu_demand': cpu_demand,
                'memory_demand': memory_demand
            }
            
            layer, confidence, reasoning = agent.predict_layer(features)
            
            # Display result
            st.divider()
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if layer == "Edge":
                    st.success(f"## 🟢 {layer}")
                elif layer == "Fog":
                    st.warning(f"## 🟡 {layer}")
                else:
                    st.info(f"## 🔵 {layer}")
                
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col_res2:
                st.subheader("Reasoning")
                st.write(reasoning)
                
                # Metrics summary
                st.subheader("Input Summary")
                metrics_df = pd.DataFrame([features])
                st.dataframe(metrics_df, use_container_width=True)
    
    # =========================================================================
    # TAB 2: SEARCH KNOWLEDGE
    # =========================================================================
    with tab2:
        st.header("Search Knowledge Base")
        
        search_query = st.text_input(
            "Enter search query",
            placeholder="e.g., low latency edge deployment"
        )
        
        top_k = st.slider("Number of results", 1, 10, 3)
        
        if st.button("🔍 Search", use_container_width=True):
            if search_query:
                results = agent.search_similar(search_query, top_k=top_k)
                
                if results:
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} (Distance: {result['distance']:.4f})"):
                            st.write(result['content'])
                            if result['metadata']:
                                st.json(result['metadata'])
                else:
                    st.warning("No results found")
    
    # =========================================================================
    # TAB 3: ASK AGENT
    # =========================================================================
    with tab3:
        st.header("Ask the RAG Agent")
        
        user_question = st.text_area(
            "Your question",
            placeholder="What deployment layer should I use for a real-time gaming application with 20ms latency requirement?",
            height=100
        )
        
        include_context = st.checkbox("Include retrieved context", value=True)
        
        if st.button("💬 Ask Agent", type="primary", use_container_width=True):
            if user_question:
                with st.spinner("Thinking..."):
                    # Search for context
                    context = ""
                    if include_context:
                        results = agent.search_similar(user_question, top_k=3)
                        context = "\n\n".join([r['content'] for r in results])
                    
                    # Generate response
                    response = agent.generate_llm_response(user_question, context)
                
                st.divider()
                st.subheader("🤖 Agent Response")
                st.write(response)
                
                if include_context and context:
                    with st.expander("📚 Retrieved Context"):
                        st.text(context)
            else:
                st.warning("Please enter a question")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>RAG Agent | ML + Threshold Algorithm | Based on Edge-Fog-Cloud Paper</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
