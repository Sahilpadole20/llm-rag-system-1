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
DATA_DIR = BASE_DIR / "data"  # Use local data folder

# Set Groq API key from environment or Streamlit secrets
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ============================================================================
# INFRASTRUCTURE CONFIGURATION (from Paper)
# ============================================================================
INFRASTRUCTURE = {
    "Edge": {
        "server_ids": [1, 2, 3, 4],
        "cpu": 16000,
        "memory": 32000,
        "disk": 50000,
        "power_watts": 200,
        "cost_per_hour": 0.50,
        "base_latency_ms": 5
    },
    "Fog": {
        "server_ids": [5, 6],
        "cpu": 64000,
        "memory": 128000,
        "disk": 200000,
        "power_watts": 500,
        "cost_per_hour": 1.00,
        "base_latency_ms": 25
    },
    "Cloud": {
        "server_ids": [7],
        "cpu": 200000,
        "memory": 512000,
        "disk": 1000000,
        "power_watts": 2000,
        "cost_per_hour": 5.00,
        "base_latency_ms": 100
    }
}

# Paper Algorithm Thresholds
LOW_PING_THRESHOLD = 20  # ms
DATARATE_33RD = 9.64e6   # 9.64 Mbps
DATARATE_66TH = 16.60e6  # 16.60 Mbps
SINR_THRESHOLD = 10      # dB


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
    scaler_path = STORE_DIR / "scaler.pkl"
    
    if model_path.exists() and encoder_path.exists():
        with open(model_path, 'rb') as f:
            models['ml_model'] = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            models['label_encoder'] = pickle.load(f)
    
    # Load scaler
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            models['scaler'] = pickle.load(f)
    
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
        self.feature_cols = ['datarate', 'sinr', 'latency_ms', 'rsrp_dbm', 'cpu_demand', 'memory_demand']
        
    def predict_layer(self, features: dict) -> tuple:
        """Predict deployment layer using ML model."""
        if 'ml_model' not in self.models:
            return None, 0.0, "ML model not loaded"
        
        X = np.array([[features.get(c, 0) for c in self.feature_cols]])
        
        # Apply scaler if available
        if 'scaler' in self.models:
            X = self.models['scaler'].transform(X)
        
        pred = self.models['ml_model'].predict(X)[0]
        prob = self.models['ml_model'].predict_proba(X)[0].max()
        layer = self.models['label_encoder'].inverse_transform([pred])[0]
        
        # Apply threshold rules from paper
        layer, reasoning = self._apply_thresholds(features, layer, prob)
        
        return layer, prob, reasoning
    
    def _apply_thresholds(self, features: dict, ml_layer: str, confidence: float) -> tuple:
        """Apply paper algorithm threshold rules (from shedular_final_paper.ipynb)."""
        latency = features.get('latency_ms', 100)
        datarate = features.get('datarate', 15e6)  # in bps
        rsrp = features.get('rsrp_dbm', -110)
        sinr = features.get('sinr', 10)
        cpu = features.get('cpu_demand', 30)
        
        # Paper thresholds
        LOW_PING_THRESHOLD = 20  # ms
        DATARATE_33RD = 9.64e6  # 9.64 Mbps
        DATARATE_66TH = 16.60e6  # 16.60 Mbps
        SINR_THRESHOLD = 10  # dB
        
        reasoning = []
        final_layer = ml_layer
        threshold_applied = False
        
        # Always show ML prediction first
        reasoning.append(f"🤖 ML Prediction: {ml_layer} (confidence: {confidence:.1%})")
        
        # Paper Algorithm Rules - ML + Thresholds:
        # Step 1: Check latency-critical condition
        if latency < LOW_PING_THRESHOLD:
            if datarate < DATARATE_66TH:
                if final_layer != "Edge":
                    final_layer = "Edge"
                    threshold_applied = True
                reasoning.append(f"📏 Threshold: Latency {latency:.1f}ms < 20ms → Edge (latency-critical)")
            else:
                reasoning.append(f"📏 Threshold: Latency {latency:.1f}ms < 20ms but high datarate → ML decides")
        else:
            # Step 2: Apply datarate + SINR thresholds
            if DATARATE_33RD <= datarate < DATARATE_66TH and sinr > SINR_THRESHOLD:
                if final_layer != "Fog":
                    final_layer = "Fog"
                    threshold_applied = True
                reasoning.append(f"📏 Threshold: Mid datarate ({datarate/1e6:.1f} Mbps) + Good SINR ({sinr:.1f}dB > 10) → Fog")
            elif datarate >= DATARATE_66TH:
                if final_layer != "Cloud":
                    final_layer = "Cloud"
                    threshold_applied = True
                reasoning.append(f"📏 Threshold: High datarate ({datarate/1e6:.1f} Mbps ≥ 16.6) → Cloud")
            elif sinr <= SINR_THRESHOLD:
                if final_layer != "Cloud":
                    final_layer = "Cloud"
                    threshold_applied = True
                reasoning.append(f"📏 Threshold: Low SINR ({sinr:.1f}dB ≤ 10) → Cloud")
            else:
                reasoning.append(f"📏 Threshold: No override, using ML prediction")
        
        # Additional safety overrides
        if rsrp < -120:
            if final_layer != "Cloud":
                final_layer = "Cloud"
                threshold_applied = True
            reasoning.append(f"⚠️ Safety: Poor signal (RSRP={rsrp:.1f}dBm < -120) → Cloud fallback")
        
        # Show final decision
        if threshold_applied:
            reasoning.append(f"✅ Final Decision: {final_layer} (ML + Threshold)")
        else:
            reasoning.append(f"✅ Final Decision: {final_layer} (ML only)")
        
        return final_layer, " || ".join(reasoning)
    
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
                content = docs[idx]
                results.append({
                    'content': content,
                    'metadata': {'index': idx},
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
            
            prompt = f"""You are an Edge-Fog-Cloud deployment assistant for autonomous vehicle task scheduling.

Your job is to recommend which layer to deploy a task: Edge, Fog, or Cloud.

DEPLOYMENT LAYERS:
- Edge (Servers 1-4): Low latency (<20ms), limited resources, best for real-time tasks
- Fog (Servers 5-6): Medium latency (25ms), moderate resources, balanced workloads  
- Cloud (Server 7): High latency (100ms), unlimited resources, compute-intensive tasks

DECISION RULES:
- Latency < 20ms AND Datarate < 16.6 Mbps → Deploy to Edge
- Datarate 9.6-16.6 Mbps AND SINR > 10 dB → Deploy to Fog
- Datarate ≥ 16.6 Mbps OR SINR ≤ 10 dB → Deploy to Cloud
- Poor signal (RSRP < -120 dBm) → Deploy to Cloud

SIMILAR NETWORK CONDITIONS FROM DATABASE:
{context}

{f"ML MODEL PREDICTION: {ml_prediction}" if ml_prediction else ""}

USER QUERY: {query}

Provide a clear response:
1. **Recommended Layer**: State Edge, Fog, or Cloud
2. **Reasoning**: Explain why based on network metrics
3. **Server ID**: Suggest specific server (1-7)

Be direct and specific about which layer to deploy."""
            
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
        **Algorithm 1: Enhanced ML with Thresholds Scheduler**
        
        **Require:** Task t, ML model M, thresholds {T₃₃, T₆₆}
        
        1. Extract features: f ← [DR, SINR, ping, RSRP, cpu, mem]
        2. **if** t.ping < 20ms **then**
           - candidates ← Edge servers
        3. **else**
           - layer ← M.predict(f)
           - candidates ← Select layer
        
        **Paper Thresholds:**
        - **Edge:** Latency < 20ms & Datarate < 16.6 Mbps
        - **Fog:** Datarate 9.6-16.6 Mbps & SINR > 10 dB
        - **Cloud:** Datarate ≥ 16.6 Mbps or SINR ≤ 10 dB
        """)
        
        st.divider()
        
        # Infrastructure info
        st.subheader("🏗️ Infrastructure")
        st.markdown("""
        **Edge Servers (ID: 1-4)**
        - CPU: 16,000 | Memory: 32,000
        - Power: 200W | Cost: $0.50/hr
        
        **Fog Servers (ID: 5-6)**
        - CPU: 64,000 | Memory: 128,000
        - Power: 500W | Cost: $1.00/hr
        
        **Cloud Server (ID: 7)**
        - CPU: 200,000 | Memory: 512,000
        - Power: 2,000W | Cost: $5.00/hr
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
            datarate_mbps = st.slider("Data Rate (Mbps)", 5.0, 35.0, 15.0, 0.5)
            latency = st.slider("Latency (ms)", 0.0, 100.0, 20.0, 1.0)
            sinr = st.slider("SINR (dB)", 5.0, 25.0, 15.0, 0.5)
            rsrp = st.slider("RSRP (dBm)", -120.0, -90.0, -105.0, 1.0)
        
        with col2:
            st.subheader("Resource Requirements")
            cpu_demand = st.slider("CPU Demand (%)", 0, 100, 30, 5)
            memory_demand = st.slider("Memory Demand (MB)", 100, 1500, 500, 50)
        
        # Predict button
        if st.button("🚀 Predict Deployment Layer", type="primary", use_container_width=True):
            features = {
                'datarate': datarate_mbps * 1e6,  # Convert Mbps to bps
                'latency_ms': latency,
                'sinr': sinr,
                'rsrp_dbm': rsrp,
                'cpu_demand': cpu_demand,
                'memory_demand': memory_demand
            }
            
            layer, confidence, reasoning = agent.predict_layer(features)
            
            # Get infrastructure details for the predicted layer
            infra = INFRASTRUCTURE.get(layer, INFRASTRUCTURE["Cloud"])
            
            # Display result
            st.divider()
            
            col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
            
            with col_res1:
                if layer == "Edge":
                    st.success(f"## 🟢 {layer}")
                elif layer == "Fog":
                    st.warning(f"## 🟡 {layer}")
                else:
                    st.info(f"## 🔵 {layer}")
                
                st.metric("Confidence", f"{confidence:.1%}")
                st.metric("Server IDs", str(infra['server_ids']))
            
            with col_res2:
                st.subheader("🖥️ Server Resources")
                st.metric("CPU Capacity", f"{infra['cpu']:,}")
                st.metric("Memory Capacity", f"{infra['memory']:,}")
                st.metric("Disk Capacity", f"{infra['disk']:,}")
            
            with col_res3:
                st.subheader("⚡ Cost & Power")
                st.metric("Power Consumption", f"{infra['power_watts']} W")
                st.metric("Cost per Hour", f"${infra['cost_per_hour']:.2f}")
                st.metric("Base Latency", f"{infra['base_latency_ms']} ms")
            
            st.divider()
            
            col_reason, col_input = st.columns(2)
            
            with col_reason:
                st.subheader("📋 ML + Threshold Reasoning")
                # Split reasoning by || and display each part
                for line in reasoning.split(" || "):
                    st.markdown(f"• {line}")
            
            with col_input:
                # Metrics summary
                st.subheader("📊 Input Summary")
                metrics_df = pd.DataFrame([{
                    'Datarate (Mbps)': features['datarate'] / 1e6,
                    'Latency (ms)': features['latency_ms'],
                    'SINR (dB)': features['sinr'],
                    'RSRP (dBm)': features['rsrp_dbm'],
                    'CPU (%)': features['cpu_demand'],
                    'Memory (MB)': features['memory_demand']
                }])
                st.dataframe(metrics_df.T.rename(columns={0: 'Value'}), use_container_width=True)
        
        # Infrastructure Comparison Table (always visible)
        st.divider()
        with st.expander("🏗️ Infrastructure Comparison Table", expanded=False):
            infra_df = pd.DataFrame([
                {"Layer": "Edge", "Servers": "1, 2, 3, 4", "CPU": "16,000", "Memory": "32,000", 
                 "Power (W)": 200, "Cost ($/hr)": 0.50, "Base Latency": "5 ms"},
                {"Layer": "Fog", "Servers": "5, 6", "CPU": "64,000", "Memory": "128,000", 
                 "Power (W)": 500, "Cost ($/hr)": 1.00, "Base Latency": "25 ms"},
                {"Layer": "Cloud", "Servers": "7", "CPU": "200,000", "Memory": "512,000", 
                 "Power (W)": 2000, "Cost ($/hr)": 5.00, "Base Latency": "100 ms"}
            ])
            st.dataframe(infra_df, use_container_width=True, hide_index=True)
    
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
