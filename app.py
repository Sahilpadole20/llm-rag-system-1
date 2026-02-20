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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import professor's 4 features
try:
    from src.service_manager import ServiceManager, ServiceType, Priority
    from src.task_scheduler import TaskScheduler
    from src.failure_handler import FailureHandler
    from src.priority_preemption import PriorityPreemptionHandler
    from src.dynamic_requirements import DynamicRequirementsHandler
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

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
        """Hybrid search: numeric filtering + semantic search."""
        import re
        
        results = []
        docs = self.models.get('metadata', {}).get('documents', [])
        df_dict = self.models.get('metadata', {}).get('df', {})
        
        # Try to extract numeric values from query
        datarate_match = re.search(r'datarate[:\s]*(\d+\.?\d*)', query.lower())
        sinr_match = re.search(r'sinr[:\s]*(\d+\.?\d*)', query.lower())
        latency_match = re.search(r'latency[:\s]*(\d+\.?\d*)', query.lower())
        layer_match = re.search(r'(edge|fog|cloud)', query.lower())
        
        # If we have numeric values, use dataframe filtering
        if df_dict and (datarate_match or sinr_match or latency_match or layer_match):
            df = pd.DataFrame(df_dict)
            df['datarate_mbps'] = df['datarate'] / 1_000_000
            
            # Apply filters with tolerance
            mask = pd.Series([True] * len(df))
            
            if datarate_match:
                target = float(datarate_match.group(1))
                mask &= (df['datarate_mbps'] >= target - 2) & (df['datarate_mbps'] <= target + 2)
            
            if sinr_match:
                target = float(sinr_match.group(1))
                mask &= (df['sinr'] >= target - 2) & (df['sinr'] <= target + 2)
            
            if latency_match:
                target = float(latency_match.group(1))
                mask &= (df['latency_ms'] >= target - 30) & (df['latency_ms'] <= target + 30)
            
            if layer_match:
                layer = layer_match.group(1).capitalize()
                mask &= df['assigned_layer'] == layer
            
            filtered = df[mask].head(top_k)
            
            for idx in filtered.index:
                if idx < len(docs):
                    results.append({
                        'content': docs[idx],
                        'metadata': {'index': int(idx)},
                        'distance': 0.0
                    })
            
            if results:
                return results
        
        # Fallback to semantic search
        if 'vectorizer' not in self.models or 'faiss_index' not in self.models:
            return []
        
        tfidf = self.models['vectorizer'].transform([query])
        query_emb = self.models['svd'].transform(tfidf).astype('float32')
        distances, indices = self.models['faiss_index'].search(query_emb, top_k)
        
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(docs):
                results.append({
                    'content': docs[idx],
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

IMPORTANT: Format your response EXACTLY like this with each bullet on a NEW LINE:

• Recommended Layer: [Edge/Fog/Cloud]

• Server ID: [1-7]

• Reasoning: [Why this layer based on network metrics]

Each bullet MUST be on its own separate line. Be direct and concise."""
            
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
    if ADVANCED_FEATURES_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🎯 Predict Deployment", 
            "🔍 Search Knowledge", 
            "💬 Ask Agent",
            "📅 Task Scheduling",
            "⚠️ Failure Handler",
            "⚡ Preemption",
            "📊 Dynamic Changes"
        ])
    else:
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
    
    # =========================================================================
    # TAB 4: TASK SCHEDULING (Professor's Feature #1)
    # =========================================================================
    if ADVANCED_FEATURES_AVAILABLE:
        with tab4:
            st.header("📅 Task Scheduling for XR & eMBB Services")
            st.markdown("""
            **Professor's Requirement #1**: Schedule different types of services:
            - **XR Services**: 15-20 Mbps, 5-20ms latency, prefer Edge
            - **eMBB Services**: 50-100 Mbps, 50-200ms latency, prefer Cloud/Fog
            """)
            
            # Initialize session state for service manager
            if 'service_manager' not in st.session_state:
                st.session_state.service_manager = ServiceManager()
                st.session_state.scheduler = TaskScheduler(st.session_state.service_manager)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Create XR Service")
                xr_throughput = st.slider("XR Throughput (Mbps)", 15.0, 25.0, 18.0, 0.5, key="xr_tp")
                xr_latency = st.slider("XR Max Latency (ms)", 5.0, 20.0, 10.0, 1.0, key="xr_lat")
                xr_users = st.slider("XR Users", 2, 5, 3, key="xr_users")
                
                if st.button("➕ Create XR Service", type="primary"):
                    service = st.session_state.service_manager.create_xr_service(
                        throughput_mbps=xr_throughput,
                        latency_ms=xr_latency,
                        num_users=xr_users
                    )
                    result = st.session_state.scheduler.schedule_service(service)
                    if result.success:
                        st.success(f"✅ {result.message}")
                    else:
                        st.error(f"❌ {result.message}")
            
            with col2:
                st.subheader("Create eMBB Service")
                embb_throughput = st.slider("eMBB Throughput (Mbps)", 50.0, 100.0, 70.0, 5.0, key="embb_tp")
                embb_latency = st.slider("eMBB Max Latency (ms)", 50.0, 200.0, 100.0, 10.0, key="embb_lat")
                embb_users = st.slider("eMBB Users", 10, 50, 25, key="embb_users")
                
                if st.button("➕ Create eMBB Service", type="primary"):
                    service = st.session_state.service_manager.create_embb_service(
                        throughput_mbps=embb_throughput,
                        latency_ms=embb_latency,
                        num_users=embb_users
                    )
                    result = st.session_state.scheduler.schedule_service(service)
                    if result.success:
                        st.success(f"✅ {result.message}")
                    else:
                        st.error(f"❌ {result.message}")
            
            st.divider()
            
            # Server Load Display
            st.subheader("🖥️ Server Load")
            server_load = st.session_state.scheduler.get_server_load_report()
            
            load_df = pd.DataFrame(server_load)
            if not load_df.empty:
                # Color code by utilization
                st.dataframe(
                    load_df[['server_id', 'layer', 'cpu_util_%', 'mem_util_%', 'services_count', 'is_active']],
                    use_container_width=True,
                    hide_index=True
                )
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            metrics = st.session_state.scheduler.metrics.to_dict()
            col_m1.metric("Total Scheduled", metrics['total_scheduled'])
            col_m2.metric("XR Services", metrics['xr_scheduled'])
            col_m3.metric("eMBB Services", metrics['embb_scheduled'])
            col_m4.metric("Avg Latency", f"{metrics['average_latency_ms']:.1f} ms")
            
            if st.button("🔄 Reset All Services"):
                st.session_state.service_manager.reset_all()
                st.session_state.scheduler.reset_metrics()
                st.rerun()
        
        # =========================================================================
        # TAB 5: FAILURE HANDLER (Professor's Feature #2)
        # =========================================================================
        with tab5:
            st.header("⚠️ Node Failure Handling")
            st.markdown("""
            **Professor's Requirement #2**: Handle node failures:
            - Detect when Edge/Fog/Cloud nodes fail
            - Migrate affected services to healthy servers
            - Rebalance load after recovery
            """)
            
            if 'failure_handler' not in st.session_state:
                st.session_state.failure_handler = FailureHandler(
                    st.session_state.service_manager,
                    st.session_state.scheduler
                )
            
            # Server status
            st.subheader("🖥️ Server Status")
            
            servers = st.session_state.service_manager.servers
            cols = st.columns(4)
            
            for i, (server_id, server) in enumerate(servers.items()):
                with cols[i % 4]:
                    if server.is_active:
                        st.success(f"**Server {server_id}** ({server.layer})")
                        st.caption(f"CPU: {server.cpu_utilization:.1f}%")
                    else:
                        st.error(f"**Server {server_id}** ({server.layer}) ❌")
                        st.caption("OFFLINE")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Simulate Failure")
                server_to_fail = st.selectbox(
                    "Select server to fail",
                    options=[s.server_id for s in servers.values() if s.is_active],
                    format_func=lambda x: f"Server {x} ({servers[x].layer})"
                )
                
                if st.button("💥 Simulate Failure", type="primary"):
                    event = st.session_state.failure_handler.simulate_server_failure(server_to_fail)
                    st.warning(f"⚠️ {event.message}")
                    st.info(f"Affected: {len(event.affected_services)} | Migrated: {len(event.migrated_services)} | Failed: {len(event.failed_migrations)}")
                    st.rerun()
            
            with col2:
                st.subheader("🔧 Recovery")
                failed_servers = [s.server_id for s in servers.values() if not s.is_active]
                
                if failed_servers:
                    server_to_recover = st.selectbox(
                        "Select server to recover",
                        options=failed_servers,
                        format_func=lambda x: f"Server {x} ({servers[x].layer})"
                    )
                    
                    if st.button("✅ Recover Server", type="primary"):
                        result = st.session_state.failure_handler.recover_server(server_to_recover)
                        st.success(result['message'])
                        st.rerun()
                else:
                    st.info("All servers are active")
            
            # System Health
            st.divider()
            st.subheader("💚 System Health")
            health = st.session_state.failure_handler.get_system_health()
            
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            col_h1.metric("Status", health['status'].upper())
            col_h2.metric("Active Servers", f"{health['active_servers']}/{health['total_servers']}")
            col_h3.metric("Avg CPU", f"{health['avg_cpu_utilization']:.1f}%")
            col_h4.metric("Failure Events", health['total_failure_events'])
        
        # =========================================================================
        # TAB 6: PREEMPTION (Professor's Feature #3)
        # =========================================================================
        with tab6:
            st.header("⚡ Priority-Based Preemption")
            st.markdown("""
            **Professor's Requirement #3**: Higher priority services preemption:
            - 80% utilization threshold triggers preemption
            - Higher priority services preempt lower priority ones
            - Priority: CRITICAL > HIGH > MEDIUM > LOW
            """)
            
            if 'preemption_handler' not in st.session_state:
                st.session_state.preemption_handler = PriorityPreemptionHandler(
                    st.session_state.service_manager,
                    st.session_state.scheduler
                )
            
            st.subheader("🚨 Create High Priority Service")
            
            col1, col2 = st.columns(2)
            
            with col1:
                priority = st.selectbox(
                    "Priority Level",
                    options=["CRITICAL", "HIGH"],
                    index=0
                )
                service_type = st.selectbox(
                    "Service Type",
                    options=["XR", "eMBB"]
                )
            
            with col2:
                hp_throughput = st.slider("Throughput (Mbps)", 10.0, 80.0, 25.0, 5.0, key="hp_tp")
                hp_users = st.slider("Users", 5, 30, 10, key="hp_users")
            
            if st.button("⚡ Schedule with Preemption", type="primary"):
                pri = Priority.CRITICAL if priority == "CRITICAL" else Priority.HIGH
                
                if service_type == "XR":
                    service = st.session_state.service_manager.create_xr_service(
                        throughput_mbps=hp_throughput,
                        num_users=hp_users,
                        priority=pri
                    )
                else:
                    service = st.session_state.service_manager.create_embb_service(
                        throughput_mbps=hp_throughput,
                        num_users=hp_users,
                        priority=pri
                    )
                
                result = st.session_state.preemption_handler.try_schedule_with_preemption(service)
                
                if result.success:
                    st.success(f"✅ {result.message}")
                    if result.preempted_services:
                        st.warning(f"⚠️ Preempted: {len(result.preempted_services)} services")
                        st.info(f"Rescheduled: {len(result.rescheduled_services)}")
                else:
                    st.error(f"❌ {result.message}")
            
            st.divider()
            
            # Preemption Statistics
            st.subheader("📊 Preemption Statistics")
            stats = st.session_state.preemption_handler.get_preemption_statistics()
            
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Total Preemptions", stats['total_preemptions'])
            col_s2.metric("Edge Preemptions", stats['preemptions_by_layer']['Edge'])
            col_s3.metric("Cloud Preemptions", stats['preemptions_by_layer']['Cloud'])
        
        # =========================================================================
        # TAB 7: DYNAMIC CHANGES (Professor's Feature #4)
        # =========================================================================
        with tab7:
            st.header("📊 Dynamic Requirement Changes")
            st.markdown("""
            **Professor's Requirement #4**: Handle runtime changes:
            - Increase/decrease in throughput requirements
            - Changes in number of users
            - Re-evaluate and migrate if needed
            """)
            
            if 'dynamic_handler' not in st.session_state:
                st.session_state.dynamic_handler = DynamicRequirementsHandler(
                    st.session_state.service_manager,
                    st.session_state.scheduler
                )
            
            # Get running services
            running_services = st.session_state.service_manager.get_running_services()
            
            if running_services:
                st.subheader("📋 Running Services")
                
                service_options = {
                    f"{s.service_id} ({s.service_type.value})": s.service_id 
                    for s in running_services
                }
                
                selected_service_name = st.selectbox(
                    "Select Service to Modify",
                    options=list(service_options.keys())
                )
                selected_service_id = service_options[selected_service_name]
                selected_service = st.session_state.service_manager.get_service(selected_service_id)
                
                if selected_service:
                    st.info(f"Current: {selected_service.throughput_mbps:.1f} Mbps, {selected_service.num_users} users, Server {selected_service.assigned_server}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("📶 Throughput")
                        new_throughput = st.slider(
                            "New Throughput (Mbps)",
                            5.0, 100.0, 
                            float(selected_service.throughput_mbps),
                            1.0,
                            key="new_tp"
                        )
                        if st.button("Update Throughput"):
                            result = st.session_state.dynamic_handler.update_throughput(
                                selected_service_id, new_throughput
                            )
                            if result.success:
                                st.success(f"✅ {result.message}")
                            else:
                                st.error(f"❌ {result.message}")
                            st.rerun()
                    
                    with col2:
                        st.subheader("👥 Users")
                        new_users = st.slider(
                            "New User Count",
                            1, 50,
                            selected_service.num_users,
                            1,
                            key="new_users"
                        )
                        if st.button("Update Users"):
                            result = st.session_state.dynamic_handler.update_users(
                                selected_service_id, new_users
                            )
                            if result.success:
                                st.success(f"✅ {result.message}")
                            else:
                                st.error(f"❌ {result.message}")
                            st.rerun()
                    
                    with col3:
                        st.subheader("⏱️ Latency")
                        new_latency = st.slider(
                            "New Max Latency (ms)",
                            5.0, 200.0,
                            float(selected_service.latency_ms),
                            5.0,
                            key="new_lat"
                        )
                        if st.button("Update Latency"):
                            result = st.session_state.dynamic_handler.update_latency_requirement(
                                selected_service_id, new_latency
                            )
                            if result.success:
                                st.success(f"✅ {result.message}")
                            else:
                                st.error(f"❌ {result.message}")
                            st.rerun()
                
                st.divider()
                
                # Change History
                st.subheader("📜 Change History")
                change_report = st.session_state.dynamic_handler.get_change_report()
                if change_report:
                    change_df = pd.DataFrame(change_report)
                    st.dataframe(change_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No changes recorded yet")
            else:
                st.warning("No running services. Create services in the Task Scheduling tab first.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>RAG Agent | ML + Threshold Algorithm | Based on Edge-Fog-Cloud Paper</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
