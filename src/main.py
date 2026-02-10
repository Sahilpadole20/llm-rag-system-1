# LLM Retrieval-Augmented Generation System

import sys
import os
from pathlib import Path

# Suppress TensorFlow warnings for faster loading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.document_loader import load_network_documents
from retrieval.vector_store import FaissVectorStore
from embeddings.embedding_models import EmbeddingPipeline
from llm.chat_models import NetworkRAGSearch


def retrieve_context(rag_search, query, top_k=3):
    """
    Retrieve context from vector store for the query
    RETRIEVAL PHASE of RAG pipeline
    """
    try:
        results = rag_search.vectorstore.query(query, top_k=top_k)
        context = ""
        for i, result in enumerate(results, 1):
            if isinstance(result, dict) and "metadata" in result and "text" in result["metadata"]:
                context += f"\n[Similar Case {i}]:\n{result['metadata']['text']}\n"
            else:
                context += f"\n[Similar Case {i}]:\n{str(result)}\n"
        return context
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""


def get_llm_deployment_recommendation(rag_search, network_metrics, retrieved_context):
    """
    Use LLM with retrieved context to generate deployment recommendation
    GENERATION PHASE of RAG pipeline
    """
    try:
        # Build prompt with retrieved context from vector store
        prompt = f"""Based on the following similar network configurations from our database:

{retrieved_context}

---

For a network with these metrics:
- Data Rate: {network_metrics['datarate']:.1f} Mbps
- SINR: {network_metrics['sinr']:.2f} dB
- Ping: {network_metrics['ping_ms']:.0f} ms
- RSRP: {network_metrics['rsrp']:.1f} dBm
- RSSI: {network_metrics['rssi']:.1f} dBm
- Jitter: {network_metrics['jitter']:.6f}

Based on the similar cases retrieved from the database above, provide:
1. **Recommended Deployment Layer** (Edge/Fog/Cloud) based on the similar cases
2. **Reasoning** based on patterns from the retrieved similar configurations
3. **Key Performance Considerations**

Be concise and technical."""
        
        # Get LLM response using the context
        response = rag_search.llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return "Unable to generate recommendation"


def main():
    """Main entry point for Network RAG system."""
    try:
        print("=" * 60)
        print("Starting Network Performance RAG System")
        print("=" * 60)

        # Groq API key should be set in environment
        if not os.environ.get("GROQ_API_KEY"):
            print("Warning: GROQ_API_KEY not set in environment")
        
        # Step 1: Load documents
        print("\nStep 1: Loading network documents...")
        docs = load_network_documents("../../data")
        print(f"Loaded {len(docs)} network configuration documents")

        # Step 2: Initialize embedding pipeline
        print("\n Step 2: Initializing embedding pipeline...")
        emb_pipe = EmbeddingPipeline()

        
        # Step 3: Process documents
        print("\n Step 3: Chunking documents...")
        chunks = emb_pipe.chunk_documents(docs)
        
        print("\n Step 4: Generating embeddings...")
        embeddings = emb_pipe.embed_chunks(chunks)
        
        # Step 5: Build vector store
        print("\n Step 5: Building FAISS vector store...")
        store = FaissVectorStore("network_faiss_store")
        
        metadatas = []
        for chunk in chunks:
            metadata = {
                "text": chunk.page_content,
                **chunk.metadata
            }
            metadatas.append(metadata)
        
        import numpy as np
        store.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        store.save()
        print(" Vector store ready!")
        
        # Step 6: Initialize RAG search
        print("\n Step 6: Initializing RAG Search...")
        rag_search = NetworkRAGSearch(store)
        print(" RAG Search system ready!")
        
        # Step 7: Run LLM-based deployment analysis with retrieved context
        print("\n" + "=" * 70)
        print("🤖 LLM DEPLOYMENT RECOMMENDATION WITH RETRIEVED CONTEXT")
        print("=" * 70)
        
        test_scenarios = [
            {
                "name": "High Latency Network",
                "datarate": 25.4,
                "sinr": 9.56,
                "ping_ms": 50521,
                "rsrp": -118.4,
                "rssi": -85.8,
                "jitter": 0.000292
            },
            {
                "name": "Moderate Performance Network",
                "datarate": 15.2,
                "sinr": 10.5,
                "ping_ms": 150,
                "rsrp": -105.0,
                "rssi": -80.0,
                "jitter": 0.0001
            },
            {
                "name": "Low Latency, High Throughput",
                "datarate": 35.8,
                "sinr": 15.8,
                "ping_ms": 25,
                "rsrp": -95.0,
                "rssi": -75.0,
                "jitter": 0.00005
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n📌 Scenario: {scenario['name']}")
            print(f"   Metrics: {scenario['datarate']:.1f}Mbps | {scenario['ping_ms']:.0f}ms ping | {scenario['sinr']:.2f}dB SINR")
            print("-" * 70)
            
            # RETRIEVAL PHASE: Get similar cases from vector store
            query = f"network {scenario['datarate']} Mbps {scenario['ping_ms']} ms latency {scenario['sinr']} dB"
            print(f"\n   📥 RETRIEVAL PHASE: Searching vector store for similar configurations...")
            context = retrieve_context(rag_search, query, top_k=3)
            if context:
                num_cases = len(context.split('[Similar Case')) - 1
                print(f"   ✅ Retrieved {num_cases} similar cases from database")
            else:
                print(f"   ⚠️  No similar cases found, LLM will use general knowledge")
            
            # GENERATION PHASE: Use LLM with retrieved context
            print(f"\n   🤖 GENERATION PHASE: LLM analyzing context and generating recommendation...")
            recommendation = get_llm_deployment_recommendation(rag_search, scenario, context)
            
            print(f"\n   📋 LLM RECOMMENDATION:\n")
            for line in recommendation.split('\n'):
                print(f"   {line}")
            print("=" * 70)
        
        print("\n✅ RAG Pipeline Execution Complete!")
        print("Pipeline: Data → [Retrieval + Context] → LLM Generation → Output")
        
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)